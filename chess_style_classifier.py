# style_classifier.py
import os
import json
import logging
import numpy as np
import chess
from stockfish import Stockfish
import vertexai
from vertexai.preview.generative_models import GenerativeModel


class StyleClassifier:
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_id: str = "gemini-2.0-flash-001",
        stockfish_path: str = "/opt/homebrew/bin/stockfish",
        sf_depth: int = 15,
    ):
        """
        :param project_id: GCP project for Vertex AI
        :param location:   Vertex AI region
        :param model_id:   Vertex AI model for semantic style inference
        :param stockfish_path: Path to your Stockfish binary
        :param sf_depth:      Depth for engine analysis
        """
        # Initialize Vertex AI LLM
        vertexai.init(project=project_id, location=location)
        self.llm = GenerativeModel(model_id)

        # Stockfish init
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish binary not found at {stockfish_path}")
        
        # Initialize Stockfish via python-stockfish
        self.stockfish = Stockfish(path=stockfish_path, depth=sf_depth)

        # Define the style labels
        self.styles = [
            "Aggressive",
            "Defensive",
            # "Positional",
            "Tactical",
            "Materialistic",
            "Endgame Grinder",
        ]

    def _evaluate_cp(self, board_fen: str) -> float:
        """
        Return centipawn evaluation for a given FEN using python-stockfish.
        """
        self.stockfish.set_fen_position(board_fen)
        eval_info = self.stockfish.get_evaluation()
        # get_board_evaluation returns {'type': 'cp' or 'mate', 'value': int}
        if eval_info["type"] == "cp":
            return float(eval_info["value"])
        # approximate mate as large cp
        return float(10000 if eval_info["value"] > 0 else -10000)

    def extract_features(self, game_history: list, color: str) -> dict:
        """
        Analyze only the opponent's SAN moves to collect pattern counts and CPL stats.
        """
        if color.lower() not in ("white", "black"):
            raise ValueError("color must be 'white' or 'black'")
        color_parity = 1 if color.lower() == "white" else 0
        patterns = {
            "early_queen_moves": 0,
            "checks": 0,
            "captures": 0,
            "sacrifices": 0,
            "quiet_moves": 0,
            "castles_early": 0,
        }
        cpl_list = []
        board = chess.Board()

        for ply_index, san in enumerate(game_history, start=1):
            # evaluate before this move
            fen_before = board.fen()
            eval_before = self._evaluate_cp(fen_before)

            try:
                move = board.parse_san(san)
            except ValueError:
                # Skip illegal or unparseable moves
                continue

            if move not in board.legal_moves:
                # Skip moves that are not legal in the current position
                continue
            
            # Update the board with the move
            board.push(move)
            print(f"Move: {san}, FEN: {board.fen()}")
            fen_after = board.fen()
            
            # only analyze opponent plies
            if ply_index % 2 == color_parity:
                eval_after = self._evaluate_cp(fen_after)
                cp_diff = abs(eval_after - eval_before)
                cpl_list.append(cp_diff)
                evalboard = chess.Board(fen_before)
                is_capture = evalboard.is_capture(move)
                is_check   = evalboard.gives_check(move)
                is_queen   = san.lower().startswith("q")
                is_castle  = san in ("O-O", "O-O-O")

                if is_queen and ply_index <= 10:
                    patterns["early_queen_moves"] += 1
                if is_capture:
                    patterns["captures"] += 1
                if is_check:
                    patterns["checks"] += 1
                if is_castle and ply_index <= 8:
                    patterns["castles_early"] += 1
                # sacrifice if capture worsens eval by >100 cp
                if is_capture and (eval_after - eval_before) < -100:
                    patterns["sacrifices"] += 1
                # quiet move = non-capture, non-check
                if not (is_capture or is_check):
                    patterns["quiet_moves"] += 1

        features = patterns.copy()
        features["avg_cpl"] = float(np.mean(cpl_list)) if cpl_list else 0.0
        features["std_cpl"] = float(np.std(cpl_list)) if cpl_list else 0.0
        return features

    def rule_based_probabilities(self, features: dict, game_phase: str) -> dict:
        """
        Convert raw feature counts into a style‐score vector, then softmax to probabilities.
        """
        # Helper for normalization to reduce bias from absolute counts
        def norm(x, base=1.0):
            avg_cpl = features.get("avg_cpl", 0)
            return x / (avg_cpl + base) if avg_cpl > 0 else x

        # --- Phase Weights (tunable) ---
        phase_weight = {
            "opening": 0.20,
            "middlegame": 0.50,
            "endgame": 0.30,
        }

        # --- Style Score Calculations ---
        scores = {
            "Aggressive": (
                phase_weight["opening"] * (
                    1.2 * norm(features.get("early_queen_moves", 0), 2)
                    + 1.5 * norm(features.get("checks", 0), 2)
                    + 2.0 * norm(features.get("sacrifices", 0), 2)
                    - 0.5 * norm(features.get("quiet_moves", 0), 2)
                ) +
                phase_weight["middlegame"] * (
                    1.7 * norm(features.get("checks", 0), 2)
                    + 2.2 * norm(features.get("sacrifices", 0), 2)
                    + 1.2 * norm(features.get("captures", 0), 2)
                    - 0.7 * norm(features.get("quiet_moves", 0), 2)
                ) +
                phase_weight["endgame"] * (
                    1.0 * norm(features.get("checks", 0), 2)
                    + 0.8 * norm(features.get("sacrifices", 0), 2)
                )
            ),

            "Defensive": (
                phase_weight["opening"] * (
                    1.5 * norm(features.get("castles_early", 0), 2)
                    + 1.0 * norm(features.get("quiet_moves", 0), 2)
                    - 1.0 * norm(features.get("sacrifices", 0), 2)
                    - 0.8 * norm(features.get("early_queen_moves", 0), 2)
                ) +
                phase_weight["middlegame"] * (
                    1.3 * norm(features.get("quiet_moves", 0), 2)
                    + 1.2 * (100 / (features.get("avg_cpl", 100) + 10))
                    - 1.0 * norm(features.get("sacrifices", 0), 2)
                    - 0.5 * norm(features.get("checks", 0), 2)
                ) +
                phase_weight["endgame"] * (
                    1.2 * (100 / (features.get("std_cpl", 100) + 10))
                    + 0.8 * norm(features.get("quiet_moves", 0), 2)
                    - 0.5 * norm(features.get("checks", 0), 2)
                )
            ),

            # "Positional": (
            #     phase_weight["opening"] * (
            #         1.1 * norm(features.get("quiet_moves", 0), 2)
            #         + 0.8 * norm(features.get("castles_early", 0), 2)
            #         - 0.5 * norm(features.get("sacrifices", 0), 2)
            #     ) +
            #     phase_weight["middlegame"] * (
            #         1.5 * norm(features.get("quiet_moves", 0), 2)
            #         + 1.2 * (100 / (features.get("avg_cpl", 100) + 10))
            #         - 0.8 * norm(features.get("sacrifices", 0), 2)
            #     ) +
            #     phase_weight["endgame"] * (
            #         1.3 * (100 / (features.get("std_cpl", 100) + 10))
            #         + 1.0 * norm(features.get("quiet_moves", 0), 2)
            #         - 0.5 * norm(features.get("checks", 0), 2)
            #     )
            # ),

            "Tactical": (
                phase_weight["opening"] * (
                    1.3 * norm(features.get("sacrifices", 0), 2)
                    + 1.1 * norm(features.get("checks", 0), 2)
                    + 1.0 * norm(features.get("captures", 0), 2)
                    + 0.5 * (features.get("avg_cpl", 0) > 50)
                ) +
                phase_weight["middlegame"] * (
                    2.0 * norm(features.get("sacrifices", 0), 2)
                    + 1.5 * norm(features.get("checks", 0), 2)
                    + 1.2 * norm(features.get("captures", 0), 2)
                    + 0.8 * (features.get("avg_cpl", 0) > 60)
                    - 0.7 * norm(features.get("quiet_moves", 0), 2)
                ) +
                phase_weight["endgame"] * (
                    1.0 * norm(features.get("sacrifices", 0), 2)
                    + 0.8 * norm(features.get("checks", 0), 2)
                    + 0.5 * norm(features.get("captures", 0), 2)
                )
            ),

            "Materialistic": (
                phase_weight["opening"] * (
                    1.0 * norm(features.get("captures", 0), 2)
                    - 1.5 * norm(features.get("sacrifices", 0), 2)
                ) +
                phase_weight["middlegame"] * (
                    1.8 * norm(features.get("captures", 0), 2)
                    - 2.2 * norm(features.get("sacrifices", 0), 2)
                    + 0.5 * (features.get("avg_cpl", 100) < 40)
                ) +
                phase_weight["endgame"] * (
                    1.5 * norm(features.get("captures", 0), 2)
                    - 1.0 * norm(features.get("sacrifices", 0), 2)
                    + 0.8 * (features.get("avg_cpl", 100) < 35)
                )
            ),

            "Endgame Grinder": (
                # Focus on accuracy, quiet moves, and low CPL in the endgame
                2.0 * phase_weight["endgame"] * (
                    1.5 * (100 / (features.get("std_cpl", 100) + 10))
                    + 1.2 * (100 / (features.get("avg_cpl", 100) + 10))
                    + 1.0 * norm(features.get("quiet_moves", 0), 2)
                    - 0.8 * norm(features.get("sacrifices", 0), 2)
                    - 0.5 * norm(features.get("checks", 0), 2)
                ) +
                0.5 * phase_weight["middlegame"] * (
                    1.0 * (100 / (features.get("avg_cpl", 100) + 10))
                    + 0.8 * norm(features.get("quiet_moves", 0), 2)
                    - 0.5 * norm(features.get("sacrifices", 0), 2)
                )
            ),
        }

        # Normalize the final scores so they sum to 1 (or use softmax)
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {style: score / total_score for style, score in scores.items()}
        else:
            # Handle case with zero scores (e.g., very short game)
            num_styles = len(scores)
            probabilities = {style: 1.0 / num_styles for style in scores.keys()}

        return probabilities

    def build_style_prompt(self, features: dict, game_phase: str, game_history:str, color: str) -> str:
        """
        Summarize features into a prompt for the LLM to return style probabilities.
        """
        stats = "\n".join(f"- {k.replace('_',' ')}: {v:.2f}" for k, v in features.items())
        return (
            f"You are a chess psychologist and expert coach. Here are summary statistics of an opponent’s play during the {game_phase} phase:\n"
            f"{stats}\n\n"
            f"Here's the game history in SAN format:\n"
            f"{game_history}\n"
            f"The opponent's color is {color}.\n\n"
            f"Classify this opponent into the following styles with probabilities:\n"
            f"{', '.join(self.styles)}.\n\n"
            "Respond in JSON format, for example:\n"
            "{\n"
            '  "Aggressive": 0.30,\n'
            '  "Defensive": 0.10,\n'
            '  "Tactical": 0.15,\n'
            '  "Materialistic": 0.15,\n'
            '  "Endgame Grinder": 0.10\n'
            "}\n"
            "Make sure to include all styles, even if the probability is 0.0.\n"
            "The sum of all probabilities should be 1.0.\n"
            "Do not include any other text or explanations.\n"
            "Make sure to return a valid JSON object."
        )

    def _validate_style_probs(self, probs: dict) -> dict:
        """
        Ensure all style keys are present, values are floats in [0,1], and sum to 1.
        If missing, fill with 0.0 and renormalize.
        """
        print("LLM returned probs:", probs)  # Debug: See what the LLM actually returned
        # Ensure all styles are present
        out = {s: float(probs.get(s, 0.0)) for s in self.styles}
        # Clamp values to [0,1]
        out = {k: min(max(v, 0.0), 1.0) for k, v in out.items()}
        total = sum(out.values())
        if total == 0:
            # Avoid division by zero, assign uniform
            n = len(self.styles)
            out = {s: 1.0 / n for s in self.styles}
        else:
            out = {k: v / total for k, v in out.items()}
        return out

    def call_llm_for_style_probs(self, prompt: str) -> dict:
        """
        Call Vertex AI LLM and parse its JSON response.
        """
        response = self.llm.generate_content(prompt)
        print("LLM response:", response.text)  # Debug: See the raw LLM response
        probs = parse_llm_style_response(response.text.strip())
        return self._validate_style_probs(probs) if probs else self._validate_style_probs({})

    def classify(
        self,
        game_phase: str,
        game_history: list,
        opponent_color: str  # "white" or "black"
    ) -> (str, dict): # type: ignore
        """
        End-to-end style classification:
        1. Determine which plies belong to the opponent.
        2. Store full history and parity.
        3. Extract features from only opponent moves.
        4. Compute rule-based & LLM-based probabilities and fuse.
        """

        # 1) Extract features
        features = self.extract_features(game_history, color=opponent_color)

        # 2) Rule-based probabilities
        P_rule = self.rule_based_probabilities(features, game_phase=game_phase)
        print("Rule-based probabilities:", P_rule)

        # 3) LLM-based probabilities
        prompt = self.build_style_prompt(features, game_phase, game_history, opponent_color)
        print("LLM prompt:", prompt)  # Debug: See the prompt sent to the LLM
        P_llm = self.call_llm_for_style_probs(prompt)
        print("LLM-based probabilities:", P_llm)

        # 4) Ensemble fusion
        alpha = 0.6
        P_final = {
            s: alpha * P_rule.get(s, 0) + (1 - alpha) * P_llm.get(s, 0)
            for s in self.styles
        }

        # Pick the top style
        best_style = max(P_final, key=P_final.get)
        return best_style, P_final


def parse_llm_style_response(response_string: str) -> dict | None:
    """
    Parses a JSON string containing chess style probabilities from an LLM response.

    Args:
        response_string: The string potentially containing the JSON data.

    Returns:
        A dictionary with style names as keys and probabilities as values,
        or None if parsing fails or the format is incorrect.
    """
    try:
        # Clean up potential markdown or extra text around the JSON
        # Find the start and end of the JSON object
        start_index = response_string.find('{')
        end_index = response_string.rfind('}')

        if start_index == -1 or end_index == -1:
            logging.error("Could not find JSON object markers '{' or '}' in the response.")
            return None

        json_string = response_string[start_index : end_index + 1]

        # Parse the extracted JSON string
        style_probabilities = json.loads(json_string)

        # Basic validation (optional but recommended)
        if not isinstance(style_probabilities, dict):
            logging.error("Parsed data is not a dictionary.")
            return None
        if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in style_probabilities.items()):
            logging.error("Dictionary values are not all strings (keys) and numbers (values).")
            return None
        # You could add more validation, e.g., check if probabilities sum close to 1

        return style_probabilities

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON: {e}")
        logging.error(f"Original string segment: {json_string if 'json_string' in locals() else response_string}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during parsing: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    classifier = StyleClassifier(project_id="chessai-453307")

    # game_history = ["e4", "e5", "Qf3", "Qf6", "Bc4", "Nf6", "d3", "Be7"]
    # game_history = [
    #     "e4", "e5", "Qf3", "Nf6", "Bc4", "d5", "exd5", "Bc5", "d4", "exd4",
    #     "Qe2+", "Qe7", "Nf3", "Qxe2+", "Bxe2", "O-O", "O-O", "Re8", "Re1", "Bg4",
    #     "Nbd2", "Bd6", "Nxd4", "Bxe2", "Nxe2", "Nxd5", "Nf3", "Nf4", "Nxf4", "Rxe1+",
    #     "Nxe1", "Bxf4", "Bxf4", "Nc6", "Bxc7", "Rd8", "c3", "Rd7", "Bf4", "b5",
    #     "Nf3", "f6", "a4", "bxa4", "Rxa4", "Rd1+", "Ne1", "Rxe1#"]    
    # game_history = [
    #     "e4",  # White opens aggressively
    #     "e5",
    #     "f4",  # King's Gambit!
    #     "exf4",
    #     "Nf3",  # Developing with attack in mind
    #     "g5",  # Black's common response, trying to hold the pawn
    #     "Bc4",  # Developing the bishop, putting pressure
    #     "Bg7",
    #     "h4",  # Continuing the attack on the kingside
    #     "gxh4",
    #     "Rh3",  # Bringing the rook into the attack early
    #     "d6",
    #     "Nc3",  # Developing another piece, keeping options open
    #     "c6",
    #     "d4",  # Opening the center, creating more attacking lines
    #     "Bg4",
    #     "Bxf4", # Sacrificing the bishop to open lines
    #     "Bxf3",
    #     "Qxf3", # Queen joins the attack
    #     "Nf6",
    #     "O-O-O", # White castles queenside, going all-in on the attack
    #     "Qe7",
    #     "e5",  # Pawn push to open lines further
    #     "dxe5",
    #     "Nb5",  # Jumping into Black's territory
    #     "a6",
    #     "Nd6+", # Forcing the king
    #     "Kd8",
    #     "Rxh4", # Taking advantage of the open h-file
    #     "Nxe5",
    #     "Nxe5", # Exchanging to simplify into an attack
    #     "Qxe5",
    #     "Bxg5", # Sacrificing another piece for a strong attack
    #     "hxg5",
    #     "Qxg5+", # Continuing the pressure
    #     "Ke8",
    #     "Rh8#", # Checkmate!
    # ]

    game_history = [
        "e4",
        "e5",
        "Nf3",  # Developing quickly and aggressively
        "Nc6",
        "Bc4",  # Italian Game - allows for sharp attacks
        "Nf6",
        "Ng5",  # The Fried Liver Attack - a very direct and forcing line
        "d5",
        "exd5", # Accepting the pawn sacrifice to open lines
        "Nxd5",
        "Nxf7", # The key sacrifice, aiming to expose the Black king
        "Kxf7",
        "Qf3+", # Forcing the king and continuing the attack
        "Ke6",
        "Nc3",  # Developing with attack
        "Bc5",
        "d3",   # Creating a safe square for the knight
        "h6",
        "Nge4+",# Another forcing move
        "Kd6",
        "Nb5+", # More checks and forcing moves
        "Kc6",
        "Bf4",  # Developing with pressure
        "a6",
        "Na4+", # Continuing the attack
        "Kb5",
        "c3+",  # Opening lines to the king
        "Ka5",
        "b4+",  # More forcing moves
        "Bxb4", # Taking advantage of the open line
        "axb4",
        "Qe3+", # Direct attack on the king
        "Kb5",
        "a4+",  # Keeping the king exposed
        "Ka5",
        "Qd2+", # More checks
        "Ka6",
        "Qa5#", # Checkmate with the Queen
    ]

#     game_history = [
#     "e4",
#     "e5",
#     "Bc4",  # Immediate threat to f7
#     "Nc6",
#     "Qf3",  # Directly attacking f7, preparing checkmate
#     "Nf6",  # Defending f7
#     "Qxf7#", # Checkmate!
# ]
    style, dist = classifier.classify(
        game_phase="endgame",
        game_history=game_history,
        opponent_color="white"
    )

    print("Predicted style:", style)
    print("Style distribution:", dist)
