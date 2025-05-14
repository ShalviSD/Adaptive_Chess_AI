#!/usr/bin/env python3
import os
import random
import re

import chess
from stockfish import Stockfish
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,  # Set level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChessAI:
    def __init__(self, 
                 project_id: str, 
                 location: str = "us-central1", 
                 model_id: str = "gemini-2.0-flash-001",
                 stockfish_path: str = "/opt/homebrew/bin/stockfish"):
        """
        Initialize the ChessAI with Google Cloud Platform - GCP project details and the specified Vertex AI model.

        :param project_id: Your GCP project ID.
        :param location: The location for Vertex AI resources (default is 'us-central1').
        :param model_id: The model identifier for the Vertex AI text generation model.
        """
        # Initialize Vertex AI with your project settings
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_id)
        
        # Stockfish init
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish binary not found at {stockfish_path}")
        self.stockfish_exec = stockfish_path


    def _get_top_moves(self, fen: str, elo: int, num_moves: int = 5, ) -> list:
        """Use Stockfish to return top moves in SAN notation."""
        sf = Stockfish(self.stockfish_exec)
        sf.update_engine_parameters({"UCI_Elo": elo})
        sf.set_fen_position(fen)
        raw = sf.get_top_moves(num_moves)
        board = chess.Board(fen)
        # Convert UCI moves to SAN
        san_moves = []
        for entry in raw:
            uci = entry['Move']
            move = chess.Move.from_uci(uci)
            try:
                san_moves.append(board.san(move))
            except ValueError:
                san_moves.append(uci)
        return san_moves
    
    def _get_game_phase(self, game_history: list) -> str:
        """
        Determine the phase of the game based on the number of moves in the game history.
        """
        num_moves = len(game_history)
        if num_moves < 10:
            return "Opening"
        elif num_moves < 30:
            return "Middlegame"
        else:
            return "Endgame"
    
    def _get_player_category(self, elo: int) -> str:
        """
        Convert ELO rating to a player category.
        """
        if elo < 1200:
            return "Beginner"
        elif elo < 1600:
            return "Intermediate"
        elif elo < 2000:
            return "Advanced"
        else:
            return "Expert"
    
    def move_prompt(self, fen: str, game_history: list, elo: int, valid_moves: list, turn: str, top_moves: list, player_style:str=None) -> str:
        """
        Create a prompt message for the Vertex AI model using the chess board's FEN notation.

        :param board_fen: Chess board position in FEN notation.
        :return: A formatted prompt string for the model.
        """
        style_text = f"\n- **Player Style**: {player_style}" if player_style else ""
        game_phase = self._get_game_phase(game_history)
        player_category = self._get_player_category(elo)
        if game_phase == "Opening":
            if player_category == "Beginner":
                game_phase_prompt = "Opening phase, focus on basic development and king safety and controlling center."
            elif player_category == "Intermediate":
                game_phase_prompt = "Opening phase, focus on development, controlling center, use standard lines and avoid early traps."
            elif player_category == "Advanced":
                game_phase_prompt = "Opening phase, focus on creative openings, controlling center, and preparing for tactical opportunities."
            else:
                game_phase_prompt = "Opening phase, focus on creative & advanced opening theory, controlling center, and preparing for tactical opportunities for dynamic gameplay."
        elif game_phase == "Middlegame":
            if player_category == "Beginner":
                game_phase_prompt = "Middlegame phase, focus on basic tactics and avoid blunders."
            elif player_category == "Intermediate":
                game_phase_prompt = "Middlegame phase, focus on tactics, piece activity, and pawn structure."
            elif player_category == "Advanced":
                game_phase_prompt = "Middlegame phase, focus on tactics, piece activity, pawn structure, and advanced planning."
            else:
                game_phase_prompt = "Middlegame phase, focus on exploiting tactics, piece activity, pawn structure, and advanced planning for dynamic gameplay."
        elif game_phase == "Endgame":
            if player_category == "Beginner":
                game_phase_prompt = "Endgame phase, focus on basic endgame principles, king activation."
            elif player_category == "Intermediate":
                game_phase_prompt = "Endgame phase, focus on basic endgame principles and pawn promotion, king activation, piece coordination"
            elif player_category == "Advanced":
                game_phase_prompt = "Endgame phase, focus on advanced endgame techniques and pawn promotion, king activation, piece coordination."
            else:
                game_phase_prompt = "Endgame phase, focus on advanced endgame techniques and pawn promotion for dynamic gameplay, king activation, piece coordination, avoid blunders at all cost."
        prompt_text = f"""
        You are a chess player with an ELO rating of {elo}. Analyze the given position step by step, considering game phase, tactics, strategy, and potential threats at your skill level.  

            Current Game Context:  
            - **Player ELO**: {elo}
            {style_text}
            - **Your Color**: {turn} 
            - **Game History (SAN)**: {game_history} 
            - **Current Position (FEN)**: {fen}
            - **Game Phase**: {game_phase}
            - **Last Move**: {game_history[-1] if game_history else "N/A"}
            - **Legal Moves**: {valid_moves}
            - **Top Moves (Stockfish)**: {top_moves}

            ### Instructions:
            1. **Step-by-step Analysis:**  
            - Evaluate opponent's last move and its implications. 
            - Identify any immediate threats by opponent's last move and see if you can defend or exploit tactical opportunities.
            - Consider the game phase (Opening, Middlegame, Endgame) and adjust your strategy accordingly.
            - {game_phase_prompt}
            - Analyze each move suggested by Stockfish and explain why they are good or bad in this position one by one. Remember you are a {elo}-rated player and the moves suggested by stockfish should align with your level.
            - Determine a reasonable move based on principles suitable for a {elo}-rated player. Ensure to find YOUR best and safe move in the position.
            - AVOID BLUNDERS!

            2. **Output the best move in Standard Algebraic Notation (SAN) on a new line, exactly in format:**  
            **`CHESS Move: <your move>`** 
        """
        return prompt_text

    def extract_move(self, response: str) -> str:
        match = re.search(
            r'CHESS Move:\s*((?:O-O(?:-O)?|[KQRBN]?[a-h1-8]{0,2}x?[a-h][1-8](?:=[QRBN])?[+#]?))',
            response
        )
        logger.info("############################")
        logger.info(match)
        logger.info("############################")
        if match:
            move_str = match.group(1).strip()
            move_str = move_str.replace('0-0', 'O-O').replace('0-0-0', 'O-O-O')
            return move_str
        return ''

    def generate_move(self, fen: str, game_history: list, elo: int, valid_moves: list, max_attempts=3, player_style: str = None) -> str:
        """
        Generate a response from the Vertex AI model based on a given prompt.

        :param board_fen: Chess board position in FEN notation.
        :return: The response text from the model.
        """
        board = chess.Board(fen)
        turn = "white" if board.turn else "black"    
        
        # Convert the ELO rating to a skill level for Stockfish suggestions
        top_moves = self._get_top_moves(fen, elo, num_moves=5)
        
        # Format the prompt based on the chess board position
        prompt_message = self.move_prompt(
            fen=fen,
            game_history=game_history,
            elo=elo,
            valid_moves=valid_moves,
            turn=turn,
            top_moves=top_moves,
            player_style=player_style
        )

        for attempt in range(max_attempts):
            logger.info("############################")
            logger.info("DEBUG: Attempting to generate a move...")
            logger.info(f"Attempt {attempt + 1}")
            logger.info("############################")
            logger.info("DEBUG: Prompt being sent to the model:")
            logger.info(prompt_message)
            logger.info("############################")
            # Use the model to predict the best move and explanation
            response = self.model.generate_content(prompt_message)
            logger.info("############################")
            logger.info("DEBUG: Response generated model:")
            logger.info(response.text)
            logger.info("############################")
            move_str = self.extract_move(response.text)
            # move_str = self.extract_move(result)
            try:
                # Check if the move is valid
                board.parse_san(move_str)
                logger.info(f"Valid move found: {move_str}")
                return move_str
            except ValueError:
                logger.info(f"Invalid move: {move_str}")
                prompt_message += f"\nInvalid move: '{move_str}'. Generate a valid SAN move starting with 'CHESS Move: <your move>'."

    def analyze_move_prompt(self, fen_before: str, fen_after: str, move_san: str, elo: int, game_history: list) -> str:
        """
        Generate a prompt for the LLM to analyze a move, using FENs before/after, SAN, ELO, and style.
        """
        game_phase = self._get_game_phase(game_history)
        # Convert the ELO rating to a skill level for Stockfish suggestions
        top_moves = self._get_top_moves(fen_before, elo, num_moves=5)
        [best_move] = self._get_top_moves(fen_before, elo, num_moves=1)
        return (
            f"You are a chess coach. Analyze the move '{move_san}' for a player rated {elo}\n"
            f"Position before the move (FEN): {fen_before}\n"
            f"Position after the move (FEN): {fen_after}\n"
            f"Game phase: {game_phase}\n"
            f"Game history (SAN): {game_history}\n"
            f"Player color: {'white' if chess.Board(fen_before).turn else 'black'}\n"
            "Analyze the purpose, strengths, weaknesses, and possible intentions behind this move."
            "Assess move quality if it was a 'Good Move', 'Inaccuracy', 'Average Move', 'Great Move', 'Best Move', 'Mistake', 'BLUNDER' in the position for this rating, style and game phase."
            "Provide a detailed analysis of the move, including tactical and strategic considerations. Keep it less than 10 sentences.\n"
            f"Based on the top moves from Stockfish: {top_moves} make alternative suggestions and analysis of each move in {game_phase} phase. Focus on the best move {best_move}\n."
            "Keep the output text well-structured well-formatted with sub-bullets and appropriate indentations and easy to read.\n"
            "Provide output in the following format ONLY (Output nothing else!):\n"
            "1. **Move Quality**: <quality>\n"
            "2. **Move Analysis**: <Concise move analysis>\n"
            f"3. **Best Move**: {best_move}\n"
            f"4. **Alternative Top Moves**: {' '.join(top_moves)}\n"
            "5. **Top Move Analysis**: <Sub-bullet list of each move and its analysis with best move first>\n"
        )

    def analyze_move(self, fen_before: str, fen_after: str, move_san: str, elo: int, game_history: list) -> str:
        prompt = self.analyze_move_prompt(fen_before, fen_after, move_san, elo, game_history=game_history)
        # Use the model to analyze the move
        logger.info("############################")
        logger.info("DEBUG: Generating analysis response...")
        logger.info("############################")
        logger.info("DEBUG: Prompt being sent to the model:")
        logger.info(prompt)
        logger.info("############################")
        response = self.model.generate_content(prompt)
        logger.info("############################")
        logger.info("DEBUG: Analysis response generated:")
        logger.info(response.text)
        logger.info("############################")
        # Extract the analysis from the response
        return response.text.strip()

if __name__ == "__main__":
    # Replace 'your-project-id' with your actual GCP project ID.
    chess_ai = ChessAI(project_id="chessai-453307")
    
    # Example: FEN for the starting chess position
    starting_position = "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6"
    
    result = chess_ai.generate_move(starting_position, game_history = "", # Assuming no moves have been made yet
                                    elo = 1200, # Example ELO rating
                                    valid_moves = [])
    print("ChessAI Response:")
    print(result)
