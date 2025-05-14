# -*- coding: utf-8 -*-
"""Streamlit application for playing chess against an AI opponent.

This application allows a user to play chess against an AI powered by
Vertex AI and Stockfish. It includes features like ELO-based difficulty,
player style selection (including automatic detection), move analysis,
take back, reset, resign, and game saving.
"""

import logging
import random
from io import BytesIO

import chess
import chess.svg
import streamlit as st
import torch
from cairosvg import svg2png
from PIL import Image

# Import custom modules
from chess_ai import ChessAI
from chess_style_classifier import StyleClassifier

# --- Constants ---
DEFAULT_ELO = 1200
MIN_ELO = 400
MAX_ELO = 2800
ELO_STEP = 1
DEFAULT_PLAYER_STYLE = "Materialistic"
MIN_MOVES_FOR_AUTO_STYLE = 6
PROJECT_ID = "chessai-453307" # Consider moving to config or env variable
SAVED_GAME_FILENAME = "saved_game.txt"
PLAYER_COLORS = ["White", "Black"]
STYLE_OPTIONS = [
    "Materialistic",
    "Aggressive",
    "Defensive",
    "Tactical",
    "Endgame Grinder",
    "Automatic",
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def render_board(board: chess.Board) -> Image.Image:
    """Renders the current chess board state as a PIL Image.

    Args:
        board: The current python-chess board object.

    Returns:
        A PIL Image object representing the board.
    """
    svg_board = chess.svg.board(board, size=1200)
    png_board = svg2png(bytestring=svg_board)
    return Image.open(BytesIO(png_board))


def get_bot_move(board: chess.Board, chess_ai: ChessAI, elo: int, player_style: str) -> str:
    """Generates a move for the AI bot.

    Uses the ChessAI class to generate a move based on the current board state,
    move history, player ELO, legal moves, and player style.

    Args:
        board: The current python-chess board object.
        chess_ai: An initialized ChessAI instance.
        elo: The user's ELO rating.
        player_style: The selected playing style for the AI.

    Returns:
        The AI's move in Standard Algebraic Notation (SAN).
    """
    fen = board.fen()
    legal_moves_san = [board.san(move) for move in board.legal_moves]
    move_history = st.session_state.get("move_history", [])
    return chess_ai.generate_move(
        fen,
        move_history,
        elo,
        legal_moves_san,
        max_attempts=3,
        player_style=player_style,
    )

def initialize_session_state():
    """Initializes necessary variables in Streamlit's session state."""
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "move_history" not in st.session_state:
        st.session_state.move_history = []
    if "fen_history" not in st.session_state:
        # Store initial FEN
        st.session_state.fen_history = [st.session_state.board.fen()]
    if "move_analysis" not in st.session_state:
        st.session_state.move_analysis = []
    if "player_style" not in st.session_state:
        st.session_state.player_style = DEFAULT_PLAYER_STYLE
    if "show_move_analysis" not in st.session_state:
        st.session_state.show_move_analysis = True
    if "user_color_bool" not in st.session_state:
        # Default to White if not set, will be updated by sidebar selection
        st.session_state.user_color_bool = chess.WHITE
    if "chess_ai" not in st.session_state:
        # device = 0 if torch.cuda.is_available() else -1 # GPU check example
        st.session_state.chess_ai = ChessAI(project_id=PROJECT_ID)
        logger.info("ChessAI initialized.")
    if 'style_classifier' not in st.session_state:
        st.session_state.style_classifier = StyleClassifier(project_id=PROJECT_ID)
        logger.info("StyleClassifier initialized.")


def reset_game(selected_style: str):
    """Resets the game state to the beginning.

    Args:
        selected_style: The style selected in the sidebar, used to reset
                        the player_style state.
    """
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.fen_history = [chess.Board().fen()]
    st.session_state.move_analysis = []
    st.session_state.player_style = selected_style # Reset to the potentially changed selected style
    st.session_state.show_move_analysis = True # Reset display preference
    logger.info("Game reset.")
    st.rerun()


def take_back_move(user_color_bool: bool):
    """Reverts the game state to the player's previous turn."""
    moves_popped = 0
    # Ensure board and histories are not empty
    while st.session_state.move_history:
        # Pop from board first to get the correct turn after popping
        st.session_state.board.pop()
        st.session_state.move_history.pop()
        st.session_state.fen_history.pop()

        # Determine if the move just undone was the user's move.
        # The user's turn is *after* the pop if the current turn is NOT theirs.
        is_user_move_undone = st.session_state.board.turn != user_color_bool

        # Pop analysis only if a user move was undone and analysis exists
        if is_user_move_undone and st.session_state.move_analysis:
            st.session_state.move_analysis.pop()

        moves_popped += 1

        # Stop if it's now the user's turn or the board is empty
        if st.session_state.board.turn == user_color_bool or not st.session_state.move_history:
            break

    if moves_popped > 0:
        logger.info(f"Took back {moves_popped} half-moves.")
        st.success(f"Reverted to your turn. {moves_popped} half-move(s) taken back.")
    else:
        st.warning("No moves to take back.")
    st.rerun() # Rerun to reflect the changes


def resign_game(user_color_bool: bool):
    """Ends the game with the current player resigning."""
    opponent_color = not user_color_bool
    st.session_state.board.outcome = chess.Outcome(
        termination=chess.Termination.VARIANT_WIN, winner=opponent_color
    )
    st.session_state.move_history.append("Resign") # Add marker to history
    st.session_state.fen_history.append(st.session_state.board.fen()) # Store final FEN
    st.session_state.move_analysis.append("Game resigned by user.") # Add analysis note
    logger.info("User resigned.")
    st.success("You resigned the game.")
    st.rerun()


def save_game():
    """Saves the move history (SAN) to a text file."""
    try:
        with open(SAVED_GAME_FILENAME, "w", encoding="utf-8") as f:
            f.write("\n".join(st.session_state.move_history))
        logger.info(f"Game saved to {SAVED_GAME_FILENAME}")
        st.success("Game saved successfully!")
    except IOError as e:
        logger.error(f"Error saving game: {e}")
        st.error(f"Failed to save game: {e}")


def update_player_style(selected_style_option: str, user_color_bool: bool):
    """Updates the player style based on selection or automatic detection.

    Args:
        selected_style_option: The style selected in the sidebar dropdown.
        user_color_bool: The color the user is playing (chess.WHITE or chess.BLACK).
    """
    if selected_style_option == "Automatic":
        move_history = st.session_state.get("move_history", [])
        if len(move_history) >= MIN_MOVES_FOR_AUTO_STYLE:
            game_phase = st.session_state.chess_ai._get_game_phase(move_history)
            # Assume the opponent is the opposite color of the user
            opponent_color_str = "white" if user_color_bool == chess.BLACK else "black"
            try:
                style, _ = st.session_state.style_classifier.classify(
                    game_phase=game_phase,
                    game_history=move_history,
                    opponent_color=opponent_color_str,
                )
                st.session_state.player_style = style
                st.sidebar.info(f"Detected style: **{style}**")
                logger.info(f"Automatic style detected: {style}")
            except Exception as e:
                logger.error(f"Error during automatic style classification: {e}")
                st.sidebar.error("Could not detect style automatically.")
                # Fallback to default if classification fails
                st.session_state.player_style = DEFAULT_PLAYER_STYLE
        else:
            # Not enough moves, use default and inform user
            st.session_state.player_style = DEFAULT_PLAYER_STYLE
            st.sidebar.info(f"Using default style ({DEFAULT_PLAYER_STYLE}). Need {MIN_MOVES_FOR_AUTO_STYLE}+ moves for automatic detection.")
    else:
        # Use the explicitly selected style
        st.session_state.player_style = selected_style_option
        logger.info(f"Player style set to: {selected_style_option}")


# --- Streamlit UI ---

st.set_page_config(page_title="ChessAI", page_icon="♞", layout="wide")
st.logo("Logo.png", icon_image="icon.png", size="large") # size and width parameters are not supported by st.logo. Logo size is primarily determined by the image file itself.
st.title("ChessAI: Play Chess with an AI Bot")

# Initialize session state variables if they don't exist
initialize_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("Player Settings")

    # User ELO
    user_elo = st.number_input(
        "Enter your ELO",
        min_value=MIN_ELO,
        max_value=MAX_ELO,
        value=st.session_state.get("user_elo", DEFAULT_ELO), # Persist ELO
        step=ELO_STEP,
        key="user_elo" # Use key to store in session state automatically
    )

    # User Color
    user_color_str = st.selectbox(
        "Choose your color",
        options=PLAYER_COLORS,
        index=PLAYER_COLORS.index("Black") if st.session_state.get("user_color_bool") == chess.BLACK else 0, # Persist color choice
        key="user_color_selection"
    )
    user_color_bool = chess.WHITE if user_color_str == "White" else chess.BLACK
    # Update session state if color changed
    if st.session_state.user_color_bool != user_color_bool:
        st.session_state.user_color_bool = user_color_bool
        # Consider resetting the game if the color changes mid-game, or provide a warning.
        # reset_game(st.session_state.player_style) # Example: uncomment to reset on color change
        logger.info(f"User color changed to: {user_color_str}")
        st.warning("Color changed. Consider resetting the game for consistency.")
        st.rerun() # Rerun to reflect potential immediate changes

    # Player Style Selection
    selected_style_option = st.selectbox(
        "Select AI playing style",
        options=STYLE_OPTIONS,
        index=STYLE_OPTIONS.index(st.session_state.player_style) if st.session_state.player_style in STYLE_OPTIONS else 0,
        key="selected_style" # Use key to store selection
    )

    # Update style based on selection or automatic detection
    update_player_style(selected_style_option, st.session_state.user_color_bool)

    st.header("Game Controls")
    # Take Back Button
    if st.button("↩️ Take Back", ):
        take_back_move(st.session_state.user_color_bool)

    # Reset Game Button
    if st.button("🔄 Reset Game"):
        reset_game(st.session_state.selected_style) # Pass the current selection

    # Resign Button
    if st.button("🚩 Resign"):
        resign_game(st.session_state.user_color_bool)

    # Save Game Button
    if st.button("💾 Save Game"):
        save_game()

    # Toggle Move Analysis Display
    st.checkbox(
        "Show move analysis",
        value=st.session_state.show_move_analysis,
        key="show_move_analysis", # Automatically updates session state
        help="Show AI analysis after each of your moves."
    )

# --- Main Content Area ---

# Display Board (using columns for better layout potentially)
col1, col2 = st.columns([2, 1]) # Adjust ratios as needed

with col1:
    st.subheader("Chess Board")
    board_image_placeholder = st.empty()
    board_image_placeholder.image(render_board(st.session_state.board), use_container_width=True)

# --- Game Logic ---

# Check for Game Over
game_over = st.session_state.board.is_game_over()
if game_over:
    outcome = st.session_state.board.outcome()
    if outcome:
        result_text = ""
        icon = ""
        if outcome.termination == chess.Termination.CHECKMATE:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            result_text = f"Checkmate! {winner} wins."
            icon = "🎉"
        elif outcome.termination == chess.Termination.STALEMATE:
            result_text = "Stalemate! It's a draw."
            icon = "🤝"
        elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            result_text = "Draw due to insufficient material."
            icon = "🤝"
        elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
            result_text = "Draw by 75-move rule."
            icon = "⏳"
        elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
            result_text = "Draw by fivefold repetition."
            icon = "🔄"
        elif outcome.termination == chess.Termination.VARIANT_WIN:
             winner = "White" if outcome.winner == chess.WHITE else "Black"
             result_text = f"Variant win for {winner}." # Adjust for specific variant if needed
             icon = "🏆"
        elif outcome.termination == chess.Termination.VARIANT_LOSS:
             winner = "White" if outcome.winner == chess.BLACK else "Black" # Loser's opponent wins
             result_text = f"Variant loss for {'Black' if outcome.winner == chess.WHITE else 'White'}. {winner} wins."
             icon = "🏆"
        elif outcome.termination == chess.Termination.VARIANT_DRAW:
             result_text = "Draw by variant rules."
             icon = "🤝"
        if result_text:
             st.success(result_text, icon=icon)
             st.toast("Game over!")


# --- User Move Input ---
is_user_turn = st.session_state.board.turn == st.session_state.user_color_bool

with col2: # Place input and history in the second column
    st.subheader("Your Move")
    with st.form(key="move_form"):
        user_move_san = st.text_input(
            "Enter move (e.g., e4, Nf3, O-O):",
            key="move_input",
            disabled=game_over or not is_user_turn,
            placeholder="Nf3" if is_user_turn else "Waiting for opponent..."
        )
        submit_move = st.form_submit_button("Make Move", disabled=game_over or not is_user_turn)

        if submit_move and user_move_san and is_user_turn:
            try:
                fen_before = st.session_state.board.fen()
                # Attempt to parse and push the move
                parsed_move = st.session_state.board.parse_san(user_move_san)
                san_move = st.session_state.board.san(parsed_move) # Get canonical SAN

                st.session_state.board.push(parsed_move)
                board_image_placeholder.image(render_board(st.session_state.board), use_container_width=True)
                fen_after = st.session_state.board.fen()

                # Update histories
                st.session_state.move_history.append(san_move)
                st.session_state.fen_history.append(fen_after)
                logger.info(f"User move: {san_move} (FEN before: {fen_before}, FEN after: {fen_after})")

                # --- Move Analysis ---
                if st.session_state.show_move_analysis:
                     with st.spinner("Analyzing your move..."):
                        try:
                            analysis = st.session_state.chess_ai.analyze_move(
                                fen_before=fen_before,
                                fen_after=fen_after,
                                move_san=san_move,
                                elo=st.session_state.user_elo,
                                game_history=st.session_state.move_history[:-1], # History *before* the move
                            )
                            analysis_entry = f"**Your Move {len(st.session_state.move_history)} ({san_move}):**\n{analysis}"
                            st.session_state.move_analysis.append(analysis_entry)
                            logger.info("Move analysis generated.")
                        except Exception as e:
                            logger.error(f"Error generating move analysis: {e}")
                            st.session_state.move_analysis.append(f"**Your Move {len(st.session_state.move_history)} ({san_move}):**\nAnalysis failed: {e}")


                # Clear the input field by rerunning
                st.rerun()

            except ValueError as e:
                logger.warning(f"Invalid user move input: {user_move_san} - Error: {e}")
                st.error(f"Invalid move: '{user_move_san}'. Please use Standard Algebraic Notation (SAN). Is it legal?")
            except Exception as e:
                 logger.error(f"An unexpected error occurred during move processing: {e}")
                 st.error("An unexpected error occurred. Please try again.")


    # --- Move History Display ---
    st.subheader("Move History (SAN)")
    formatted_moves = []
    for i, move in enumerate(st.session_state.move_history):
        if i % 2 == 0: # White's move
            formatted_moves.append(f"{i//2 + 1}. {move}") # Add move number and space
        else: # Black's move
            formatted_moves.append(move)
    history_text = " ".join(formatted_moves)
    st.text_area(
        "History",
        value=history_text if st.session_state.move_history else "No moves yet.",
        height=150, # Increased height slightly
        key="history_display",
        disabled=True
    )

# --- Move Analysis Display ---
if st.session_state.show_move_analysis:
    with st.expander("Latest Move Analysis", expanded=True):
        if st.session_state.move_analysis:
            # Display the most recent analysis
            last_analysis = st.session_state.move_analysis[-1]
            st.markdown(last_analysis, unsafe_allow_html=True) # Allow potential HTML in analysis
        else:
            st.info("No analysis available yet. Make a move!")


# --- Bot Move Logic ---
bot_color_bool = not st.session_state.user_color_bool
is_bot_turn = st.session_state.board.turn == bot_color_bool

if not game_over and is_bot_turn:
    with st.spinner("AI is thinking..."):
        logger.info("AI turn started.")
        try:
            bot_move_san = get_bot_move(
                st.session_state.board,
                st.session_state.chess_ai,
                st.session_state.user_elo, # AI adjusts based on user ELO
                st.session_state.player_style
            )
            logger.info(f"AI generated move: {bot_move_san}")
            parsed_move = st.session_state.board.parse_san(bot_move_san)
            st.session_state.board.push(parsed_move)
            st.session_state.move_history.append(bot_move_san)
            st.session_state.fen_history.append(st.session_state.board.fen())
            logger.info("AI move executed.")
            # Bot move analysis is generally not shown/generated to save resources
        except ValueError as e:
            logger.error(f"AI generated an invalid move: '{bot_move_san}'. Error: {e}. Attempting fallback.")
            st.error(f"AI Error: Bot tried an invalid move ({bot_move_san}). Using random fallback.")
            try:
                # Fallback to a random legal move
                legal_moves = list(st.session_state.board.legal_moves)
                if legal_moves:
                    fallback_move = random.choice(legal_moves)
                    fallback_san = st.session_state.board.san(fallback_move)
                    st.session_state.board.push(fallback_move)
                    st.session_state.move_history.append(fallback_san)
                    st.session_state.fen_history.append(st.session_state.board.fen())
                    logger.info(f"AI fallback move executed: {fallback_san}")
                else:
                    logger.error("AI Error: No legal moves available for fallback.")
                    st.error("AI Error: No legal moves available for the bot.")
                    # This state implies game over, which should be caught earlier,
                    # but handle defensively.
            except Exception as fallback_e:
                 logger.critical(f"Critical error during AI fallback move: {fallback_e}")
                 st.error("Critical error during AI fallback move.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during AI move generation/execution: {e}")
            st.error("An unexpected error occurred while processing the AI move.")

        # Rerun to update the board and switch turns visually
        st.rerun()


# Ensure the board image is updated after potential state changes (like bot move)
# This is slightly redundant if rerun is called, but ensures latest state display
board_image_placeholder.image(render_board(st.session_state.board), use_container_width=True)