import chess
import chess.engine
from chess import Board
from chessboard import display
from extract_features import get_unique_opening_moves, select_move_by_weighted_choice, get_current_opening, count_all_features, get_all_features
from extract_features import get_all_features_uf
import pandas as pd
from helpers import ChessLogger
import pickle
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Descargar y extraer toda la carpeta en CHESS/stockfish
# https://stockfishchess.org/download/windows/
stockfish_path = "../CHESS/stockfish/stockfish-windows-x86-64.exe"

def manual_move(move_input:str, board:Board, displayed_board):
    try:
        # Salir si el usuario envía x
        if move_input == "x":
            display.terminate()
        # push movement
        board.push_san(move_input)
        # Get fen from board
        fen_human = board.fen()
        display.check_for_quit()
        # Show
        display.update(fen_human, displayed_board)
        return move_input
    except chess.InvalidMoveError as e:
        print(f"Entrada no válida. Utiliza notación SAN (por ejemplo, 'e3').\n{e}")
    except chess.IllegalMoveError as e:
        print(f"Movimiento Ilegal. Intente de nuevo\n{e}")
    except chess.AmbiguousMoveError as e:
        print(f"Movimiento ambiguo, intente ser más específico. Por ejemplo especificar la letra origen (Rad1)\n{e}")
    except ValueError as e:
        print(f"Error inesperado.\n{e}")

def stockfish_move(board, displayed_board):
    board_analysis = engine.analyse(board, limit=chess.engine.Limit(time=1))
    best_move = board_analysis.get("pv")[0]
    san_move = board.san(best_move)
    board.push(best_move)
    fen_robot = board.fen()
    display.check_for_quit()
    display.update(fen_robot, displayed_board)
    return san_move

def run_human_white_board(board, displayed_board, unique_opening_moves):
    if board.turn == chess.WHITE:
        return  run_human_movement(board, displayed_board, unique_opening_moves)
    else:
        return  run_machine_movement(board, displayed_board, unique_opening_moves)

def run_human_black_board(board, displayed_board, unique_opening_moves):
    if board.turn == chess.BLACK:
        flip_board_if_needed(displayed_board)
        return run_human_movement(board, displayed_board, unique_opening_moves)
    else:
        return run_machine_movement(board, displayed_board, unique_opening_moves)

def run_human_movement(board, displayed_board, unique_opening_moves):
    logger.write("================HUMAN================")
    # TODO: Ventajas posicionales (A partir del movimiento 3)
    if unique_opening_moves:
        logger.write(f"Movimientos comunes {opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
    move_input = input("Tu movimiento: ")
    logger.write(f"Humano mueve: {move_input}")
    return manual_move(move_input, board, displayed_board)

def run_machine_movement(board, displayed_board, unique_opening_moves):
    logger.write("================Openings IA================")
    if unique_opening_moves:
        move_input = select_move_by_weighted_choice(unique_opening_moves)
        logger.write(f"Movimientos comunes {opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
        logger.write(f"IA mueve: {move_input}")
        print(f"IA mueve: {move_input}")
        return manual_move(move_input, board, displayed_board)
    else:
        # Si no existen movimientos para alcanzar alguna posición de apertura
        # Se habilita el motor Stockfish para el resto de la partida
        move_input = stockfish_move(board, displayed_board)
        logger.write(f"IA mueve: {move_input}")
        print(f"IA mueve: {move_input}")
        return move_input

def flip_board_if_needed(displayed_board):
    if not displayed_board.flipped:
        display.flip(displayed_board)

def load_historical_games():
    moves_data_path = "../CHESS/data/df_2_just_moves_fen.csv"
    df = pd.read_csv(moves_data_path, encoding='utf-8', engine='python')
    return df

def apply_played_move_to_df(df, turn_column_name, played_move):
    '''
    Filtra las partidas en el data set historico, para que las siguientes sugerencias sean con respecto 
    a nuevas posiciones alcanzadas.
    '''
    if df_moves.empty or df.shape[0]<=1:
        return pd.DataFrame()
    played_move_exists = df[turn_column_name].isin([played_move]).any()
    if played_move_exists: 
        df = df[df[turn_column_name] == played_move]
    return df

def predict_position(fen, moves, model):
    features = count_all_features(fen, moves)
    return model.predict_proba(features)

def print_position_predictions(board, loaded_model):
    prob = predict_position(board.fen(), board.ply(), loaded_model)
    position_predictions = f"Negras: {'{:.6f}'.format(prob[0][0])}. Empate:{'{:.6f}'.format(prob[0][1])}. Blancas:{'{:.6f}'.format(prob[0][2])}"
    logger.write(position_predictions)

def print_opening_reached(board, df_moves, turn_column_name):
    opening_reached = get_current_opening(df_moves, turn_column_name, board.fen(), board.ply())
    if opening_reached:
        logger.write(f"Apertura alcanzada: {opening_reached}")
        print(f"Apertura alcanzada: {opening_reached}")

def show_features(fen, turns):
    logger.write("-----Position-----")
    df = get_all_features_uf(fen, turns)
    logger.write(f"({turns}):{fen}")
    text_df = df.to_string(index=True)
    logger.write(text_df)

with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    board = chess.Board()
    logger = ChessLogger("./logs")

    loaded_model = pickle.load(open("./pickles/models/xgoboost_model0410.pkl", 'rb'))

    opening_shortname = input("Apertura a practicar: ")

    df_moves = load_historical_games()

    df_filter_moves = df_moves[df_moves['opening_shortname'] == opening_shortname].copy()

    b_or_w_input = input("Blancas 'w' o negras 'b': ")

    displayed_board = display.start()

    play_game = run_human_black_board if b_or_w_input == 'b' else run_human_white_board

    while not board.is_game_over():
        unique_opening_moves, turn_column_name = get_unique_opening_moves(
            df_filter_moves, # Df already filter or new
            board.turn, 
            board.fullmove_number)

        played_move = play_game(board, displayed_board, unique_opening_moves)

        print_position_predictions(board, loaded_model)
        show_features(board.fen(), board.ply())

        df_filter_moves = apply_played_move_to_df(df_filter_moves, turn_column_name, played_move)

        print_opening_reached(board, df_moves, turn_column_name)

