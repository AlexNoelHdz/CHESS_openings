import chess
import chess.engine
from chess import Board
from chessboard import display
from extract_features import get_unique_opening_moves, select_move_by_weighted_choice
import pandas as pd
from chess_helpers import mostrar_df_en_dialogo

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
    print("================HUMAN================")
    if len(unique_opening_moves) > 0:
        print(f"Movimientos comunes {opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
    move_input = input("Tu movimiento: ")
    return manual_move(move_input, board, displayed_board)

def run_machine_movement(board, displayed_board, unique_opening_moves):
    if len(unique_opening_moves) > 0:
        print("================Openings IA================")
        move_input = select_move_by_weighted_choice(unique_opening_moves)
        print(f"Movimientos comunes {opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
        print(f"IA mueve: {move_input}")
        return manual_move(move_input, board, displayed_board)
    else:
        print("================Openings IA================")
        # Si no existen movimientos para alcanzar alguna posición de apertura
        # Se habilita el motor Stockfish para el resto de la partida
        move_input = stockfish_move(board, displayed_board)
        print(f"IA mueve: {move_input}")
        return move_input

def flip_board_if_needed(displayed_board):
    if not displayed_board.flipped:
        display.flip(displayed_board)

def load_historical_games(opening_shortname):
    moves_data_path = "../CHESS/data/df_2_just_moves_fen.csv"
    df = pd.read_csv(moves_data_path, encoding='utf-8', engine='python')
    df = df[df['opening_shortname'] == opening_shortname]
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

with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    board = chess.Board()

    opening_shortname = input("Apertura a practicar: ")

    df_moves = load_historical_games(opening_shortname)

    b_or_w_input = input("Blancas 'w' o negras 'b': ")

    displayed_board = display.start()

    play_game = run_human_black_board if b_or_w_input == 'b' else run_human_white_board

    while not board.is_game_over():
        unique_opening_moves, turn_column_name = get_unique_opening_moves(
            df_moves, # Df already filter
            board.turn, 
            board.fullmove_number)

        played_move = play_game(board, displayed_board, unique_opening_moves)

        df_moves = apply_played_move_to_df(df_moves, turn_column_name, played_move)

