import chess
import chess.engine
from chess import Board
from chessboard import display
from extract_features import get_unique_opening_moves
import pandas as pd

# Descargar y extraer toda la carpeta en CHESS/stockfish
# https://stockfishchess.org/download/windows/
stockfish_path = "../CHESS/stockfish/stockfish-windows-x86-64.exe"

def manual_move(move_input:str, board:Board, displayed_board, san_or_uci):
    try:
        # Salir si el usuario envía x
        if move_input == "x":
            display.terminate()

        if int(san_or_uci): 
            # push movement
            board.push_san(move_input)
        else:
            # push movement
            move = chess.Move.from_uci(move_input)
            if move in board.legal_moves:
                # push movement
                board.push(move)
            
            # Get fen from board
            fen_human = board.fen()
            display.check_for_quit()
            # Show
            display.update(fen_human, displayed_board)
    except chess.InvalidMoveError:
        print("Entrada no válida. Utiliza notación SAN (por ejemplo, 'e3').")
    except chess.IllegalMoveError:
        print("Movimiento Ilegal. Intente de nuevo")
    except chess.AmbiguousMoveError as e:
        print("Movimiento ambiguo, intente ser más específico. Por ejemplo especificar la letra origen (Rad1)")
    except ValueError:
        print("Error inesperado.")

def stock_fish_move(board, displayed_board):
    board_analysis = engine.analyse(board, limit=chess.engine.Limit(time=1))
    best_move = board_analysis.get("pv")[0]
    board.push(best_move)
    fen_robot = board.fen()
    display.check_for_quit()
    display.update(fen_robot, displayed_board)

def run_human_white_board(board, displayed_board, san_or_uci, unique_opening_moves):
    if board.turn == chess.WHITE:
        #Captura el movimiento del humano
        print(f"Movimientos de apertura disponibles Humano: [{unique_opening_moves}].")
        move_input = input("Tu movimiento: ")
        manual_move(move_input, board, displayed_board, san_or_uci)
    else:
        # Turno del motor Stockfish
        print(f"Movimientos de apertura disponibles PC: [{unique_opening_moves}].")
        stock_fish_move(board, displayed_board)

def run_human_black_board(board, displayed_board, san_or_uci, unique_opening_moves):
    if board.turn == chess.BLACK:
        # Flip board if necessary
        if not displayed_board.flipped:
            display.flip(displayed_board)
        #Captura el movimiento del humano
        print(f"Movimientos de apertura disponibles Humano: [{unique_opening_moves}].")
        move_input = input("Tu movimiento: ")
        manual_move(move_input, board, displayed_board, san_or_uci)
    else:
        # Turno del motor Stockfish
        print(f"Movimientos de apertura disponibles PC: [{unique_opening_moves}].")
        stock_fish_move(board, displayed_board)

# Iniciar Stockfish engine
with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    # Crear un objeto tipo tablero con la posición de ajedrez
    # Inicializa el tablero
    board = chess.Board()
    # Cargar las partidas historicas en un df_moves 
    moves_data_path = "../CHESS/data/df_2_just_moves_fen.csv"
    df_moves = pd.read_csv(moves_data_path)
    # Muestra el tablero en una ventana emergente
    displayed_board = display.start()
    # Preguntar qué apertura se quiere practicar
    opening_shortname = input("Opening you want to practice: ")
    # Pregunta al humano si quiere jugar con blancas o negras
    san_or_uci = input("UCI notation: 0. San notation 1: ")
    b_or_w_input = input("Play white 'w' or black 'b': ")
    play_game = run_human_black_board if b_or_w_input == 'b' else run_human_white_board

    # Mientras que el board no indique jaque mate
    while not board.is_game_over():
        unique_opening_moves = get_unique_opening_moves(df_moves, 
                                                        board.turn, 
                                                        board.fullmove_number,
                                                        opening_shortname)
        play_game(board, displayed_board, san_or_uci, unique_opening_moves)


