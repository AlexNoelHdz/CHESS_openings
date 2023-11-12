import chess
import chess.engine
from chessboard import display

# Descargar y extraer toda la carpeta en CHESS/stockfish
# https://stockfishchess.org/download/windows/
stockfish_path = "../CHESS/stockfish/stockfish-windows-x86-64.exe"

def human_move(move_input, board, displayed_board):
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
    except chess.InvalidMoveError:
        print("Entrada no válida. Utiliza notación SAN (por ejemplo, 'e3').")
    except chess.IllegalMoveError:
        print("Movimiento Ilegal. Intente de nuevo")
    except chess.AmbiguousMoveError as e:
        print("Movimiento ambiguo, intente ser más específico. Por ejemplo especificar la letra origen (Rad1)")
    except ValueError:
        print("Error inesperado.")

def stock_fish_move(board):
    board_analysis = engine.analyse(board, limit=chess.engine.Limit(time=3))
    best_move = board_analysis.get("pv")[0]
    board.push(best_move)
    fen_robot = board.fen()
    display.check_for_quit()
    display.update(fen_robot, displayed_board)

def run_human_white_board(board, displayed_board):
    # Si el turno es del jugador humano
    if board.turn == chess.WHITE:
        #Captura el movimiento del humano
        move_input = input("Tu jugada: ")
        human_move(move_input, board, displayed_board)
    else:
        # Turno del motor Stockfish
        stock_fish_move(board)

def run_human_black_board(board, displayed_board):
    # Si el turno es del jugador humano
    if board.turn == chess.BLACK:
        # Flip board if necessary
        if not displayed_board.flipped:
            display.flip(displayed_board)
        #Captura el movimiento del humano
        move_input = input("Tu jugada: ")
        human_move(move_input, board, displayed_board)
    else:
        # Turno del motor Stockfish
        stock_fish_move(board)

# Iniciar Stockfish engine
with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    # Crear un objeto tipo tablero con la posición de ajedrez
    # Inicializa el tablero
    board = chess.Board()
    # Muestra el tablero en una ventana emergente con tamaño definido en el atributo size
    displayed_board = display.start()
    # Pregunta al humano si quiere jugar con blancas o negras
    b_or_w_input = input("Play white 'w' or black 'b': ")
    play_game = run_human_black_board if b_or_w_input == 'b' else run_human_white_board

    # Mientras que el board no indique jaque mate
    while not board.is_game_over():
        play_game(board, displayed_board)


