import chess
import chess.engine
from chessboard import display

# Ruta al ejecutable de Stockfish previamente instalado desde
# https://stockfishchess.org/download/windows/
stockfish_path = "../CHESS/code/stockfish/stockfish-windows-x86-64.exe"

def human_move(move_input, board, displayed_board):
    try:
        if move_input == "x":
            return False
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            # push movement
            board.push(move)
            # Get fen from board
            fen_human = board.fen()
            display.check_for_quit()
            # Show
            display.update(fen_human, displayed_board)
        else:
            print("Movimiento no válido. Inténtalo de nuevo.")
    except ValueError:
        print("Entrada no válida. Utiliza notación UCI (por ejemplo, 'e2e4').")

# Iniciar Stockfish engine
with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    # Crear un objeto tipo tablero con la posición de ajedrez
    # Inicializa el tablero
    board = chess.Board()
    # Muestra el tablero en una ventana emergente con tamaño definido en el atributo size
    displayed_board = display.start()

    # Mientras que el board no indique jaque mate
    while not board.is_game_over():
        # Si el turno es del jugador humano
        if board.turn == chess.WHITE:
            #Captura el movimiento del humano
            move_input = input("Tu jugada: ")
            human_move(move_input, board, displayed_board)
        else:
            # Turno del motor Stockfish
            result = engine.analyse(board, limit=chess.engine.Limit(time=3))
            best_move = result.get("pv")[0]
            board.push(best_move)
            fen_robot = board.fen()
            display.check_for_quit()
            display.update(fen_robot, displayed_board)