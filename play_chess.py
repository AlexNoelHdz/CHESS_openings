import chess
import chess.engine
from chessboard import display
import pickle
from openings_ialex import OpeningsIA
from utils import clean_console

# Descargar y extraer toda la carpeta en ./stockfish
# https://stockfishchess.org/download/windows/
stockfish_path = "./stockfish/stockfish-windows-x86-64.exe"

# Determina si el juego estará generando openings random o los pedirá al usuario
RANDOM_OPENINGS = True

while True:
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        clean_console()
        
        if RANDOM_OPENINGS:
            openingsIA = OpeningsIA(engine, random=True)
            input("'z' para reiniciar. 'x' para cerrar. 'Enter' iniciar:")
        else:
            opening_shortname = input("Apertura a practicar: ")
            b_or_w_input = input("Blancas 'w' o negras 'b': ")
            openingsIA = OpeningsIA(engine, opening_shortname = opening_shortname, b_or_w_input = b_or_w_input)
        
        displayed_board = display.start()
        loaded_model = pickle.load(open("./pickles/models/xgoboost_model0416_18:06.pkl", 'rb'))
        board = chess.Board()

        while not board.is_game_over() and not openingsIA.restart_game:
            unique_opening_moves, turn_column_name = openingsIA.get_unique_opening_moves(board.turn, board.fullmove_number)

            played_move = openingsIA.play_game(board, displayed_board, unique_opening_moves)

            if openingsIA.restart_game: 
                break

            openingsIA.show_position_predictions(board, loaded_model)
            openingsIA.show_position_features(board.fen(), board.ply())
            openingsIA.filter_openings_by_played_move(turn_column_name, played_move)
            openingsIA.show_opening_reached(board, turn_column_name)

