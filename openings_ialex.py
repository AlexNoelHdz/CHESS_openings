import chess
import chess.engine
from chess import Board
from chessboard import display
from extract_features import get_current_opening, count_all_features
from extract_features import get_all_features_uf
import pandas as pd
from utils import ChessLogger
import numpy as np
import random

class OpeningsIA:
    def __init__(self, engine, random = False, opening_shortname = 'Sicilian Defense', b_or_w_input = 'b', logs_path = "./logs") -> None:
        self.restart_game = False
        self.engine = engine
        self.logger = ChessLogger(logs_path)
        self.df_hist_moves = self.load_historical_games()
        if random:
            b_or_w_input = self.select_random_color()
            opening_shortname = self.select_random_opening()
        self.play_game = self.play_game_function(b_or_w_input)
        self.opening_shortname = opening_shortname
        
        self.df_hist_moves_filter = self.filter_possible_moves_by_opening()

    def stockfish_move(self, board, displayed_board):
        board_analysis = self.engine.analyse(board, limit=chess.engine.Limit(time=1))
        best_move = board_analysis.get("pv")[0]
        san_move = board.san(best_move)
        board.push(best_move)
        fen_robot = board.fen()
        display.check_for_quit()
        display.update(fen_robot, displayed_board)
        return san_move

    def manual_move(self, move_input:str, board:Board, displayed_board):
        try:
            # Salir si el usuario envía x
            if move_input == "x":
                display.terminate()
            if move_input == 'z':
                self.restart_game = True
                return
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


    def run_human_white_board(self, board, displayed_board, unique_opening_moves):
        if board.turn == chess.WHITE:
            return  self.run_human_movement(board, displayed_board, unique_opening_moves)
        else:
            return  self.run_machine_movement(board, displayed_board, unique_opening_moves)

    def run_human_black_board(self, board, displayed_board, unique_opening_moves):
        if board.turn == chess.BLACK:
            self.flip_board_if_needed(displayed_board)
            return self.run_human_movement(board, displayed_board, unique_opening_moves)
        else:
            return self.run_machine_movement(board, displayed_board, unique_opening_moves)

    def run_human_movement(self, board, displayed_board, unique_opening_moves):
        print("================HUMAN================")
        self.logger.write("================HUMAN================")
        # TODO: Ventajas posicionales (A partir del movimiento 3)
        if unique_opening_moves:
            self.logger.write(f"Movimientos comunes {self.opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
            print(f"Movimientos comunes {self.opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
        move_input = input("Tu movimiento: ")
        self.logger.write(f"Humano mueve: {move_input}")
        return self.manual_move(move_input, board, displayed_board)

    def run_machine_movement(self,board, displayed_board, unique_opening_moves):
        print(("================Openings IA================"))
        self.logger.write("================Openings IA================")
        if unique_opening_moves:
            self.logger.write(f"Movimientos comunes {self.opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
            print(f"Movimientos comunes {self.opening_shortname}: {[move[0] for move in unique_opening_moves]}.")
            move_input = self.select_move_by_weighted_choice(unique_opening_moves)
            self.logger.write(f"IA mueve: {move_input}")
            print(f"IA mueve: {move_input}")
            return self.manual_move(move_input, board, displayed_board)
        else:
            # Si no existen movimientos para alcanzar alguna posición de apertura
            # Se habilita el motor Stockfish para el resto de la partida
            move_input = self.stockfish_move(board, displayed_board)
            self.logger.write(f"IA mueve: {move_input}")
            print(f"IA mueve: {move_input}")
            return move_input

    def flip_board_if_needed(self, displayed_board):
        if not displayed_board.flipped:
            display.flip(displayed_board)

    def load_historical_games(self, moves_data_path = "./data/df_2_just_moves_fen.csv"):
        df = pd.read_csv(moves_data_path, encoding='utf-8', engine='python')
        return df

    def filter_openings_by_played_move(self, turn_column_name, played_move):
        '''
        Filtra las partidas en el data set historico, para que las siguientes sugerencias sean con respecto 
        a nuevas posiciones alcanzadas.
        '''
        df = self.df_hist_moves_filter
        if df.empty or df.shape[0]<=1:
            df = pd.DataFrame()
        elif turn_column_name in df.columns:
            played_move_exists = df[turn_column_name].isin([played_move]).any()
            if played_move_exists: 
                df = df[df[turn_column_name] == played_move]
        else:
            df = pd.DataFrame()
        self.df_hist_moves_filter = df

    def predict_position(self, fen, moves, model):
        features = count_all_features(fen, moves)
        return model.predict_proba(features)

    def show_position_predictions(self, board, loaded_model):
        prob = self.predict_position(board.fen(), board.ply(), loaded_model)
        position_predictions = f"Negras: {'{:.6f}'.format(prob[0][0])}. Blancas:{'{:.6f}'.format(prob[0][1])}"
        self.logger.write(position_predictions)
        print(position_predictions)

    def show_opening_reached(self, board, turn_column_name):
        opening_reached = get_current_opening(self.df_hist_moves, turn_column_name, board.fen(), board.ply())
        if opening_reached:
            self.logger.write(f"Apertura alcanzada: {opening_reached}")
            print(f"Apertura alcanzada: {opening_reached}")

    def show_position_features(self, fen, turns):
        self.logger.write(f"-----Turno {turns}:{fen}-----")
        df_white = get_all_features_uf(fen, True)
        df_black = get_all_features_uf(fen, False)
        self.logger.write(f"---------------Posicion Blancas---------------")
        self.logger.write(df_white.to_string(index=True))
        self.logger.write(f"---------------Posicion Negras---------------")
        self.logger.write(df_black.to_string(index=False))

    def play_game_function(self, b_or_w_input):
        return self.run_human_black_board if b_or_w_input == 'b' else self.run_human_white_board
    
    def filter_possible_moves_by_opening(self):
     return self.df_hist_moves[self.df_hist_moves['opening_shortname'] == self.opening_shortname].copy()
    
    def select_random_opening(self):
        # Calcula las proporciones de cada apertura en el conjunto de datos
        counts = self.df_hist_moves['opening_shortname'].value_counts()
        weights = counts / counts.sum()

        # Redondea las proporciones a dos decimales y convierte a porcentajes
        rounded_weights = np.round(weights, 2) * 100

        # Ajuste para asegurar que ninguna apertura tenga una probabilidad menor que 1%
        adjusted_weights = np.maximum(rounded_weights, 1).astype(int)

        # Selecciona una apertura al azar basado en los pesos ajustados
        opening_selected = random.choices(adjusted_weights.index, weights=adjusted_weights.values, k=1)[0]
        
        # Muestra e imprime la apertura seleccionada
        print(f"Apertura a practicar {opening_selected}")
        self.logger.write(f"Apertura a practicar {opening_selected}")

        return opening_selected
    
    def select_random_color(self):
        return random.choice(['w', 'b'])
    
    def select_move_by_weighted_choice(self, weights):
        """
        Selecciona un movimiento de ajedrez basado en una lista de pesos para cada movimiento.
        
        Parámetros:
        - weights: Lista de tuplas, donde cada tupla contiene un movimiento (como 'e4') y su peso asociado.
        
        Retorna:
        - Movimiento seleccionado de manera ponderada.
        """
        # Desempaquetar la lista de tuplas en movimientos y sus respectivos pesos
        moves, move_weights = zip(*weights) # from get_unique_opening_moves
        
        # Seleccionar un movimiento de manera ponderada basada en los pesos
        selected_move = random.choices(moves, weights=move_weights, k=1)[0] 
        return selected_move
    
    def get_unique_opening_moves(self, turno, fullmove_number):
        """
        Filtra el DataFrame basado en el 'opening_shortname' proporcionado y retorna los valores únicos
        del turno.
        
        Parámetros:
        - turno: Booleano, True para WHITE, False para BLACK.
        - fullmove_number: Entero, número de la jugada de la partida de ajedrez.
        - opening_shortname: String, nombre de la apertura a filtrar.
        
        Retorna:
        - Lista de valores únicos de la columna 'turn_column_name' o None si la columna no existe.
        """
        turno_str = "0w" if turno else "1b"
        turn_column_name = f"{turno_str}_{fullmove_number}"

        if self.df_hist_moves_filter.empty:
            return None, turn_column_name

        # Verificar si la columna generada existe en el DataFrame filtrado
        if turn_column_name in self.df_hist_moves_filter.columns:
            # Calcular la frecuencia de cada valor único
            value_counts = self.df_hist_moves_filter[turn_column_name].value_counts(normalize=True) # Valores entre 0 y 1

            # Normalización para aumentar la probabilidad de que la máquina te juegue jugadas poco comunes
            # 0-5: 1
            # 5-50: 3
            # 50-100: 20
            determine_weight = lambda proportion: 1 if proportion <= 0.05 else \
                                            5 if proportion <= 0.50 else 20

            # Crear la lista de tuplas (valor único, peso normalizado) usando la función lambda
            weights = [(value, determine_weight(proportion)) for value, proportion in value_counts.items()]
        
            return weights, turn_column_name
        else:
            return None, turn_column_name