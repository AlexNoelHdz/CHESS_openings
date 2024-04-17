import pandas as pd
import extract_features as ef
from chess import Board
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NAMES
import numpy as np
data_path = "../CHESS/data/df_1.csv"
df_1 = pd.read_csv(data_path)
df_1.info()

# aplicando la función get_fen_from_moves a todas las filas de la columna moves
df_1['moves_fen'] = df_1['moves'].apply(ef.get_fen_from_moves)

# Aplicamos la función get_current_turn en todas las observaciones y generamos la nueva columna
df_1["current_turn"] = df_1['moves_fen'].apply(ef.get_current_turn)

# Aplicar funciones de características tanto a blancas como a negras
for color in ['w', 'b']:
    turn = True if color == 'w' else False
    df_1[f'{color}_ctrld_pawn'] =            df_1['moves_fen'].apply(lambda x: ef.count_pawn_capture_squares(x, turn))
    df_1[f'{color}_ctrld_knight'] =          df_1['moves_fen'].apply(lambda x: ef.count_legal_moves_from_piece_type(x, KNIGHT, turn))
    df_1[f'{color}_ctrld_bishop'] =          df_1['moves_fen'].apply(lambda x: ef.count_legal_moves_from_piece_type(x, BISHOP, turn))
    df_1[f'{color}_ctrld_rook'] =            df_1['moves_fen'].apply(lambda x: ef.count_legal_moves_from_piece_type(x, ROOK, turn))
    df_1[f'{color}_ctrld_queen'] =           df_1['moves_fen'].apply(lambda x: ef.count_legal_moves_from_piece_type(x, QUEEN, turn))
    df_1[f'{color}_ctrld_king'] =            df_1['moves_fen'].apply(lambda x: ef.count_legal_moves_from_piece_type(x, KING, turn))
    df_1[f'{color}_preassure_points'] =      df_1['moves_fen'].apply(lambda x: ef.count_pressure_points(x, turn))
    df_1[f'{color}_ctrld_diagonals'] =      df_1['moves_fen'].apply(lambda x: ef.count_all_controlled_diagonals(x, turn))
    df_1[f'{color}_ctrld_lines'] =          df_1['moves_fen'].apply(lambda x: ef.count_all_controlled_lines(x, turn))

# Guardar el CSV con features como df_2.csv
df_1.to_csv("../CHESS/data/df_3.csv", index=False) # Data frame con features basadas en ventaja posicional del turno


# Validación 
muestra_aleatoria = np.random.randint(0, df_1.shape[0])

fen = df_1['moves_fen'][muestra_aleatoria]
print(f'FEN: {fen}')
print(f"w_ctrld_pawn: {df_1['w_ctrld_pawn'][muestra_aleatoria]}")
print(f"w_ctrld_knight: {df_1['w_ctrld_knight'][muestra_aleatoria]}")
print(f"w_ctrld_bishop: {df_1['w_ctrld_bishop'][muestra_aleatoria]}") 
print(f"w_ctrld_rook: {df_1['w_ctrld_rook'][muestra_aleatoria]}")  
print(f"w_ctrld_queen: {df_1['w_ctrld_queen'][muestra_aleatoria]}") 
print(f"w_ctrld_king: {df_1['w_ctrld_king'][muestra_aleatoria]}")  
print(f"w_preassure_points: {df_1['w_preassure_points'][muestra_aleatoria]}")
print(f"w_ctrld_diagonals: {df_1['w_ctrld_diagonals'][muestra_aleatoria]}") 
print(f"w_ctrld_lines: {df_1['w_ctrld_lines'][muestra_aleatoria]}")  

# Validar contra la aplicacion de la funcion count all features
print(f'FEN: {fen}')
print(ef.count_all_features(fen, 'n').T)

print(f'FEN: {fen}')
pd.set_option('display.max_colwidth', None)
print(ef.get_all_features(fen, 'n').T)