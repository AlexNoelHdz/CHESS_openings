import pandas as pd
import extract_features as ef
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NAMES
org_data_path = "../CHESS/data/df_1.csv"
df_1 = pd.read_csv(org_data_path)
df_1.info()

# aplicando la función get_fen_from_moves a todas las filas de la columna moves
df_1['moves_fen'] = df_1['moves'].apply(ef.get_fen_from_moves)

