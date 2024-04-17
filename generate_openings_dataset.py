import pandas as pd
org_data_path = "../CHESS/data/df_2_just_moves.csv"
df_2 = pd.read_csv(org_data_path)
df_2 = df_2.drop(columns=['moves_fen'])

from extract_features import get_fen_from_moves

def create_fen_columns(df_moves):
    '''
    Se añaden columnas para cada movimiento de apertura (de 1 a 28 máximo) en formato fen al df_moves proporcionado
    '''
    max_opening_moves = 28//2  # Número máximo de turnos de apertura (1 turno = 2 movimientos)

    # Crear las columnas para cada movimiento de apertura
    for turn_num in range(1, max_opening_moves + 1):
        # Rellenar con None donde no hay movimientos
        df_moves[f'0w_{turn_num}'] = None
        df_moves[f'0w_{turn_num}_fen'] = None
        df_moves[f'1b_{turn_num}'] = None
        df_moves[f'1b_{turn_num}_fen'] = None

    for index, row in df_moves.iterrows():
        moves = row['moves'].split()
        opening_moves = int(row['opening_moves'])

        if opening_moves % 2 == 0:
            opening_turns = opening_moves//2
        else:
            opening_turns = (opening_moves//2) + 1
        
        for turn_num in range(1, opening_turns + 1):
            san_white = moves[:turn_num*2-1]
            san_black = moves[:turn_num*2]
            fen_white = get_fen_from_moves(' '.join(san_white))
            fen_black = get_fen_from_moves(' '.join(san_black))
            df_moves.at[index, f'0w_{turn_num}'] = san_white[-1]
            df_moves.at[index, f'0w_{turn_num}_fen'] = fen_white

            # No rellenar el ultimo movimiento de negras cuando es el ultimo turno pero movimientos son impares
            if opening_moves % 2 != 0 and turn_num == opening_turns:
                break

            df_moves.at[index, f'1b_{turn_num}'] = san_black[-1]
            df_moves.at[index, f'1b_{turn_num}_fen'] = fen_black

# Aplicar función para crear n columnas como turnos existan para generar una apertura
create_fen_columns(df_2)
df_2 = df_2.drop(columns=['moves', 'game_id'])
df_2.columns

# Guardar el CSV con features como df_2.csv
df_2.to_csv("../CHESS/data/df_2_just_moves_fen.csv", index=False) # Data frame con features basadas en ventaja posicional del turno