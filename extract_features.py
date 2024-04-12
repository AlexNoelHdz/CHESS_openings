"""
Este archivo de python tiene la intención de proveer una serie de metodos para extraer caracteristicas 
desde posiciones dadas por cadenas FEN. 
"""
from chess import Board
from chess import square_rank, square_file, square
from chess import PIECE_TYPES, SQUARES, square_name
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
import random
import pandas as pd

def toggle_turn_on_fen(fen):
    """Cambia el turno en un string FEN.
    Este método tiene la intención de poder calcular características del oponente estableciendo
    una situación ficticia en la que tu turno es su turno
    """
    parts = fen.split(' ')
    parts[1] = 'w' if parts[1] == 'b' else 'b'
    return ' '.join(parts)

def get_custom_board(fen, turn = None) -> Board:
    """Devuelve una instancia de Board con el turno especificado
    """
    if turn is not None: # A turn was specified
        parts = fen.split(' ')
        parts[1] = 'w' if turn else 'b'
        fen = ' '.join(parts)

    board = Board(fen)
    return board

def get_legal_moves_san(fen):
    """
    Devuelve una lista con todos los movimientos legales de cada pieza en el tablero
    para el jugador en turno

    Args:
    fen (str): La cadena FEN que representa la posición actual del tablero.

    Returns:
    list: Lista de las casillas de puntos de presión en notación SAN.
    """
    legal_moves = {}
    # Fill legal moves according one piece type
    for piece_type in PIECE_TYPES:
        legal_moves[piece_type] = get_legal_moves_from_piece_type(fen, piece_type)
    return legal_moves

def get_legal_moves_from_piece_type(fen, piece_type, turn = None):
    """
    Devuelve los movimientos legales de una pieza dada en el turno correspondiente a la posición fen

    Args:
    fen (str): The FEN string representing the current board position.
    piece_type (int):
        PAWN = 1
        KNIGHT = 2
        BISHOP = 3
        ROOK = 4
        QUEEN = 5
        KING = 6

    Returns:
    list: A list of legal moves for given piece in uci notation.
    """
    board = get_custom_board(fen, turn)
    legal_moves = board.legal_moves
    moves = [board.san(move) for move in legal_moves if board.piece_at(move.from_square).piece_type == piece_type] # move.uci for uci format
    return sorted(moves)

def count_legal_moves_from_piece_type(fen, piece_type, turn = None):
    """
    Cuenta el número de movimientos legales de una pieza dada en el turno correspondiente a la posición fen

    Args:
    fen (str): The FEN string representing the current board position.
    piece_type (int):
        PAWN = 1
        KNIGHT = 2
        BISHOP = 3
        ROOK = 4
        QUEEN = 5
        KING = 6

    Returns:
    list: A list of legal moves for given piece in uci notation.
    """
    board = get_custom_board(fen, turn)
    legal_moves = board.legal_moves
    moves = [1 for move in legal_moves if board.piece_at(move.from_square).piece_type == piece_type] # move.uci for uci format
    return sum(moves)

def get_pawn_capture_squares(fen, turn:bool = True):
    """Obtiene las casillas controladas por peon (puede comer en siguiente turno) 

    Args:
        fen (str): Cadena Fen 
        turn (bool, optional): True: Blancas. False: Negras

    Returns:
        set: Obtiene las casillas controladas por peon (puede comer en siguiente turno)
    """
    # Dividir la cadena FEN para obtener la disposición de las piezas en el tablero
    board_setup = fen.split()[0]

    # Convertir la disposición del tablero a una matriz 8x8 para facilitar el acceso a cada casilla
    rows = board_setup.split('/')
    board = [list(row.replace('8', '........')
                  .replace('7', '.......')
                  .replace('6', '......')
                  .replace('5', '.....')
                  .replace('4', '....')
                  .replace('3', '...')
                  .replace('2', '..')
                  .replace('1', '.')) for row in rows]
    captures = {'white': set(), 'black': set()} 
    # Movimiento diagonal para captura
    direction_white = -1
    direction_black = 1
    diagonals = [-1, 1] # Diagonales izquierda y derecha
    for row in range(8):
        for col in range(8):
            if board[row][col] == 'P':  # Peón Blancas
                for dcol in diagonals:
                    nrow, ncol = row + direction_white, col + dcol
                    if 0 <= nrow < 8 and 0 <= ncol < 8:  # Verificar límites del tablero
                        captures['white'].add(chr(97 + ncol) + str(8 - nrow))
            elif board[row][col] == 'p':  # Peón Negras
                for dcol in diagonals:
                    nrow, ncol = row + direction_black, col + dcol
                    if 0 <= nrow < 8 and 0 <= ncol < 8:  # Verificar límites del tablero
                        captures['black'].add(chr(97 + ncol) + str(8 - nrow))               
    captures_white = sorted(list(captures['white'])) # Ordenar alfabeticamente
    captures_black = sorted(list(captures['black']))
                        
    return captures_white if  turn else captures_black

def count_pawn_capture_squares(fen, turn:bool = True):
    """Cuenta las casillas controladas por peon (puede comer en siguiente turno) 

    Args:
        fen (str): Cadena Fen 
        turn (bool, optional): True: Blancas. False: Negras

    Returns:
        int: Cuenta las casillas controladas por peon (puede comer en siguiente turno)
    """              
    return len(get_pawn_capture_squares(fen, turn))

def get_pressure_points_san(fen, turn = None):
    """
    Devuelve una lista de las casillas que son puntos de presión en notación SAN.
    Un punto de presión se define como una casilla que es atacada por más de una pieza
    del jugador cuyo turno es actualmente.

    Args:
    fen (str): La cadena FEN que representa la posición actual del tablero.

    Returns:
    list: Lista de las casillas de puntos de presión en notación SAN.
    """
    board = get_custom_board(fen, turn)
    current_turn = board.turn # TRUE (WHITE) FALSE (BLACK)
    attack_map = {square: 0 for square in SQUARES}

    # Iterar solo sobre las piezas del color que tiene el turno
    for piece_type in PIECE_TYPES:
        for square in board.pieces(piece_type, current_turn):
            attacked_squares = board.attacks(square)
            for attacked_square in attacked_squares:
                attacked_piece = board.piece_at(attacked_square)
                # Incrementar si la casilla atacada está vacía o tiene una pieza del color opuesto
                if not attacked_piece or attacked_piece.color != current_turn:
                    attack_map[attacked_square] += 1

    # Los puntos de presión son cuadrados a los que más de una pieza ataca
    pressure_points = [square_name(sq) for sq, count in attack_map.items() if count > 1]

    return sorted(pressure_points)

def count_pressure_points(fen, turn = None):
    """
    Cuenta las casillas que son puntos de presión.
    Un punto de presión se define como una casilla que es atacada por más de una pieza
    del jugador cuyo turno es actualmente.

    Args:
    fen (str): La cadena FEN que representa la posición actual del tablero.

    Returns:
    int: Conteo de las casillas de puntos de presión.
    """
    board = get_custom_board(fen, turn)
    current_turn = board.turn # TRUE (WHITE) FALSE (BLACK)
    attack_map = {square: 0 for square in SQUARES}

    # Iterar solo sobre las piezas del color que tiene el turno
    for piece_type in PIECE_TYPES:
        for square in board.pieces(piece_type, current_turn):
            attacked_squares = board.attacks(square)
            for attacked_square in attacked_squares:
                attacked_piece = board.piece_at(attacked_square)
                # Incrementar si la casilla atacada está vacía o tiene una pieza del color opuesto
                if not attacked_piece or attacked_piece.color != current_turn:
                    attack_map[attacked_square] += 1

    # Los puntos de presión son cuadrados a los que más de una pieza ataca
    pressure_points = [1 for _, count in attack_map.items() if count > 1]

    return sum(pressure_points)
        
def get_all_controlled_diagonals(fen, turn = None):
    """
    Obtiene el número de diagonales controladas por alfiles o reinas en un tablero de ajedrez,
    dado por la notación FEN.
    Una diagonal está controlada si al menos dos casillas están libres o contienen una 
    pieza del color opuesto.
    """
    # Crear un tablero a partir de la notación FEN
    tablero = get_custom_board(fen, turn)
    current_turn = tablero.turn # TRUE (WHITE) FALSE (BLACK)
    diagonales_controladas = {}
    diagonales_controladas_sum  = 0

    # Por cada casilla entre las 64 posibles
    for piece_type in BISHOP, QUEEN:
        for square in tablero.pieces(piece_type, current_turn):
            # Sumar diagonales controladas de alfiles y reinas
            result = get_controlled_diagonals_by_square(square, tablero)
            diagonales_controladas[square_name(square)] = result
            diagonales_controladas_sum += result

    return diagonales_controladas

def count_all_controlled_diagonals(fen, turn = None):
    """
    Cuenta el número de diagonales controladas por alfiles o reinas en un tablero de ajedrez,
    dado por la notación FEN.
    Una diagonal está controlada si al menos dos casillas están libres o contienen una 
    pieza del color opuesto.
    """
    # Crear un tablero a partir de la notación FEN
    tablero = get_custom_board(fen, turn)
    current_turn = tablero.turn # TRUE (WHITE) FALSE (BLACK)
    diagonales_controladas_sum  = 0

    # Por cada casilla entre las 64 posibles
    for piece_type in BISHOP, QUEEN:
        for square in tablero.pieces(piece_type, current_turn):
            # Sumar diagonales controladas de alfiles y reinas
            result = get_controlled_diagonals_by_square(square, tablero)
            diagonales_controladas_sum += result

    return diagonales_controladas_sum

def get_controlled_diagonals_by_square(casilla, board: Board):
    # Direcciones de las diagonales: superior izquierda, superior derecha, inferior izquierda, inferior derecha
    direcciones = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    controlled_diagonals = 0
    # Por cada diagonal en las direcciones
    for dx, dy in direcciones:
        # Verificar las dos primeras casillas en cada dirección diagonal
        for i in range(1, 3):
            nueva_fila, nueva_columna = square_rank(casilla) + i * dx, square_file(casilla) + i * dy
            # Si alguna de las dos casillas está fuera del tablero, la diagonal no esta  controlada o es inutil
            if not (0 <= nueva_fila <= 7 and 0 <= nueva_columna <= 7):
                break
            nueva_casilla = square(nueva_columna, nueva_fila)
            pieza_original = board.piece_at(casilla)
            pieza_en_diagonal = board.piece_at(nueva_casilla)

            # Si alguna de las dos casillas tiene una pieza del mismo color, la diagonal no está controlada o no es util
            if pieza_en_diagonal and pieza_en_diagonal.color == pieza_original.color:
                break
        else:
            controlled_diagonals += 1
    return controlled_diagonals

def get_all_controlled_lines(fen, turn = None):
    """
    Cuenta el número de líneas horizontales y verticales controladas por torres y reina en un tablero de ajedrez,
    dada una determinada posición FEN
    """
    # Crear un tablero a partir de la notación FEN
    tablero = get_custom_board(fen, turn)
    current_turn = tablero.turn # TRUE (WHITE) FALSE (BLACK)
    lineas_controladas = {}
    lineas_controladas_sum = 0

    # Por cada torre del jugador actual
    for piece_type in ROOK, QUEEN:
        for square in tablero.pieces(piece_type, current_turn):
            # Sumar líneas horizontales y verticales controladas por la pieza
            result = get_controlled_lines_by_square(square, tablero)
            lineas_controladas[square_name(square)] = result
            lineas_controladas_sum += result

    return lineas_controladas

def count_all_controlled_lines(fen, turn = None):
    """
    Cuenta el número de líneas horizontales y verticales controladas por torres y reina en un tablero de ajedrez,
    dada una determinada posición FEN
    """
    # Crear un tablero a partir de la notación FEN
    tablero = get_custom_board(fen, turn)
    current_turn = tablero.turn # TRUE (WHITE) FALSE (BLACK)
    lineas_controladas_sum = 0

    # Por cada torre del jugador actual
    for piece_type in ROOK, QUEEN:
        for square in tablero.pieces(piece_type, current_turn):
            # Sumar líneas horizontales y verticales controladas por la pieza
            result = get_controlled_lines_by_square(square, tablero)
            lineas_controladas_sum += result

    return lineas_controladas_sum

def get_controlled_lines_by_square(casilla, board: Board):
    """
    Cuenta el numero de lineas controladas por determinada casilla   
    Una linea está controlada si al menos dos casillas contiguas a la pieza están libres o contienen una 
    pieza del color opuesto.
    """
    # Direcciones: izquierda, derecha, arriba, abajo
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    controlled_lines = 0
    # Por cada dirección
    for dx, dy in direcciones:
        # Verificar las dos primeras casillas en cada dirección
        for i in range(1, 3):
            nueva_fila, nueva_columna = square_rank(casilla) + i * dy, square_file(casilla) + i * dx
            # Si alguna de las dos casillas está fuera del tablero, la línea no está controlada o es inútil
            if not (0 <= nueva_fila <= 7 and 0 <= nueva_columna <= 7):
                break
            nueva_casilla = square(nueva_columna, nueva_fila)
            pieza_original = board.piece_at(casilla)
            pieza_en_linea = board.piece_at(nueva_casilla)

            # Si alguna de las dos casillas tiene una pieza del mismo color, la línea no está controlada o no es útil
            if pieza_en_linea and pieza_en_linea.color == pieza_original.color:
                break
        else:
            controlled_lines += 1
    return controlled_lines

def get_fen_from_moves(moves):
    game = Board()
    for move in moves.split():
        game.push_san(move)
    
    # Get FEN string of position
    fen = game.fen()
    return fen

def get_current_turn(fen):
    # Crear un tablero a partir de la notación FEN
    tablero = Board(fen)
    current_turn = tablero.turn # TRUE (WHITE) FALSE (BLACK)
    return current_turn

def get_current_opening(df,turn_column_name, fen, opening_move):
    """
    Filtra el DataFrame proporcionado y retorna el opening producido del turno.
    
    Parámetros:
    - df: DataFrame a filtrar.
    - turn_column_name: Name of column, last played
    - opening_move: Entero, número de jugada, (2 jugadas por turno).
    
    Retorna:
    - Nombre de la apertura alcanzada
    """
    if df.empty:
        return None
    turn_column_name = f"{turn_column_name}_fen"
    
    # Verificar si la columna generada existe en el DataFrame filtrado
    if turn_column_name not in df.columns:
        return None
    
    df_filter = df[(df[turn_column_name]==fen) & (df['opening_moves']==opening_move)]
    if df_filter.empty:
        return None
    return df_filter['opening_fullname'].iloc[0]

def select_move_by_weighted_choice(weights):
    """
    Selecciona un movimiento de ajedrez basado en una lista de pesos para cada movimiento.
    
    Parámetros:
    - weights: Lista de tuplas, donde cada tupla contiene un movimiento (como 'e4') y su peso asociado.
    
    Retorna:
    - Movimiento seleccionado de manera ponderada.
    """
    # Desempaquetar la lista de tuplas en movimientos y sus respectivos pesos
    moves, move_weights = zip(*weights)
    
    # Seleccionar un movimiento de manera ponderada basada en los pesos
    selected_move = random.choices(moves, weights=move_weights, k=1)[0]
    
    return selected_move

def count_all_features(fen:str, moves:int) -> pd.DataFrame:
    '''
    Obtiene todas las features utilizadas para una predicción
    # fen = board.fen()
    # moves = len(board.move_stack) or board.ply()
    '''
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)
    
    df = pd.DataFrame({
        'turns': moves,
        'w_ctrld_pawn':           count_pawn_capture_squares(fen, True),
        'w_ctrld_knight':         count_legal_moves_from_piece_type(fen, KNIGHT, True),
        'w_ctrld_bishop':         count_legal_moves_from_piece_type(fen, BISHOP, True),
        'w_ctrld_rook':           count_legal_moves_from_piece_type(fen, ROOK, True),
        'w_ctrld_queen':          count_legal_moves_from_piece_type(fen, QUEEN, True),
        'w_ctrld_king':           count_legal_moves_from_piece_type(fen, KING, True),
        'w_preassure_points':     count_pressure_points(fen, True),
        'w_ctrld_diagonals': count_all_controlled_diagonals(fen, True),
        'w_ctrld_lines':     count_all_controlled_lines(fen, True),
        'b_ctrld_pawn':           count_pawn_capture_squares(fen, False),
        'b_ctrld_knight':         count_legal_moves_from_piece_type(fen, KNIGHT, False),
        'b_ctrld_bishop':         count_legal_moves_from_piece_type(fen, BISHOP, False),
        'b_ctrld_rook':           count_legal_moves_from_piece_type(fen, ROOK, False),
        'b_ctrld_queen':          count_legal_moves_from_piece_type(fen, QUEEN, False),
        'b_ctrld_king':           count_legal_moves_from_piece_type(fen, KING, False),
        'b_preassure_points':     count_pressure_points(fen, False),
        'b_ctrld_diagonals': count_all_controlled_diagonals(fen, False),
        'b_ctrld_lines':     count_all_controlled_lines(fen, False)
        }, index=[0])
    return df

def get_all_features(fen:str, moves:int) -> pd.DataFrame:
    '''
    Obtiene todas las features utilizadas para una predicción
    # fen = board.fen()
    # moves = len(board.move_stack) or board.ply()
    '''
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)
    
    df = pd.DataFrame({
        'turns': moves,
        'w_ctrld_pawn': str(get_pawn_capture_squares(fen, True)),
        'w_ctrld_knight': str(get_legal_moves_from_piece_type(fen, KNIGHT, True)),
        'w_ctrld_bishop': str(get_legal_moves_from_piece_type(fen, BISHOP, True)),
        'w_ctrld_rook': str(get_legal_moves_from_piece_type(fen, ROOK, True)),
        'w_ctrld_queen': str(get_legal_moves_from_piece_type(fen, QUEEN, True)),
        'w_ctrld_king': str(get_legal_moves_from_piece_type(fen, KING, True)),
        'w_preassure_points': str(get_pressure_points_san(fen, True)),
        'w_ctrld_diagonals': str(get_all_controlled_diagonals(fen, True)),
        'w_ctrld_lines': str(get_all_controlled_lines(fen, True)),
        'b_ctrld_pawn': str(get_pawn_capture_squares(fen, False)),
        'b_ctrld_knight': str(get_legal_moves_from_piece_type(fen, KNIGHT, False)),
        'b_ctrld_bishop': str(get_legal_moves_from_piece_type(fen, BISHOP, False)),
        'b_ctrld_rook': str(get_legal_moves_from_piece_type(fen, ROOK, False)),
        'b_ctrld_queen': str(get_legal_moves_from_piece_type(fen, QUEEN, False)),
        'b_ctrld_king': str(get_legal_moves_from_piece_type(fen, KING, False)),
        'b_preassure_points': str(get_pressure_points_san(fen, False)),
        'b_ctrld_diagonals': str(get_all_controlled_diagonals(fen, False)),
        'b_ctrld_lines': str(get_all_controlled_lines(fen, False))
        }, index=[0])
    return df

def get_all_features_uf(fen: str, turn: bool) -> pd.DataFrame:
    '''
    Obtiene todas las features utilizadas para una predicción
    # turn: TRUE (WHITE) FALSE (BLACK)
    '''
    # Características a obtener
    ctrld_pawn = get_pawn_capture_squares(fen, True)
    ctrld_knight = get_legal_moves_from_piece_type(fen, KNIGHT, True)
    ctrld_bishop = get_legal_moves_from_piece_type(fen, BISHOP, True)
    ctrld_rook = get_legal_moves_from_piece_type(fen, ROOK, True)
    ctrld_queen = get_legal_moves_from_piece_type(fen, QUEEN, True)
    ctrld_king = get_legal_moves_from_piece_type(fen, KING, True)
    preassure_points = get_pressure_points_san(fen, True)
    ctrld_diagonals = get_all_controlled_diagonals(fen, True)
    ctrld_lines = get_all_controlled_lines(fen, True)

    features = [
        ('Casillas controladas por peon',ctrld_pawn, len(ctrld_pawn)),
        ('Casillas controladas por caballo',ctrld_knight, len(ctrld_knight)),
        ('Casillas controladas por alfil', ctrld_bishop, len(ctrld_bishop)),
        ('Casillas controladas por torre', ctrld_rook, len(ctrld_rook)),
        ('Casillas controladas por reina', ctrld_queen, len(ctrld_queen)),
        ('Casillas controladas por el rey', ctrld_king, len(ctrld_king)),
        ('Puntos de presion (>1 atacando)', preassure_points, len(preassure_points)),
        ('Diagonales controladas', ctrld_diagonals, sum(ctrld_diagonals.values())),
        ('Lineas controladas', ctrld_lines, sum(ctrld_lines.values())),
    ]
    
    # Lista para almacenar las filas del DataFrame
    rows = []

    # Iterar sobre cada característica para ambos colores
    for feature_name, feature_values, total in features:
        # total = len(feature_values)

        # Añadir fila al DataFrame
        rows.append({
            'Total': total,
            'Caracteristica': feature_name,
            'Casillas': str(feature_values)
        })

    return pd.DataFrame(rows)