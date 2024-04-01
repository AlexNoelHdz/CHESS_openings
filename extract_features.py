"""
Este archivo de python tiene la intención de proveer una serie de metodos para extraer caracteristicas 
desde posiciones dadas por cadenas FEN. 
"""
from chess import Board
from chess import square_rank, square_file, square
from chess import PIECE_NAMES, PIECE_TYPES, SQUARES, square_name
from chess import BISHOP, ROOK, QUEEN
import random

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

def get_legal_moves_from_piece_type(fen, piece_type):
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
    board = Board(fen)
    legal_moves = board.legal_moves
    moves = [board.san(move) for move in legal_moves if board.piece_at(move.from_square).piece_type == piece_type] # move.uci for uci format
    return moves

def count_legal_moves_from_piece_type(fen, piece_type):
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
    board = Board(fen)
    legal_moves = board.legal_moves
    moves = [1 for move in legal_moves if board.piece_at(move.from_square).piece_type == piece_type] # move.uci for uci format
    return sum(moves)

def get_pressure_points_san(fen):
    """
    Devuelve una lista de las casillas que son puntos de presión en notación SAN.
    Un punto de presión se define como una casilla que es atacada por más de una pieza
    del jugador cuyo turno es actualmente.

    Args:
    fen (str): La cadena FEN que representa la posición actual del tablero.

    Returns:
    list: Lista de las casillas de puntos de presión en notación SAN.
    """
    board = Board(fen)
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

    return pressure_points

def count_pressure_points(fen):
    """
    Cuenta las casillas que son puntos de presión.
    Un punto de presión se define como una casilla que es atacada por más de una pieza
    del jugador cuyo turno es actualmente.

    Args:
    fen (str): La cadena FEN que representa la posición actual del tablero.

    Returns:
    int: Conteo de las casillas de puntos de presión.
    """
    board = Board(fen)
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
        
def get_all_controlled_diagonals(fen):
    """
    Obtiene el número de diagonales controladas por alfiles o reinas en un tablero de ajedrez,
    dado por la notación FEN.
    Una diagonal está controlada si al menos dos casillas están libres o contienen una 
    pieza del color opuesto.
    """
    # Crear un tablero a partir de la notación FEN
    tablero = Board(fen)
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

    return (diagonales_controladas, diagonales_controladas_sum)

def count_all_controlled_diagonals(fen):
    """
    Cuenta el número de diagonales controladas por alfiles o reinas en un tablero de ajedrez,
    dado por la notación FEN.
    Una diagonal está controlada si al menos dos casillas están libres o contienen una 
    pieza del color opuesto.
    """
    # Crear un tablero a partir de la notación FEN
    tablero = Board(fen)
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

def get_all_controlled_lines(fen):
    """
    Cuenta el número de líneas horizontales y verticales controladas por torres y reina en un tablero de ajedrez,
    dada una determinada posición FEN
    """
    # Crear un tablero a partir de la notación FEN
    tablero = Board(fen)
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

    return (lineas_controladas, lineas_controladas_sum)

def count_all_controlled_lines(fen):
    """
    Cuenta el número de líneas horizontales y verticales controladas por torres y reina en un tablero de ajedrez,
    dada una determinada posición FEN
    """
    # Crear un tablero a partir de la notación FEN
    tablero = Board(fen)
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

def get_unique_opening_moves(df, turno, fullmove_number):
    """
    Filtra el DataFrame basado en el 'opening_shortname' proporcionado y retorna los valores únicos
    del turno.
    
    Parámetros:
    - df: DataFrame a filtrar.
    - turno: Booleano, True para WHITE, False para BLACK.
    - fullmove_number: Entero, número de la jugada de la partida de ajedrez.
    - opening_shortname: String, nombre de la apertura a filtrar.
    
    Retorna:
    - Lista de valores únicos de la columna 'turn_column_name' o None si la columna no existe.
    """
    if df.empty:
        return None, None
    turno_str = "0w" if turno else "1b"
    turn_column_name = f"{turno_str}_{fullmove_number}"
    
    # Verificar si la columna generada existe en el DataFrame filtrado
    if turn_column_name in df.columns:
        # Calcular la frecuencia de cada valor único
        value_counts = df[turn_column_name].value_counts(normalize=True) # Valores entre 0 y 1

        # Normalización para aumentar la probabilidad de que la máquina te juegue jugadas poco comunes
        # 0-5: 1
        # 5-50: 5
        # 50-100: 20
        determine_weight = lambda proportion: 1 if proportion <= 0.05 else \
                                           5 if proportion <= 0.50 else 20

        # Crear la lista de tuplas (valor único, peso normalizado) usando la función lambda
        weights = [(value, determine_weight(proportion)) for value, proportion in value_counts.items()]
    
        return weights, turn_column_name
    else:
        return None, None

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

# # Example FEN string
# # fen_example = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# # fen_example = 'rnbqkb1r/p6p/6pn/1p1pB3/1p6/N2Q2P1/P1P1PP1P/1R2KBNR w Kkq - 0 12'
# fen_example = 'rnbqk1nr/pp4pp/2p1p3/b7/3P1B2/2N2N2/PP2PPPP/R2QKB1R b KQkq - 5 7'

# print(f"Obteniendo datos de la posición FEN: {fen_example}")

# for piece_type in PIECE_TYPES:
#     legal_moves = get_legal_moves_from_piece_type(fen_example, piece_type)
#     print(f"{PIECE_NAMES[piece_type]}'s legal moves({len(legal_moves)}): {legal_moves}")

# # for piece_type in PIECE_TYPES:
# #     count_legal_moves = count_legal_moves_from_piece_type(fen_example, piece_type)
# #     print(f"{PIECE_NAMES[piece_type]}'s legal moves({count_legal_moves}).")

# pressure_points = get_pressure_points_san(fen_example)
# print(f"Puntos de presión ({len(pressure_points)}): {pressure_points}")

# # count_pp = count_pressure_points(fen_example)
# # print(f"Puntos de presión ({count_pp})")

# diagonals, diagonals_sum = get_all_controlled_diagonals(fen_example)
# print(f"Diagonales controladas ({diagonals_sum}): {diagonals}")

# lines, lines_sum = get_all_controlled_lines(fen_example)
# print(f"Lineas controladas ({lines_sum}): {lines}")

# moves = "e4 e5 Nf3 d6 d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 Nf6 Bg5 O-O b5 Nc5 Bxf6 Bxf6 Bd3 Qd7 O-O Nxd3 Qxd3 c6 a4 cxd5 Nxd5 Qe6 Nc7 Qg4 Nxa8 Bd7 Nc7 Rc8 Nd5 Qg6 Nxf6+ Qxf6 Rfd1 Re8 Qxd6 Bg4 Qxf6 gxf6 Rd3 Bxf3 Rxf3 Rd8 Rxf6 Kg7 Rf3 Rd2 Rg3+ Kf8 c3 Re2 f3 Rc2 Rg5 f6 Rh5 Kg7 Rd1 Kg6 Rh3 Rxc3 Rd7 Rc1+ Kf2 Rc2+ Kg3 h5 Rxb7 Kg5 Rxa7 h4+ Rxh4 Rxg2+ Kxg2 Kxh4 b6 Kg5 b7 f5 exf5 Kxf5 b8=Q e4 Rf7+ Kg5 Qg8+ Kh6 Rh7#"
# print(f"Fen from moves: {get_fen_from_moves(moves)}")


