from chess import Board
from chess import square_rank, square_file, square
from chess import PIECE_NAMES, PIECE_TYPES, SQUARES, square_name
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

def get_legal_moves_from_piece_type(fen, piece_type):
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
    moves = [board.san(move) for move in legal_moves if board.piece_at(move.from_square).piece_type == piece_type] # move.uci for uci format
    return moves

def pressure_points_san(fen):
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
        
def get_all_controled_diagonals(fen):
    """
    Cuenta el número de diagonales controladas por alfiles o reinas en un tablero de ajedrez,
    dado por la notación FEN.
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
            if not (1 <= nueva_fila <= 8 and 1 <= nueva_columna <= 8):
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


# Example FEN string
# fen_example = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen_example = 'rnbqkbnr/ppp2ppp/4p3/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 3'

print(f"Obteniendo datos de la posición FEN: {fen_example}")
legal_moves = get_legal_moves_san(fen_example)
for piece_type in PIECE_TYPES:
    print(f"{PIECE_NAMES[piece_type]}'s legal moves({len(legal_moves[piece_type])}): {legal_moves[piece_type]}")

pressure_points = pressure_points_san(fen_example)
print(f"Puntos de presión ({len(pressure_points)}): {pressure_points}")

diagonals_fen_test = "rnbqkb1r/pp4pp/4pp1n/2pp4/3P1B2/3Q2P1/PPP1PP1P/RN2KBNR w KQkq - 0 7"
diagonals, diagonals_sum = get_all_controled_diagonals(diagonals_fen_test)
print(f"Diagonales controladas ({diagonals_sum}): {diagonals}")