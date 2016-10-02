import shogi as sh
import numpy as np

# 'a-Z' is position dimention
# '_' is movement dimention
# k_k_r_r_R_R_b_b_B_B_g_g_g_g_s_s_s_s_S_S_S_S_n_n_n_n_N_N_N_N_l_l_l_l_L_L_L_L_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_

'''
PieceType2VectorMap
k r R b B g s S n N l L p P 
'''
PT2VM = {
    sh.KING: 0,
    sh.ROOK: 1,
    sh.PROM_ROOK: 2,
    sh.BISHOP: 3,
    sh.PROM_BISHOP: 4,
    sh.GOLD: 5,
    sh.SILVER: 6,
    sh.PROM_SILVER: 7,
    sh.KNIGHT: 8,
    sh.PROM_KNIGHT: 9,
    sh.LANCE: 10,
    sh.PROM_LANCE: 11,
    sh.PAWN: 12,
    sh.PROM_PAWN: 13,
    }

IN_HANDS_TYPE = { 1: 'R', 3: 'B', 5: 'G', 6: 'S', 8: 'N', 10:'L', 12:'P', }
TypeHeads = [0, 4, 8, 12, 16, 20, 28, 36, 44, 52, 60, 68, 76, 112,]

def skipped_board(board):
    sfen = board.sfen()
    splitted = sfen.split(' ')
    splitted[1] = 'w' if splitted[1] == 'b' else 'b'
    return sh.Board( ' '.join(splitted) )

def sqr2rc(square):
    return (square // 9, square % 9)

def sfen_to_vector(sfen, debug=False):
    return board_to_vector(sh.Board(sfen), debug)

def board_to_vector(board, debug=False):
    sides = [ [[] for i in range(len(PT2VM))], [[] for i in range(len(PT2VM))] ]
    hands = [ [0] * len(PT2VM), [0] * len(PT2VM) ]

    for r in range(0,9):
        for c in range(0,9):
            p = board.piece_at(r * 9 + c)
            if p is None: continue

            side = 0 if p.color == 0 else 1
            sides[side][PT2VM[p.piece_type]] = sides[side][PT2VM[p.piece_type]] + [(r,c)]

    for i in range(len(board.pieces_in_hand)):
        for key, value in board.pieces_in_hand[i].items():
            hands[i][PT2VM[key]] = value

    moves = [board.legal_moves, skipped_board(board).legal_moves]

    if not debug:
        vec = np.zeros([1,9,9,148])
    else:
        vec = np.zeros([148,9,9])

    p_counts = [0] * len(PT2VM)

    side_value = [1,-1]

    for i, side in [(i, side) for i in range(len(PT2VM)) for side in range(len(sides))]:
        for piece in sides[side][i]:
            ''' pieces on board '''
            ''' placement '''
            vec_i = TypeHeads[i] + p_counts[i] * 2
            
            if not debug:
                vec[0][piece[0]][piece[1]][vec_i] = side_value[side]
            else:
                vec[vec_i][piece[0]][piece[1]] = side_value[side]

            ''' movement '''
            vec_i = vec_i + 1
            vec_ip = 0 if i+1 == len(PT2VM) else TypeHeads[i+1] + p_counts[i+1] * 2 + 1
            pos_sqr = piece[0] * 9 + piece[1]

            prom_found = False

            for move in moves[side]:
                if move.from_square == pos_sqr:
                    to_pos = sqr2rc(move.to_square)
                    temp_vec_i =  vec_ip if move.promotion else vec_i

                    if not debug:
                        vec[0][to_pos[0]][to_pos[1]][temp_vec_i] = side_value[side]
                    else:
                        vec[temp_vec_i][to_pos[0]][to_pos[1]] = side_value[side]

                    prom_found = prom_found or move.promotion

            p_counts[i] = p_counts[i] + 1
            if prom_found:
                p_counts[i+1] = p_counts[i+1] + 1

        for j in range(hands[side][i]):
            ''' pieces in hand '''
            vec_i = TypeHeads[i] + p_counts[i] * 2 + 1
            for move in moves[side]:
                if str(move)[0] != IN_HANDS_TYPE[i]: continue

                to_pos = sqr2rc(move.to_square)

                if not debug:
                    vec[0][to_pos[0]][to_pos[1]][vec_i] = side_value[side]
                else:
                    vec[vec_i][to_pos[0]][to_pos[1]] = side_value[side]

            p_counts[i] = p_counts[i] + 1

    return vec

def vector_to_usi_movement():
    pass

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    board = sh.Board()
    vec = sfen_to_vector(board.sfen())
    print(vec)

