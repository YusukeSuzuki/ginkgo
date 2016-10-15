import shogi as sh
import numpy as np

# 'a-Z' is position dimention
# '_' is movement dimention
# promotable piece has two movement dimentions
# k_r_r_R_R_b_b_B_B_g_g_g_g_s_s_s_s_S_S_S_S_n_n_n_n_N_N_N_N_l_l_l_l_L_L_L_L_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_
# k_r_r_R_R_b_b_B_B_g_g_g_g_s_s_s_s_S_S_S_S_n_n_n_n_N_N_N_N_l_l_l_l_L_L_L_L_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_p_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_P_
# 9*9*180*2

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

PROMS = {
    PT2VM[sh.ROOK]: PT2VM[sh.PROM_ROOK],
    PT2VM[sh.BISHOP]: PT2VM[sh.PROM_BISHOP],
    PT2VM[sh.SILVER]: PT2VM[sh.PROM_SILVER],
    PT2VM[sh.KNIGHT]: PT2VM[sh.PROM_KNIGHT],
    PT2VM[sh.LANCE]: PT2VM[sh.PROM_LANCE],
    PT2VM[sh.PAWN]: PT2VM[sh.PROM_PAWN],
    }

IN_HANDS_TYPE = { 1: 'R', 3: 'B', 5: 'G', 6: 'S', 8: 'N', 10:'L', 12:'P', }

#            k  r  R   b   B   g   s   S   n   N   l   L   p    P
TypeHeads = [0, 2, 8, 12, 18, 22, 30, 42, 50, 62, 70, 82, 90, 144,] # 180
TypeSkips = [2, 3, 2,  3,  2,  2,  3,  2,  3,  2,  3,  2,  3,   2,] # 180

def skipped_board(board):
    sfen = board.sfen()
    splitted = sfen.split(' ')
    splitted[1] = 'w' if splitted[1] == 'b' else 'b'
    return sh.Board( ' '.join(splitted) )

def sqr2rc(square):
    return (square // 9, square % 9)

def skipped_sfen(sfen):
    splitted = sfen.split(' ')
    splitted[1] = 'w' if splitted[1] == 'b' else 'b'
    return ' '.join(splitted)

def sfen_to_vector(sfen, usi=None, debug=False):
    initial_board = sh.Board(sfen)

    if usi is not None:
        initial_board.push_usi(usi)

    boards = [ initial_board, sh.Board(skipped_sfen(sfen)), ]
    sides = [0,1]

    # make moves and drops
    moves = [ [[] for i in range(81)], [[] for i in range(81)] ]
    drops = [ [[]] * len(PT2VM), [[]] * len(PT2VM) ]

    for side in sides:
        for move in boards[side].legal_moves:
            if move.from_square is not None:
                moves[side][move.from_square].append( move )
            else:
                drops[side][PT2VM[move.drop_piece_type]].append( move )

    # make pieces
    pieces = [[],[]]

    for sqr in range(81):
        p = boards[0].piece_at(sqr)

        if p is None: continue

        pieces[p.color].append( (sqr, PT2VM[p.piece_type], moves[p.color][sqr]) )

    for side in sides:
        for key, p in boards[0].pieces_in_hand[side].most_common():
            for i in range(p):
                pieces[side].append( (None, PT2VM[key], drops[side][PT2VM[key]]) )

    vec = np.zeros([1,9,9,360])

    p_cnt = [[0] * len(PT2VM), [0] * len(PT2VM)]
    ch_base = [0,180]

    for side in sides:
        for p in pieces[side]:
            pos_type = p[1]

            # position channel
            chB = ch_base[side] +  TypeHeads[p[1]] + p_cnt[side][p[1]] * TypeSkips[p[1]]
            # movement channel
            chM = ch_base[side] +  TypeHeads[p[1]] + p_cnt[side][p[1]] * TypeSkips[p[1]] + 1
            # movement with promotion channel
            chP = ch_base[side] +  TypeHeads[p[1]] + p_cnt[side][p[1]] * TypeSkips[p[1]] + 2

            if p[0] is not None:
                to = sqr2rc(p[0])
                vec[0][to[0]][to[1]][chB] = 1

            for m in p[2]:
                to = sqr2rc(m.to_square)
                tmp_ch = chM if not m.promotion else chP
                vec[0][to[0]][to[1]][tmp_ch] = 1

            if pos_type is not None: p_cnt[side][pos_type] += 1

    if debug:
        vec = np.transpose(vec, [0,3,1,2])

    return vec

def to_debug(vec):
    return np.transpose(vec, [0,3,1,2])

def vector_to_usi_movement():
    pass

def fliplr(vec):
    res = np.zeros(vec.shape)

    for r in range(9):
        for c in range(9):
            res[0][r][8-c] = vec[0][r][c]

    return res

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    board = sh.Board()
    #vec = sfen_to_vector(board.sfen(), debug=True)
    vec = sfen_to_vector(board.sfen())
    #vec = fliplr(vec)
    #vec = np.rot90(vec)
    #vec[:,[0,1]] = vec[:,[1,0]]
    vec = to_debug(vec)
    print(vec)

