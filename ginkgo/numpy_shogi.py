from itertools import product
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

VM2PT = { v: k for k,v in PT2VM.items()}

PROMS = {
    PT2VM[sh.ROOK]: PT2VM[sh.PROM_ROOK],
    PT2VM[sh.BISHOP]: PT2VM[sh.PROM_BISHOP],
    PT2VM[sh.SILVER]: PT2VM[sh.PROM_SILVER],
    PT2VM[sh.KNIGHT]: PT2VM[sh.PROM_KNIGHT],
    PT2VM[sh.LANCE]: PT2VM[sh.PROM_LANCE],
    PT2VM[sh.PAWN]: PT2VM[sh.PROM_PAWN],
    }

IN_HANDS_TYPE = { 1: 'R', 3: 'B', 5: 'G', 6: 'S', 8: 'N', 10:'L', 12:'P', }

#            0  1  2   3   4   5   6   7   8   9  10  11  12   13
#            k  r  R   b   B   g   s   S   n   N   l   L   p    P
TypeHeads = [0, 2, 8, 12, 18, 22, 30, 42, 50, 62, 70, 82, 90, 144,] # 180
TypeSkips = [2, 3, 2,  3,  2,  2,  3,  2,  3,  2,  3,  2,  3,   2,] # 180
TypeCounts= [1, 2, 2,  2,  2,  4,  4,  4,  4,  4,  4,  4, 18,  18,]

def create_position_channels():
    index = 0
    positions = {}

    for color in [0,1]:
        for type_i, counts in enumerate(TypeCounts):
            for n in range(counts):
                positions[index] = (type_i, color)
                index += TypeSkips[type_i]

    return positions

PositionChannles = create_position_channels()

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

    boards = [ initial_board, sh.Board(skipped_sfen(sfen)), ] if initial_board.turn == 0 else \
        [ sh.Board(skipped_sfen(sfen)), initial_board, ] 
    sides = [0,1]

    # make moves and drops
    moves = [ [[] for i in range(81)], [[] for i in range(81)] ]
    drops = [ [[] for i in range(len(PT2VM))], [[] for i in range(len(PT2VM))] ]

    if False:
        print('====')
        print('len(PT2VM)',len(PT2VM))
        print('drops',drops)
        print('input legal moves 0: ', sorted([m.usi() for m in list(boards[0].legal_moves )]))
        print('input legal moves 1: ', sorted([m.usi() for m in list(boards[1].legal_moves )]))
        print('====')

    for side in sides:
        for move in boards[side].legal_moves:
            #print('append move:', move.drop_piece_type, move)
            if move.drop_piece_type is None:
                moves[side][move.from_square].append( move )
            else:
                if False:
                    print('drops[side][PT2VM[{}]={}].append( {} )'.format(
                        move.drop_piece_type,PT2VM[move.drop_piece_type],move))
                drops[side][PT2VM[move.drop_piece_type]].append( move )

    if False:
        for side in sides:
            for ds in drops[side]:
                print(side, ds)

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
                # position channel
                to = sqr2rc(p[0])
                vec[0][to[0]][to[1]][chB] = 1

            for row, col in product(range(9), range(9)):
                vec[0][row][col][chM] = -1
                if chP not in PositionChannles:
                    vec[0][row][col][chP] = -1

            for m in p[2]:
                # movement channels
                # print(m)
                if False:
                    if (ch_base[side] +  TypeHeads[p[1]]) == (ch_base[side] +  TypeHeads[PT2VM[sh.PAWN]]):
                        print('pawn:',p[1], m)
                    else:
                        print('other', m)
                to = sqr2rc(m.to_square)
                tmp_ch = chM if not m.promotion else chP
                vec[0][to[0]][to[1]][tmp_ch] = 1

            if pos_type is not None: p_cnt[side][pos_type] += 1

    if debug:
        vec = np.transpose(vec, [0,3,1,2])

    return vec

def to_debug(vec):
    return np.transpose(vec, [0,3,1,2])

def vector_to_board_and_moves(vec):
    # assert vec shape
    b = sh.Board()
    b.clear()

    moves = [[],[]]

    color = 0
    pos_found = False
    pos_sqr = None
    pos_type = None

    move_found = False

    for ch, row, col in product(range(360), range(9), range(9)):
        pre_color = color
        color = 0 if ch < 180 else 1

        if ch in PositionChannles:
            if row == 0 and col == 0:
                if move_found and not pos_found:
                    b.pieces_in_hand[pre_color][VM2PT[pos_type]] += 1

                pos_found = False
                pos_sqr = None
                pos_type, _ = PositionChannles[ch]

            if vec[0][row][col][ch] > 0:
                pos_found = True
                pos_sqr = row * 9 + col
                b.set_piece_at(pos_sqr, sh.Piece(VM2PT[pos_type], color))
        else:
            promote = False

            if ch - 1 in PositionChannles:
                if row == 0 and col == 0:
                    move_found = False
            else:
                promote = True

            if vec[0][row][col][ch] > 0:
                move_found = True
                to_sqr = row * 9 + col
                moves[color].append(
                    sh.Move(pos_sqr, to_sqr, promotion=promote,
                        drop_piece_type=(None if pos_sqr is not None else VM2PT[pos_type])).usi())
            elif vec[0][row][col][ch] < 0:
                move_found = True

    return b, moves

def vector_to_usi_movement(board_vec, move_vec):
    pass

def fliplr(vec):
    res = np.zeros(vec.shape)

    for r in range(9):
        for c in range(9):
            res[0][r][8-c] = vec[0][r][c]

    return res

def player_inverse(vec):
    res = np.zeros(vec.shape)

    for r in range(9):
        for c in range(9):
            res[0][r][c][  0:180] = vec[0][r][c][180:360]
            res[0][r][c][180:360] = vec[0][r][c][  0:180]

    res2 = np.zeros(vec.shape)

    for r in range(9):
        for c in range(9):
            res2[0][8-r][8-c] = res[0][r][c]

    return res2

from random import choice

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    board = sh.Board()
    #vec = sfen_to_vector(board.sfen(), debug=True)

    if False:
        vec = sfen_to_vector(board.sfen())
        vec = fliplr(vec)
        vec = to_debug(vec)
        print(vec)

    if False:
        for i in range(20):
            board.push(choice(list(board.legal_moves)))

        vec = sfen_to_vector(board.sfen())
        vec = player_inverse(vec)
        vec = to_debug(vec)
        print(board)
        print(vec)

    if True:
        import collections
        temp = collections.OrderedDict(sorted(PositionChannles.items()))
        for k, v in temp.items():
            print(k,v)

