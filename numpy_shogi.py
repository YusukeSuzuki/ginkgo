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

PROMS = {
    PT2VM[sh.ROOK]: PT2VM[sh.PROM_ROOK],
    PT2VM[sh.BISHOP]: PT2VM[sh.PROM_BISHOP],
    PT2VM[sh.SILVER]: PT2VM[sh.PROM_SILVER],
    PT2VM[sh.KNIGHT]: PT2VM[sh.PROM_KNIGHT],
    PT2VM[sh.LANCE]: PT2VM[sh.PROM_LANCE],
    PT2VM[sh.PAWN]: PT2VM[sh.PROM_PAWN],
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

def skipped_sfen(sfen):
    splitted = sfen.split(' ')
    splitted[1] = 'w' if splitted[1] == 'b' else 'b'
    return ' '.join(splitted)

def sfen_to_vector(sfen, usi=None, debug=False):
    initial_board = sh.Board(sfen)

    if usi is not None:
        initial_board.push_usi(usi)

    boards = [ initial_board, sh.Board(skipped_sfen(sfen)), ]
    sides = {0:1, 1:-1}

    # make moves and drops
    moves = [ [[] for i in range(81)], [[] for i in range(81)] ]
    drops = [ [[]] * len(PT2VM), [[]] * len(PT2VM) ]

    for side, val in sides.items():
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

    for side, val in sides.items():
        for key, p in boards[0].pieces_in_hand[side].most_common():
            for i in range(p):
                pieces[side].append( (None, PT2VM[key], drops[side][PT2VM[key]]) )

    vec = np.zeros([1,9,9,148]) if not debug else np.zeros([148,9,9])
    p_cnt = [0] * len(PT2VM)

    for side, val in sides.items():
        for p in pieces[side]:
            pos_type = p[1]
            pro_type = PROMS.get(p[1])

            ch = TypeHeads[p[1]] + p_cnt[p[1]] * 2

            if p[0] is not None:
                to = sqr2rc(p[0])
                if not debug:
                    vec[0][to[0]][to[1]][ch] = val
                else:
                    vec[ch][to[0]][to[1]] = val
                pos_type = p[1]

            ch = ch + 1
            if pro_type is not None:
                #print('pro_type = {}'.format(pro_type))
                ch_pro = TypeHeads[pro_type] + p_cnt[pro_type] * 2

            for m in p[2]:
                to = sqr2rc(m.to_square)
                tmp_ch = ch if not m.promotion else ch_pro
                if not debug:
                    vec[0][to[0]][to[1]][tmp_ch] = val
                else:
                    vec[tmp_ch][to[0]][to[1]] = val

            if pos_type is not None: p_cnt[pos_type] += 1
            if pro_type is not None: p_cnt[pro_type] += 1

    return vec

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
    vec = sfen_to_vector(board.sfen())
    vec = fliplr(vec)
    vec = np.transpose(vec, [0,3,1,2])
    print(vec)

