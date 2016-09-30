import sys
import shogi

def convert_pos(s):
    tbl = 'abcdefghi'
    return s[0]+tbl[int(s[1])-1]

P_C2U = {
    'FU': shogi.PAWN,
    'KY': shogi.LANCE,
    'KE': shogi.KNIGHT,
    'GI': shogi.SILVER,
    'KI': shogi.GOLD,
    'KA': shogi.BISHOP,
    'HI': shogi.ROOK,
    'OU': shogi.KING,
    'TO': shogi.PROM_PAWN,
    'NY': shogi.PROM_LANCE,
    'NK': shogi.PROM_KNIGHT,
    'NG': shogi.PROM_SILVER,
    'UM': shogi.PROM_BISHOP,
    'RY': shogi.PROM_ROOK,
    }

P_U2C = dict((v,k) for k,v in P_C2U.items())
UCHI={'FU':'P*','KY':'L*','KE':'N*','GI':'S*','KI':'G*','KA':'B*','HI':'R*'}

def csa_move_to_usi(b, s):
    if s[1:3] == '00':
        return UCHI[s[5:7]] + convert_pos(s[3:5])

    from_sqr = (int(s[2]) - 1) * 9 + 9 - int(s[1])
    from_p = b.piece_at(from_sqr)

    if from_p is None:
        raise ValueError('no piece at from position')
    
    nari = '' if s[5:7] == P_U2C[from_p.piece_type] else '+'

    return convert_pos(s[1:3]) + convert_pos(s[3:5]) + nari

def is_csa_movement(s):
    if len(s) != 7:
        return False

    sign = "+-"
    num = "0123456789"
    al = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    return s[0] in sign \
        and s[1] in num and s[2] in num and s[3] in num and s[4] in num \
        and s[5] in al and s[6] in al

def convert(in_path, out_path):
    in_file = open(in_path)
    in_lines = in_file.readlines()

    winner = None
    pre_player = 0

    total_turn = 0

    for line in in_lines:
        line = line.rstrip()

        if is_csa_movement(line):
            total_turn = total_turn + 1

        if line[0] == '+':
            pre_player = 0
        elif line[0] == '-':
            pre_player = 1
        elif line == '%TORYO':
            winner = 0 if pre_player == 1 else 1
            break

    #print('winner: {}'.format(winner))

    in_file.seek(0)

    board = shogi.Board()
    out_file = open(out_path,mode='w')
    turn = 0

    for line in in_lines:
        line = line.rstrip()
        if is_csa_movement(line):
            usi = csa_move_to_usi(board, line)
            sfen = board.sfen()
            win = 'b' if winner == 0 else 'w'
            print('{},{},{},{}'.format( sfen, total_turn, usi, win), file=out_file)
            board.push_usi(usi)

