from random import choice
from pathlib import Path
import shogi
import numpy as np
import numpy_shogi
import shogi_records

def print_board():
    b = shogi.Board()
    print(b)
    print('--------------------')

    i = 0
    while not b.is_checkmate():
        b.push(choice(list(b.legal_moves)))
        i = i + 1

        if i % 100 == 0:
            #print(list(b.legal_moves))
            #print(b.pieces_in_hand)
            pass

        if i == 150:
            break

    np.set_printoptions(threshold=np.inf)

    print(b)
    print('--------------------')
    print(list(b.legal_moves))
    print('--------------------')
    print(shogiutils.board_to_vector(b))

def load_files():
    d = Path('data/sfen')

    records = []

    for f in d.glob('*.csa'):
        lines = f.open().readlines()
        lines = list(map(lambda x: x.rstrip(), lines))
        lines = filter(lambda x: len(x) > 0, lines)
        records.extend(lines)

    return records

def read_record1():
    d = Path('data/sfen')

    line = ''
    i = 0
    for p in d.glob('*.csa'):
        with p.open() as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.rstrip(), lines))
            lines = list(filter(lambda x: len(x) > 0, lines))
            line = lines[0]

            sfen, side, turn, total, move, winner = shogi_records.to_data(line)
            vec = numpy_shogi.sfen_to_vector(sfen, debug=True)

            i += 1
            if i == 10000:
                return

def read_record2():
    d = Path('data/sfen')

    line = ''
    for p in d.glob('*111.csa'):
        with p.open() as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.rstrip(), lines))
            lines = list(filter(lambda x: len(x) > 0, lines))
            line = lines[0]

            sfen, side, turn, total, move, winner = shogi_records.to_data(line)
            vec = numpy_shogi.board_to_vector(shogi.Board(sfen), debug=True)

if __name__ == '__main__':
    #print_board()
    #records = load_files()
    #print(len(records))
    read_record1()

    pass

