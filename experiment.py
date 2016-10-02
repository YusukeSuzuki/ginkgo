from random import choice
from pathlib import Path
import shogi
import numpy as np
import numpy_shogi 

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

if __name__ == '__main__':
    #print_board()

    records = load_files()
    print(len(records))

    pass

