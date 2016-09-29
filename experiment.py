import shogi
from random import choice
import numpy as np
import shogiutils

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

