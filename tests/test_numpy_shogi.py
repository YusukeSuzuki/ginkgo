import unittest as ut
from random import choice
import shogi
import ginkgo.numpy_shogi as ns
import numpy as np

class NumpyShogiTest(ut.TestCase):
    CLS_VAL='none'

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_inverse_conversion(self):
        b1 = shogi.Board()
        vec = ns.sfen_to_vector(b1.sfen())
        b2, moves = ns.vector_to_board_and_moves(vec)
        self.assertEqual(b1.sfen(), b2.sfen(), 'board -> vec -> board')

    def test_inverse_conversion2(self):
        for i in range(50):
            b1 = shogi.Board()
            n = 0
            while not b1.is_checkmate():
                print('test', n)
                n += 1
                b1.push( choice(list(b1.legal_moves)) )
                vec = ns.sfen_to_vector(b1.sfen())
                b2, moves = ns.vector_to_board_and_moves(vec)
                b1v = b1.sfen().split(' ')
                b2v = b2.sfen().split(' ')

                if b1.pieces_in_hand[0] or b1.pieces_in_hand[1]:
                    print(b1.sfen())
                    np.set_printoptions(threshold=np.inf)
                    print(ns.to_debug(vec))
                self.assertEqual(b1v[0], b2v[0], 'board -> vec -> board')
                self.assertEqual(b1v[2], b2v[2], 'board -> vec -> board')

if __name__ == '__main__':
    ut.main()

