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
        for i in range(5):
            b1 = shogi.Board()
            n = 0

            while not b1.is_checkmate():
                #print('turn: ', n)
                if n > 300:
                    break

                np.set_printoptions(threshold=np.inf)

                b1s = ns.skipped_board(b1)

                vec = ns.sfen_to_vector(b1.sfen())
                b2, moves = ns.vector_to_board_and_moves(vec)
                b1v = b1.sfen().split(' ')
                b2v = b2.sfen().split(' ')
                #print(ns.to_debug(vec))

                self.assertEqual(b1v[0], b2v[0], 'board -> vec -> board\n{}\n{}'.format(b1v[0], b2v[0]))
                self.assertEqual(b1v[2], b2v[2], 'board -> vec -> board\n{}\n{}'.format(b1v[2], b2v[2]))

                b1_moves = [
                        sorted([m.usi() for m in list(b1.legal_moves )]),
                        sorted([m.usi() for m in list(b1s.legal_moves)])
                    ]

                if b1.turn == 1:
                    b1_moves[0], b1_moves[1] = b1_moves[1], b1_moves[0]

                moves[0] = sorted(list(set(moves[0])))
                moves[1] = sorted(list(set(moves[1])))

                if b1_moves[0] != moves[0] or b1_moves[1] != moves[1]:
                    #print(ns.to_debug(vec))
                    print(b1.sfen())
                    print(b1s.sfen())
                    print(b2.sfen())

                if False:
                    print('-------')
                    print('a',b1_moves[0])
                    print('b',moves[0])
                    print('c',b1_moves[1])
                    print('d',moves[1])
                    print('--')
                    print('e',sorted([m.usi() for m in list(b2.legal_moves)]))
                    print('f',sorted([m.usi() for m in list(ns.skipped_board(b2).legal_moves)]))

                self.assertEqual(b1_moves[0], moves[0],
                    'check self moves: {}\n{}\n{}'.format(n, b1_moves[0], moves[0]))
                self.assertEqual(b1_moves[1], moves[1],
                    'check counter moves: {}\n{}\n{}'.format(n, b1_moves[1], moves[1]))
                b1.push( choice(list(b1.legal_moves)) )
                n += 1

    def test_extract_movement_vec(self):
        self_move_indices = sorted([key for key,val in ns.MovementChannels.items() if val[1] == 0])

        for i in range(5):
            b1 = shogi.Board()
            n = 0
            for n in range(300):
                if b1.is_checkmate(): break

                vec = ns.sfen_to_vector(b1.sfen())
                move_vec = ns.extract_movement_vec(vec)

                move_i = 0

                for key in self_move_indices:
                    self.assertTrue( np.array_equal(
                        np.maximum(vec[0,:,:,key],0)
                        ,move_vec[0,:,:,move_i]))

                    move_i += 1

if __name__ == '__main__':
    ut.main()

