from argparse import ArgumentParser as AP
from pathlib import Path
from random import choice
import time
#import importlib as im

import tensorflow as tf
import numpy as np
import shogi

import ginkgo.numpy_shogi

DEFAULT_READ_OP_THREADS_NUM = 4
DEFAULT_MODELPB = 'model.pb'
DEFAULT_LOG_DIRECTORY = './logs'
DEFAULT_DO_RESTORE = True

MINIBATCH_SIZE=256
NUM_GPU = 2

# ------------------------------------------------------------
# command line option parser
# ------------------------------------------------------------

def create_parser():
    parser = AP(prog='ginkgo self match program')

    # IO parameter
    parser.add_argument('--modelpb', type=str, default=DEFAULT_MODELPB)

    #parser.add_argument('--model-py', type=str, default=None)
    #parser.add_argument('--samples', type=str)

    # test apllication
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOG_DIRECTORY)
    #parser.add_argument('--log-gradients', type=bool, default=False)
    #parser.add_argument('--log-variables', type=bool, default=False)

    # computing parameter
    #parser.add_argument('--num-read-threads', type=int, default=DEFAULT_READ_OP_THREADS_NUM)
    #parser.add_argument('--num-gpus', type=int, default=1)
    #parser.add_argument('--var-device', type=str, default=DEFAULT_VAR_DEVICE)
    #parser.add_argument('--accum-device', type=str, default=DEFAULT_ACCUM_DEVICE)

    # training parameters
    #parser.add_argument('--minibatch-size', type=int, default=MINIBATCH_SIZE)

    return parser

# ------------------------------------------------------------
# sub commands
# ------------------------------------------------------------

def do_match(ns):
    config_proto = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=config_proto)

    with tf.gfile.FastGFile(ns.modelpb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    placeholders = []

    for i in range(NUM_GPU):
        placeholders.append(tf.get_default_graph().get_tensor_by_name('input_{}:0'.format(i)))

    inference_op = tf.get_default_graph().get_tensor_by_name('inference:0')


    # ===

    for i in range(1):
    #for i in range(900):
        b_move = i % 30
        w_move = i // 30

        board = shogi.Board()

        board.push(list(board.legal_moves)[b_move])
        print('----')
        print('turn 1: manual movement')
        print(board.kif_str())

        board.push(list(board.legal_moves)[w_move])
        print('----')
        print('turn 2: manual movement')
        print(board.kif_str())

        turn = 3

        while not board.is_checkmate():
            measure_start = time.time()
            # create placeholder
            legal_moves = list(board.legal_moves)
            side = board.turn
            board_tensors = ginkgo.numpy_shogi.create_next_move_tensors(
                    board, MINIBATCH_SIZE, NUM_GPU)

            minibatches = []

            print('----')
            print('turn {}:'.format(turn))
            turn += 1
            print('len(legal_moves) = ', len(legal_moves))

            for i in range( len(board_tensors) // MINIBATCH_SIZE ):
                minibatches.append( np.concatenate(
                    board_tensors[i * MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE], axis=0) )


            max_index = 0
            max_value = 0
            tensor_index = 0

            total_inferences = []

            for tensor_index in range(len(minibatches) // NUM_GPU):
                feed_dict = {}
                for i in range(NUM_GPU):
                    feed_dict[placeholders[i]] = minibatches[tensor_index + i]
                tensor_index += NUM_GPU
                inferences = sess.run([inference_op], feed_dict=feed_dict)
                total_inferences.append(inferences[0])

            total_inferences = np.concatenate(total_inferences, axis=0)

            #print("total_inferences.shape = ", total_inferences.shape)
            #print("total_inferences[:,0].shape = ", total_inferences[:,0].shape)
            am = np.argmax(total_inferences[0:len(legal_moves), 0], axis=0)
            #print("am.shape = ", am.shape, ", am = ", am)

            max_value = total_inferences[am][0]
            board.push(legal_moves[am])
            print('max index, value: {}, {}'.format(am, max_value))
            print(board.kif_str())
            measure_end = time.time()
            print('time spent: {:0.4f}'.format(measure_end - measure_start))

            del(legal_moves, board_tensors, minibatches, total_inferences, am)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = create_parser()
    namespace = parser.parse_args()
    do_match(namespace)

if __name__ == '__main__':
    main()

