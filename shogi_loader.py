import numpy_shogi
import shogi_records as sr

import tensorflow as tf
import numpy as np

from threading import Thread
from queue import Queue
import math

def record_to_vec(r):
    sfen, side, turn, total, move, winner = r

    board_vec = numpy_shogi.sfen_to_vector(sfen, usi=move)
    
    match_vec = np.array([
        1.0 if side == winner else 0.0,
        1.0 if side != winner else 0.0])

    weight_vec = np.array([math.sqrt(float(turn)/float(total))])

    return (np.squeeze(board_vec, axis=0), match_vec, weight_vec)


def load_loop(coord, sess, enqueue_op, path_q,
        input_vector_ph, label_ph, turn_weight_ph):

    while not coord.should_stop():
        try:
            path = path_q.get()

            records = sr.load_file(path)

            for r in records:
                sfen, side, turn, total, move, winner =  r = sr.to_data(r)
                if side != 'b': continue
                if int(turn) < 20: continue

                board_vec, match_vec, weight = record_to_vec(r)

                sess.run(enqueue_op, feed_dict={
                    input_vector_ph: board_vec,
                    label_ph: match_vec,
                    turn_weight_ph: weight
                    })

            path_q.put(path)
        except tf.errors.AbortedError as e:
            print(e)
            break
        except tf.errors.CancelledError as e:
            print(e)
            break

def load_sfenx_threads_and_queue(coord, sess, path_list, batch_size,
        input_vector_ph, label_ph, turn_weight_ph,
        threads_num=1):

    q = tf.RandomShuffleQueue(50000, 8000,
        [tf.float32, tf.float32, tf.float32], [[9,9,148], [2], [1]])
    enqueue_op = q.enqueue([input_vector_ph, label_ph, turn_weight_ph])
    path_q = Queue()

    for p in path_list:
        path_q.put(p)

    threads = [Thread(
        target=load_loop,
        args=(coord, sess, enqueue_op, path_q,
            input_vector_ph, label_ph, turn_weight_ph))
        for i in range(threads_num)]

    input_batch, label_batch, turn_weight_batch = \
        q.dequeue_many(batch_size)

    tf.scalar_summary('shogi_loader/size', q.size())

    return threads, input_batch, label_batch, turn_weight_batch

