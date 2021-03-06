import ginkgo.numpy_shogi as numpy_shogi
import ginkgo.shogi_records as sr

import tensorflow as tf
import numpy as np

import threading
from threading import Thread
import queue
from queue import Queue
import math
from concurrent.futures import ProcessPoolExecutor as Executor
import gc

def record_to_vec(r):
    sfen, side, turn, total, move, winner = r

    board_vec = numpy_shogi.sfen_to_vector(sfen, usi=move)

    if side == 'w':
        board_vec = numpy_shogi.player_inverse(board_vec)
    
    match_vec = np.array([
        1.0 if side == winner else 0.0,
        1.0 if side != winner else 0.0])

    weight_vec = np.array([math.sqrt(float(turn)/float(total))])

    return (np.squeeze(board_vec, axis=0), match_vec, weight_vec)

def map_func(r):
    sfen, side, turn, total, move, winner = r = sr.to_data(r)
    if int(turn) < 30: return None

    #if side != 'b': return None

    board_vec, label_vec, weight_vec = record_to_vec(r)

    return (
        np.expand_dims(board_vec,0),
        np.expand_dims(label_vec,0),
        np.expand_dims(weight_vec,0))

def flipdata(r):
    return (numpy_shogi.fliplr(r[0]), r[1], r[2])

def load_loop(coord, sess, enqueue_op, close_op,  path_q, pool, loop,
        input_vector_ph, label_ph, turn_weight_ph):

    while not coord.should_stop():
        try:
            path = path_q.get(timeout=10)

            records = sr.load_file(path)

            #sfen, side, turn, total, move, winner = r = sr.to_data(r)
            data_list = list(pool.map(map_func, records))
            data_list = list(filter(lambda x: x is not None, data_list))

            #data_list2 = list(pool.map(flipdata, data_list))
            #data_list.extend(data_list2)

            vecs = [list(t) for t in zip(*data_list)]
            vecs = list(map(np.concatenate, vecs))

            if len(vecs) != 3:
                print('some error occured in reading file. skip this file: {}'.format(path))
                continue

            sess.run(enqueue_op, feed_dict={
                input_vector_ph: vecs[0],
                label_ph: vecs[1],
                turn_weight_ph: vecs[2]})

            if loop:
                path_q.put(path)

            del data_list, vecs, records,
            gc.collect()

        except queue.Empty  as e:
            try:
                sess.run(close_op)
            except tf.errors.CancelledError:
                pass
            break
        except tf.errors.AbortedError as e:
            break
        except tf.errors.CancelledError as e:
            break

def load_sfenx_threads_and_queue(
        coord, sess, path_list, batch_size, loop=False, threads_num=1, queue_max=50000, queue_min=8000):

    input_vector_ph = tf.placeholder(tf.float32, [None,9,9,360])
    label_ph = tf.placeholder(tf.float32, [None,2])
    turn_weight_ph = tf.placeholder(tf.float32, [None,1])

    q = tf.RandomShuffleQueue(queue_max, queue_min,
        [tf.float32, tf.float32, tf.float32], [[9,9,360], [2], [1]])
    enqueue_op = q.enqueue_many(
        [input_vector_ph, label_ph, turn_weight_ph])
    close_op = q.close()
    path_q = Queue()

    for p in path_list:
        path_q.put(p)

    pool = Executor(max_workers=threads_num+2)

    threads = [Thread(target=load_loop,
        args=(coord, sess, enqueue_op, close_op, path_q, pool, loop,
            input_vector_ph, label_ph, turn_weight_ph))
        for i in range(threads_num)]

    tf.summary.scalar('shogi_loader/size', q.size())

#    input_batch, label_batch, turn_weight_batch = q.dequeue_many(batch_size)
#    return threads, input_batch, label_batch, turn_weight_batch
    return threads, q

