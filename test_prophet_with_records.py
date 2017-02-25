from argparse import ArgumentParser as AP
from pathlib import Path
import fnmatch
from random import choice, randrange, shuffle
import math
import sys

import tensorflow as tf
import numpy as np
import shogi

import ginkgo.yaml_loader as yl
import ginkgo.numpy_shogi as numpy_shogi
import ginkgo.shogi_yaml
import ginkgo.shogi_records as shogi_records
import ginkgo.shogi_loader as shogi_loader


ROOT_VARIABLE_SCOPE='prophet'
MODEL_YAML_PATH = 'prophet_model.yaml'
MODELS_DIR = 'models'

MINI_BATCH_SIZE = 32

# ------------------------------------------------------------
# sub commands
# ------------------------------------------------------------

def do_test(ns):
    sess = tf.Session()
    coordinator = tf.train.Coordinator()

    # build read data threads
    path_list = list(Path(ns.samples).glob('*.csa'))
    print('len(path_list) = ', len(path_list))
    shuffle(path_list)

    with tf.variable_scope('input'), tf.device('/cpu:0'):
        load_threads, input_queue = \
            shogi_loader.load_sfenx_threads_and_queue(
                coordinator, sess, path_list, ns.minibatch_size,
                threads_num=6, queue_max=50000, queue_min=16000)

    # build model
    
    input_batch, label_batch, weight_batch = input_queue.dequeue_many(ns.minibatch_size)

    with tf.device('/gpu:0'):
        with tf.name_scope('tower_{}'.format(i)) as scope:
            graph_root = yl.load(ns.prophet_yaml)
            tags = graph_root.build(feed_dict={
                'root': input_batch, 'label': label_batch, 'turn_weight': weight_batch})
            loss = tags['p_loss']
            correct_rate = tags['']

    # create saver and logger
    saver = tf.train.Saver()

    # ready to run
    print('initialize')
    sess.run(tf.initialize_all_variables())

    ckpt = tf.train.get_checkpoint_state('models')
    print(ckpt)
    print(ckpt.model_checkpoint_path)

    if ns.input_model:
        print('restore {}'.format(ns.input_model))
        input_model = ns.input_model
        saver.restore(sess, input_model)

    # run
    print('test')

    for t in load_threads: t.start()

    rate_total = 0.
    rate_avg = 0.
    rate_count = 0
    rate_min = 2.
    rate_max = -1
    correct_rate_op = tags['rate']

    try:
        while not coordinator.should_stop():
            rate = sess.run( (correct_rate_op), feed_dict={} )
            rate_count += 1
            rate_min = min(rate_min, rate)
            rate_max = max(rate_max, rate)
            rate_total += rate
            rate_avg = rate_total / rate_count

            if rate_count % 100 == 0:
                print('correct rate({}): avg {}, min {}, max {}'.format(
                    rate_count, rate_avg, rate_min, rate_max))

    except tf.errors.OutOfRangeError as e:
        print('sample exausted')

    print('correct rate: {}'.format(rate_total))

    # finalize
    coordinator.request_stop()
    coordinator.join(load_threads)

# ------------------------------------------------------------
# command line option parser
# ------------------------------------------------------------

def create_parser():
    parser = AP(prog='train_prophet_with_records')
    parser.set_defaults(func=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--growth-memory', type=bool, default=False)
    parser.add_argument('--input-model', type=str)
    parser.add_argument('--prophet-yaml', type=str)
    parser.add_argument('--modeldir', type=str)
    parser.add_argument('--samples', type=str)
    parser.add_argument('--minibatch-size', type=int, default=MINI_BATCH_SIZE)

    return parser

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def run():
    parser = create_parser()
    namespace = parser.parse_args()
    do_test(namespace)

if __name__ == '__main__':
    run()

