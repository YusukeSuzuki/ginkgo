from argparse import ArgumentParser as AP
from pathlib import Path
from random import shuffle
import time
import importlib as im

import tensorflow as tf
import numpy as np
import shogi

import ginkgo.yaml_loader as yl
import ginkgo.numpy_shogi as numpy_shogi
import ginkgo.shogi_yaml
import ginkgo.shogi_records as shogi_records
import ginkgo.shogi_loader as shogi_loader
import ginkgo.shogi_tfrecords as shogi_tfrecords
import ginkgo.train

ROOT_VARIABLE_SCOPE='prophet'

MINIBATCH_SIZE = 32

DEFAULT_READ_OP_THREADS_NUM = 4
DEFAULT_MODEL_DIRECTORY = './model_training'
DEFAULT_MODEL_NAME = 'model'
DEFAULT_LOG_DIRECTORY = './logs'
DEFAULT_DO_RESTORE = True

DEFAULT_VAR_DEVICE = '/cpu:0'
DEFAULT_ACCUM_DEVICE = '/cpu:0'

DEFAULT_MODEL_MODULE = 'ginkgo.models.prophet_model_20170419_0'

# ------------------------------------------------------------
# command line option parser
# ------------------------------------------------------------

def create_parser():
    parser = AP(prog='train_prophet_with_records')

    # IO parameter
    parser.add_argument('--model-directory', type=str, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument('--model-py', type=str, default=None)
    parser.add_argument('--samples', type=str)

    # test apllication
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOG_DIRECTORY)
    #parser.add_argument('--log-gradients', type=bool, default=False)
    parser.add_argument('--log-variables', type=bool, default=False)

    # computing parameter
    parser.add_argument('--num-read-threads', type=int, default=DEFAULT_READ_OP_THREADS_NUM)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--var-device', type=str, default=DEFAULT_VAR_DEVICE)
    parser.add_argument('--accum-device', type=str, default=DEFAULT_ACCUM_DEVICE)

    # training parameters
    parser.add_argument('--minibatch-size', type=int, default=MINIBATCH_SIZE)

    return parser

# ------------------------------------------------------------
# sub commands
# ------------------------------------------------------------

def do_test(ns):
    models_dir = Path(ns.model_directory)
    model_path = models_dir/ns.model_name

    # build read data threads
    path_list = [str(p) for p in Path(ns.samples).glob('**/*.tfrecords')]
    print('len(path_list) = ', len(path_list))
    shuffle(path_list)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        sess = tf.Session( config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        coordinator = tf.train.Coordinator()

        # build read data threads
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(path_list, num_epochs=1)

            input_batches = [tf.train.shuffle_batch(
                shogi_tfrecords.train_read_op(filename_queue),
                ns.minibatch_size, 4000, 1000,
                num_threads=ns.num_read_threads) for _ in range(ns.num_gpus)]

        if ns.model_py:
            print('load external model file: {}'.format(ns.model_py))
            model_module = im.machinery.SourceFileLoader('externalmodel', ns.model_py).load_module()
        else:
            model_module = im.import_module(DEFAULT_MODEL_MODULE)

        with tf.device('/gpu:0'):
            with tf.name_scope('tower_0') as scope:
                inference = model_module.inference(
                    input_batches[0][0], reuse=False, var_device=ns.var_device)
                correct_rate_op = ginkgo.train.correct_rate(input_batches[0][1], inference)

        print(correct_rate_op)

        # create saver and logger
        saver = tf.train.Saver()

        # ready to run
        print('initialize')
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        latest_checkpoint = tf.train.latest_checkpoint(str(ns.model_directory))

        if latest_checkpoint:
            print('restore {}'.format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
        else:
            raise ValueError('no model at {}'.format(str(ns.model_directory)))

        # run
        print('test')

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        rate_total = 0.
        rate_avg = 0.
        rate_count = 0
        rate_min = 2.
        rate_max = -1

        display_interval = 100
        format_string = \
            'correct rate({:0.3f}): avg {:0.3f}, min {:0.3f}, max {:0.3f},' + \
            ' time {:0.3f} (sec per minibatch({} samples))'


        try:
            measure_start = time.time()

            while not coordinator.should_stop():
                rate = sess.run( (correct_rate_op), feed_dict={} )
                rate_count += 1
                rate_min = min(rate_min, rate)
                rate_max = max(rate_max, rate)
                rate_total += rate
                rate_avg = rate_total / rate_count

                if rate_count % display_interval == 0:
                    measure_end = time.time()
                    print(format_string.format(
                        rate_count, rate_avg, rate_min, rate_max,
                        (measure_end - measure_start) / display_interval, ns.minibatch_size))
        except tf.errors.OutOfRangeError as e:
            print('sample exausted')

        print('correct rate: {}'.format(rate_avg))

        # finalize
        coordinator.request_stop()
        coordinator.join(threads)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = create_parser()
    namespace = parser.parse_args()
    do_test(namespace)

if __name__ == '__main__':
    main()

