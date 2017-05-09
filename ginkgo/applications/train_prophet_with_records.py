from argparse import ArgumentParser as AP
from pathlib import Path
from random import shuffle
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
MODEL_YAML_PATH = 'prophet_model.yaml'
MODELS_DIR = 'models'

MINIBATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-5

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

    # training apllication
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
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)

    # advanced
    parser.add_argument('--growth-memory', type=bool, default=False)

    return parser

# ------------------------------------------------------------
# train functions
# ------------------------------------------------------------

def average_gradients(tower_grads):
    average_grads = []

    with tf.name_scope('average_gradients'):
        for grad_and_vars in zip(*tower_grads):
            grads = []

            for g, u in grad_and_vars:
                expanded_g = tf.expand_dims(g,0)
                grads.append(expanded_g)
                tf.summary.histogram(u.name, g, collections=['histogram', 'grads'])

            grad = tf.concat(axis=0,values=grads)
            grad = tf.reduce_mean(grad,0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

def do_train(ns):
    models_dir = Path(ns.model_directory)
    model_path = models_dir/ns.model_name

    # build read data threads
    path_list = [str(p) for p in Path(ns.samples).glob('**/*.tfrecords')]
    print('len(path_list) = ', len(path_list))
    shuffle(path_list)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        sess = tf.Session( config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        with tf.variable_scope('train'):
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            opt = tf.train.AdamOptimizer(ns.learning_rate, beta1=ns.adam_beta1, beta2=ns.adam_beta2)

        # build read data threads
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(path_list, num_epochs=1)

            input_batches = [tf.train.shuffle_batch(
                shogi_tfrecords.train_read_op(filename_queue),
                ns.minibatch_size, 4000, 1000,
                num_threads=ns.num_read_threads) for _ in range(ns.num_gpus)]

            for i in input_batches:
                print(i)

        tower_grads = []
        reuse = False

        if ns.model_py:
            print('load external model file: {}'.format(ns.model_py))
            model_module = im.machinery.SourceFileLoader('externalmodel', ns.model_py).load_module()
        else:
            model_module = im.import_module(DEFAULT_MODEL_MODULE)

        for i in range(ns.num_gpus):
            with tf.device('/gpu:{}'.format(i)), tf.name_scope('tower_{}'.format(i)) as scope:
                print('build model for /gpu:{}'.format(i))

                inference = model_module.inference(
                    input_batches[i][0], reuse=reuse, var_device=ns.var_device)
                loss = ginkgo.train.loss(input_batches[i][1], inference)
                correct_rate = ginkgo.train.correct_rate(input_batches[i][1], inference)

                grads = opt.compute_gradients(loss)
                tower_grads.append( grads )

                reuse = True

        with tf.device(ns.accum_device):
            grads = average_gradients(tower_grads)
            train_op = opt.apply_gradients(grads, global_step=global_step)

        if ns.log_variables:
            for var in tf.trainable_variables():
                tf.summary.histogram(
                    var.op.name, var, collections=['histogram', 'variables'])

        merged_summary_op = tf.summary.merge_all()
        histogram_summary_op = tf.summary.merge(tf.get_collection('histogram'))
        scalar_summary_op = tf.summary.merge(tf.get_collection('scalar'))

        # ready to run
        saver = tf.train.Saver()

        print('initialize')
        writer = tf.summary.FileWriter(ns.logdir, sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        latest_checkpoint = tf.train.latest_checkpoint(str(ns.model_directory))

        if latest_checkpoint:
            print('restore {}'.format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        writer.add_graph(tf.get_default_graph())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # training
        print('train')

        gs = 0

        try:
            while not coordinator.should_stop():
                gs = int( sess.run( (global_step) ) )
                print('loop: {}'.format(gs))

                detailed_log_ops = [merged_summary_op, scalar_summary_op, histogram_summary_op]
                simple_log_ops = [merged_summary_op, scalar_summary_op]
                log_ops = simple_log_ops if gs % 500 != 0 else detailed_log_ops

                _, logs = sess.run([train_op, log_ops])

                for log in logs: writer.add_summary(log, gs)

                if gs > 10 and gs % 10000 == 1:
                    print('save backup to: {}'.format(model_path))
                    saver.save(sess, str(model_path), global_step=gs)

                if gs % 5 == 0:
                    writer.flush()

                i += 1
        except tf.errors.OutOfRangeError as e:
            print('sample exausted')

        print('save to: {}'.format(model_path))
        saver.save(sess, str(model_path))

        # finalize
        coordinator.request_stop()
        writer.close()

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = create_parser()
    namespace = parser.parse_args()
    do_train(namespace)

if __name__ == '__main__':
    main()

