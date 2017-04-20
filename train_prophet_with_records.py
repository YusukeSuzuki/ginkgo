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
import ginkgo.shogi_tfrecords as shogi_tfrecords

ROOT_VARIABLE_SCOPE='prophet'
MODEL_YAML_PATH = 'prophet_model.yaml'
MODELS_DIR = 'models'

MINIBATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-5

# ------------------------------------------------------------
# command line option parser
# ------------------------------------------------------------

def create_parser():
    parser = AP(prog='train_prophet_with_records')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--growth-memory', type=bool, default=False)
    parser.add_argument('--input-model', type=str, default='')
    parser.add_argument('--output-model', type=str)
    parser.add_argument('--prophet-yaml', type=str)
    parser.add_argument('--modeldir', type=str)
    parser.add_argument('--samples', type=str)
    parser.add_argument('--minibatch-size', type=int, default=MINIBATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--log-gradients', type=bool, default=False)
    parser.add_argument('--log-variables', type=bool, default=False)
    parser.add_argument('--num-gpus', type=int, default=1)

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

            grad = tf.concat(0,grads)
            grad = tf.reduce_mean(grad,0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

def do_train(ns):
    models_dir = Path(ns.modeldir)
    model_path = models_dir/ns.output_model

    # build read data threads
    path_list = list(Path(ns.samples).glob('*.tfrecords'))
    print('len(path_list) = ', len(path_list))
    shuffle(path_list)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        sess = tf.Session( config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        opt = tf.train.AdamOptimizer(ns.learning_rate, beta1=ns.adam_beta1, beta2=adam_beta2)

        # build read data threads
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(path_list)

            input_batches = tf.train.shuffle_batch(
                [shogi_tfrecords.train_read_op(filename_queue) for i in range(ns.num_gpus)],
                ns.minibatch_size, 40000, 10000, num_threads=8))

        tower_grads = []

        for i in range(ns.num_gpus):
            with tf.device('/gpu:{}'.format(i)), tf.name_scope('tower_{}'.format(i)) as scope:
                graph_root = yl.load(ns.prophet_yaml)
                tags = graph_root.build(feed_dict={
                    'root': input_batches[i][0], 'label': input_batches[i][1]})

                loss = tags['p_loss']
                tf.get_variable_scope().reuse_variables()

                grads = opt.compute_gradients(loss)
                tower_grads.append( grads )

        grads = average_gradients(tower_grads, ns.log_gradients)

        train_op = opt.apply_gradients(grads, global_step=global_step)

        if ns.log_variables:
            for var in tf.trainable_variables():
                tf.summary.histogram(
                    var.op.name, var, collections=['histogram', 'variables'])

        merged_summary_op = tf.summary.merge_all()
        histogram_summary_op = tf.summary.merge(tf.get_collections('histogram'))
        scalar_summary_op = tf.summary.merge(tf.get_collections('scalar'))

        # ready to run
        saver = tf.train.Saver()

        print('initialize')
        writer = tf.summary.FileWriter(ns.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        latest_checkpoint = tf.train.latest_checkpoint(str(ns.input_model))

        if latest_checkpoint:
            print('restore {}'.format(ns.input_model))
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

                if gs > 10 and gs % 5000 == 1:
                    print('save backup to: {}'.format(model_backup_path))
                    saver.save(sess, str(model_path), global_step=gs)

                if i < 100 and gs % 2 == 0:
                    writer.flush()
                elif i < 2000 and gs % 5 == 0:
                    writer.flush()
                elif i < 5000 and gs % 10 == 0:
                    writer.flush()

                i += 1
        except tf.errors.OutOfRangeError as e:
            print('sample exausted')

        print('save to: {}'.format(model_path))
        saver.save(sess, str(model_path), global_step=gs)

        # finalize
        coordinator.request_stop()
        coordinator.join(load_threads)
        writer.close()

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def run():
    parser = create_parser()
    namespace = parser.parse_args()
    do_train(namespace)

if __name__ == '__main__':
    run()

