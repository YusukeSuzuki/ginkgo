from argparse import ArgumentParser as AP
from pathlib import Path
import fnmatch
from random import choice, randrange, shuffle
import math
import sys

import tensorflow as tf
import numpy as np
import shogi

import yaml_loader as yl
import numpy_shogi
import shogi_yaml
import shogi_records
import shogi_loader


ROOT_VARIABLE_SCOPE='prophet'
MODEL_YAML_PATH = 'prophet_model.yaml'
MODELS_DIR = 'models'

MINI_BATCH_SIZE = 100

# ------------------------------------------------------------
# sub commands
# ------------------------------------------------------------

def do_train(namespace):
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir/namespace.modelfile
    model_backup_path = models_dir/(namespace.modelfile+'.back')

    sess = tf.Session()
    coordinator = tf.train.Coordinator()

    # build read data threads
    path_list = list(Path('data/sfen').glob('*.csa'))

    with tf.variable_scope('input'), tf.device('/cpu:0'):
        load_threads, input_batch, label_batch, weight_batch = \
            shogi_loader.load_sfenx_threads_and_queue(
                coordinator, sess, path_list, MINI_BATCH_SIZE,
                threads_num=8)

    # build model
    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(MODEL_YAML_PATH)
        tags = graph_root.build(feed_dict={
            'root': input_batch, 'label': label_batch, 'turn_weight': weight_batch})

    # get optimizer for train
    train = tf.get_default_graph().get_operation_by_name(
            namespace.optimizer)

    # create saver and logger
    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()

    # ready to run

    print('initialize')
    writer = tf.train.SummaryWriter(namespace.logdir, sess.graph)
    sess.run(tf.initialize_all_variables())

    # run

    if namespace.restore:
        print('restore {}'.format(namespace.restore))
        saver.restore(sess, namespace.restore)

    writer.add_graph(tf.get_default_graph())

    print('convert records')

    print('train')

    for t in load_threads: t.start()

    for i in range(0, 40000):
        if i % 5 == 0:
            summary, res = sess.run( (merged, train), feed_dict={} )
            writer.add_summary(summary, i)
        else:
            res = sess.run( (train), feed_dict={} )

        print('loop: {}'.format(i))

        if i > 10 and i % 4000 == 1:
            print('save backup to: {}'.format(model_backup_path))
            saver.save(sess, str(model_backup_path))

    print('save to: {}'.format(model_path))
    saver.save(sess, str(model_path))

    # finalize
    coordinator.request_stop()
    coordinator.join(load_threads)
    writer.close()

def do_test(namespace):
    print('unavailable now')

def do_eval(namespace):
    print('unavailable now')

def do_dump_network(namespace):
    # build
    with tf.variable_scope('input'), tf.device('/cpu:0'):
        input_vector = tf.placeholder(tf.float32, [1,9,9,148])

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(MODEL_YAML_PATH)
        tags = graph_root.build(feed_dict={'root': input_vector})

    print('-- variables')
    for variable in tf.all_variables():
        if fnmatch.fnmatch(variable.name, namespace.pattern):
            print(variable.name)

    print('-- operations')
    for operation in tf.get_default_graph().get_operations():
        if fnmatch.fnmatch(operation.name, namespace.pattern):
            print(operation.name)

def do_dump_graph_log(namespace):
    # build
    #print('exclude tags: {}'.format(namespace.exclude_tags.split(',')))

    with tf.variable_scope('input'), tf.device('/cpu:0'):
        input_vector = tf.placeholder(tf.float32, [1,9,9,148])

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(MODEL_YAML_PATH)
        tags = graph_root.build(feed_dict={'root': input_vector})

    sess = tf.Session()
    writer = tf.train.SummaryWriter(namespace.logdir, sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)
    writer.add_graph(tf.get_default_graph())
    writer.close()

# ------------------------------------------------------------
# command line option parser
# ------------------------------------------------------------

def create_parser():
    parser = AP(prog='train_prophet_with_records')
    parser.set_defaults(func=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--modelfile', type=str, default='model.ckpt')
    parser.add_argument('--restore', type=str, default='')
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser('train')
    sub_parser.set_defaults(func=do_train)
    sub_parser.add_argument('--optimizer', type=str, required=True)
    sub_parser.add_argument('--samples', type=str, default='./samples')
    sub_parser.add_argument('--trainable-scope', type=str, default='')

    sub_parser = sub_parsers.add_parser('test')
    sub_parser.set_defaults(func=do_test)

    sub_parser = sub_parsers.add_parser('eval')
    sub_parser.set_defaults(func=do_eval)

    sub_parser = sub_parsers.add_parser('dump-network')
    sub_parser.set_defaults(func=do_dump_network)
    sub_parser.add_argument('--pattern', type=str, default='*')

    sub_parser = sub_parsers.add_parser('dump-graph-log')
    sub_parser.set_defaults(func=do_dump_graph_log)
    sub_parser.add_argument('--exclude-tags', type=str, default='')

    return parser

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def run():
    parser = create_parser()
    namespace = parser.parse_args()

    if namespace.func:
        namespace.func(namespace)
    else:
        parser.print_help()

if __name__ == '__main__':
    run()

