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

ROOT_VARIABLE_SCOPE='prophet'
MODEL_YAML_PATH = 'prophet_model.yaml'
MODELS_DIR = 'models'

MINI_BATCH_SIZE = 20

# ------------------------------------------------------------
# sub commands
# ------------------------------------------------------------

def do_train(namespace):
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir/namespace.modelfile
    model_backup_path = models_dir/(namespace.modelfile+'.back')

    # read data
    records = shogi_records.load_files('data/sfen')

    # build
    with tf.variable_scope('input'), tf.device('/cpu:0'):
        input_vector = tf.placeholder(tf.float32, [MINI_BATCH_SIZE,9,9,148])

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(MODEL_YAML_PATH)
        tags = graph_root.build(feed_dict={'root': input_vector})

    # get optimizer for train
    train = tf.get_default_graph().get_operation_by_name(namespace.optimizer)
    label = tf.get_default_graph().get_operation_by_name(
            ROOT_VARIABLE_SCOPE+'/train/label')

    # create saver and logger
    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()

    # rate log
#    with tf.variable_scope('log'), tf.device('/cpu:0'):
#        correct_rate = tf.placeholder(tf.float32, [])
#        correct_summary = tf.scalar_summary(
#            'correct_answer_rate', correct_rate)

    # ready to run

    print('initialize')
    sess = tf.Session()
    writer = tf.train.SummaryWriter(namespace.logdir, sess.graph)
    sess.run(tf.initialize_all_variables())
    #coordinator = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # run

    if namespace.restore:
        print('restore {}'.format(namespace.restore))
        saver.restore(sess, namespace.restore)

    writer.add_graph(tf.get_default_graph())

    print('convert records')

    #print(tags)
    #print(train)
    np.set_printoptions(threshold=np.inf)

    score = []

    shuffle(records)
    new_records = []

    for r in records:
        sfen, side, turn, total, move, winner = shogi_records.to_data(r)
        if side != 'b':
            continue
        if int(turn) < 20:
            continue
        new_records.append( (sfen, side, turn, total, move, winner) )

    records = new_records
    
    def record_to_vec(r):
        sfen, side, turn, total, move, winner = r

        board_vec = numpy_shogi.sfen_to_vector(sfen, usi=move)
        
        match_vec = np.array([
            1.0 if side == winner else 0.0,
            1.0 if side != winner else 0.0])

        return (np.squeeze(board_vec, axis=0), match_vec, [math.sqrt(float(turn)/float(total))])

    records_iter = iter(records)

    def create_batch():
        board_list = []
        match_list = []
        weight_list = []

        for i in range(MINI_BATCH_SIZE):
            b,m,w = record_to_vec(records_iter.__next__())

            board_list.append(b)
            match_list.append(m)
            weight_list.append(w)

        board_batch = np.stack( tuple(board_list) )
        match_batch = np.stack( tuple(match_list) )
        weight_batch = np.stack( tuple(weight_list) )

        return board_batch, match_batch, weight_batch

    print('train')

    for i in range(0, 50000):
        board_batch, match_batch, weight_batch = create_batch()

        if (i + 1) % 10 == 0:
            summary, out, res = sess.run( (merged, tags['out'], train),
                feed_dict={ input_vector: board_batch, tags['label']: match_batch,
                    tags['turn_weight']: weight_batch} )
            writer.add_summary(summary, i)
        else:
            out, res = sess.run( (tags['out'], train),
                feed_dict={ input_vector: board_batch, tags['label']: match_batch,
                    tags['turn_weight']: weight_batch} )

        if False:
            ret_winner = 'b' if out[0][0] > out[0][1] else 'w'
            score.append(1 if ret_winner == winner else 0)

            print('loop: {}, {}, {}, {}, {}'.format(i, out,
                '<' if out[0][0] > out[0][1] else '>',
                '<' if match_vec[0] < match_vec[1] else '>',
                sum(score) / len(score)))

            if len(score) == 200:
                score.pop(0)
        else:
            print('loop: {}'.format(i))

        if i > 10 and i % 4000 == 1:
            print('save backup to: {}'.format(model_backup_path))
            saver.save(sess, str(model_backup_path))

    print('save to: {}'.format(model_path))
    saver.save(sess, str(model_path))

    # finalize

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

