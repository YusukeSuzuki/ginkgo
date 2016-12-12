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

MINI_BATCH_SIZE = 100

# ------------------------------------------------------------
# sub commands
# ------------------------------------------------------------

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        
        for g, u in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad = tf.concat(0,grads)
        grad = tf.reduce_mean(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

def do_train(ns):
    models_dir = Path(ns.modeldir)
    model_path = models_dir/ns.output_model
    model_backup_path = models_dir/(ns.output_model+'.back')

    # build read data threads
    path_list = list(Path(ns.samples).glob('*.csa'))
    shuffle(path_list)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        sess = tf.Session( config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        coordinator = tf.train.Coordinator()

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        opt = tf.train.AdamOptimizer(1e-4)

        # build read data threads
        with tf.variable_scope('input'):
            load_threads, input_batch, label_batch, weight_batch = \
                shogi_loader.load_sfenx_threads_and_queue(
                    coordinator, sess, path_list, MINI_BATCH_SIZE, threads_num=24)

        tower_grads = []

        for i in range(ns.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    graph_root = yl.load(ns.prophet_yaml)
                    tags = graph_root.build(feed_dict={
                        'root': input_batch, 'label': label_batch,
                        'turn_weight': weight_batch})
                    loss = tags['p_loss']

                    tf.get_variable_scope().reuse_variables()

                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        train_op = apply_gradient_op

        saver = tf.train.Saver()
        merged = tf.merge_all_summaries()

        print('initialize')
        writer = tf.train.SummaryWriter(ns.logdir, sess.graph)
        sess.run(tf.initialize_all_variables())

        if ns.input_model:
            print('restore {}'.format(ns.input_model))
            input_model = ns.modeldir+'/'+ns.input_model
            saver.restore(sess, input_model)

        writer.add_graph(tf.get_default_graph())

        print('train')

        for t in load_threads: t.start()

        gs = -1

        try:
            while not coordinator.should_stop():
                if gs % 5 == 0:
                    summary, res, gs = sess.run( (merged, train_op, global_step), feed_dict={} )
                    writer.add_summary(summary, gs)
                else:
                    res, gs = sess.run( (train_op, global_step), feed_dict={} )

                gs = int(gs)

                print('loop: {}'.format(gs))

                if gs > 10 and gs % 4000 == 1:
                    print('save backup to: {}'.format(model_backup_path))
                    saver.save(sess, str(model_backup_path))

                if gs < 100 and gs % 10:
                    writer.flush()
                if gs < 2000 and gs % 100:
                    writer.flush()
                if gs < 5000 and gs % 200:
                    writer.flush()
        except tf.errors.OutOfRangeError as e:
            print('sample exausted')

        print('save to: {}'.format(model_path))
        saver.save(sess, str(model_path))

        # finalize
        coordinator.request_stop()
        coordinator.join(load_threads)
        writer.close()

def do_test(ns):
    models_dir = Path(MODELS_DIR)
    model_path = models_dir/ns.model

    sess = tf.Session()
    coordinator = tf.train.Coordinator()

    # build read data threads
    path_list = list(Path(ns.samples).glob('*.csa'))

    with tf.variable_scope('input'), tf.device('/cpu:0'):
        load_threads, input_batch, label_batch, weight_batch = \
            shogi_loader.load_sfenx_threads_and_queue(
                coordinator, sess, path_list, MINI_BATCH_SIZE,
                threads_num=24)

    # build model
    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(ns.prophet_yaml)
        tags = graph_root.build(feed_dict={
            'root': input_batch, 'label': label_batch,
            'turn_weight': weight_batch})

    # create saver and logger
    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()

    # ready to run

    print('initialize')
    writer = tf.train.SummaryWriter(ns.logdir, sess.graph)
    sess.run(tf.initialize_all_variables())

    # run
    
    ckpt = tf.train.get_checkpoint_state('models')
    print(ckpt)
    print(ckpt.model_checkpoint_path)

    if ns.input_model:
        print('restore {}'.format(ns.input_model))
        input_model = ns.modeldir+'/'+ns.input_model
        saver.restore(sess, input_model)

    writer.add_graph(tf.get_default_graph())

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

            if rate_count % 10 == 0:
                print('correct rate({}): avg {}, min {}, max {}'.format(
                    rate_count, rate_avg, rate_min, rate_max))
    except tf.errors.OutOfRangeError as e:
        print('sample exausted')

    print('correct rate: {}'.format(rate_total))

    # finalize
    coordinator.request_stop()
    coordinator.join(load_threads)
    writer.close()

def do_eval(ns):
    print('unavailable now')

def do_dump_network(ns):
    # build
    with tf.variable_scope('input'), tf.device('/cpu:0'):
        input_vector = tf.placeholder(tf.float32, [MINI_BATCH_SIZE,9,9,360])
        label_vector = tf.placeholder(tf.float32, [MINI_BATCH_SIZE,2])
        weight_vector = tf.placeholder(tf.float32, [MINI_BATCH_SIZE,1])

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(ns.prophet_yaml)
        tags = graph_root.build(feed_dict={
            'root': input_vector, 'label': label_vector,
            'turn_weight': weight_vector})

    print('-- variables')
    for variable in tf.all_variables():
        if fnmatch.fnmatch(variable.name, ns.pattern):
            print(variable.name)

    print('-- operations')
    for operation in tf.get_default_graph().get_operations():
        if fnmatch.fnmatch(operation.name, ns.pattern):
            print(operation.name)

def do_dump_graph_log(ns):
    # build
    #print('exclude tags: {}'.format(ns.exclude_tags.split(',')))

    with tf.variable_scope('input'), tf.device('/cpu:0'):
        input_vector = tf.placeholder(tf.float32, [MINI_BATCH_SIZE,9,9,360])
        label_vector = tf.placeholder(tf.float32, [MINI_BATCH_SIZE,2])
        weight_vector = tf.placeholder(tf.float32, [MINI_BATCH_SIZE,1])


    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(ns.prophet_yaml)
        tags = graph_root.build(feed_dict={'root': input_vector})

    sess = tf.Session()
    writer = tf.train.SummaryWriter(ns.logdir, sess.graph)
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
    #parser.add_argument('--modelfile', type=str, default='model.ckpt')
    #parser.add_argument('--restore', type=str, default='')

    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser('train')
    sub_parser.set_defaults(func=do_train)
    sub_parser.add_argument('--input-model', type=str, default='')
    sub_parser.add_argument('--output-model', type=str)
    sub_parser.add_argument('--prophet-yaml', type=str)
    sub_parser.add_argument('--modeldir', type=str)
    sub_parser.add_argument('--samples', type=str)
    sub_parser.add_argument('--num-gpus', type=int, default=1)

    sub_parser = sub_parsers.add_parser('test')
    sub_parser.set_defaults(func=do_test)
    sub_parser.add_argument('--input-model', type=str)
    sub_parser.add_argument('--prophet-yaml', type=str)
    sub_parser.add_argument('--modeldir', type=str)
    sub_parser.add_argument('--samples', type=str)
    sub_parser.add_argument('--num-gpus', type=int, default=1)

    sub_parser = sub_parsers.add_parser('eval')
    sub_parser.set_defaults(func=do_eval)

    sub_parser = sub_parsers.add_parser('dump-network')
    sub_parser.set_defaults(func=do_dump_network)
    sub_parser.add_argument('--pattern', type=str, default='*')
    sub_parser.add_argument('--prophet-yaml', type=str)

    sub_parser = sub_parsers.add_parser('dump-graph-log')
    sub_parser.set_defaults(func=do_dump_graph_log)
    sub_parser.add_argument('--exclude-tags', type=str, default='')
    sub_parser.add_argument('--prophet-yaml', type=str)

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

