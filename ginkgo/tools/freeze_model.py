from argparse import ArgumentParser as AP
from pathlib import Path
import time
import importlib as im

import tensorflow as tf
import tensorflow.python.tools.freeze_graph as freeza

DEFAULT_MODEL_DIRECTORY = './'
DEFAULT_MODEL_NAME = 'model'
DEFAULT_DEVICE='cpu'
DEFAULT_DEVICE_NUM=1
DEFAULT_MINIBATCH_SIZE=256
DEFAULT_OUTPUT_NAME = 'model.pb'
DEFAULT_VAR_DEVICE = '/cpu:0'

DEFAULT_MODEL_MODULE = 'ginkgo.models.prophet_model_20170419_0'

# ------------------------------------------------------------
# command line option parser
# ------------------------------------------------------------

def create_parser():
    parser = AP(prog='freeze ginkgo model for deploy')
    parser.add_argument('--model-directory', type=str, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--model-py', type=str, default=None)
    parser.add_argument('-o','--output', type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument('-d','--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--var-device', type=str, default=DEFAULT_VAR_DEVICE)
    parser.add_argument('--num-devices', type=int, default=DEFAULT_DEVICE_NUM)
    parser.add_argument('-m','--minibatch-size', type=int, default=DEFAULT_MINIBATCH_SIZE)
    return parser

# ------------------------------------------------------------
# freeze model
# ------------------------------------------------------------

def freeze(ns):
    models_dir = Path(ns.model_directory)
    model_path = models_dir/ns.model_name

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        sess = tf.Session( config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        # build read data threads
        with tf.name_scope('input'):
            input_batches = [
                tf.placeholder(dtype=tf.float32, shape=[ns.minibatch_size, 9,9,360])
                for _ in range(ns.num_devices)]

        if ns.model_py:
            print('load external model file: {}'.format(ns.model_py))
            model_module = im.machinery.SourceFileLoader('externalmodel', ns.model_py).load_module()
        else:
            model_module = im.import_module(DEFAULT_MODEL_MODULE)

        with tf.device('/{}:0'.format(ns.device)):
            with tf.name_scope('tower_0') as scope:
                inference = model_module.inference(
                    input_batches[0], reuse=False, var_device=ns.var_device)

        inference = tf.identity(inference, name='inference')
        # create saver and logger
        saver = tf.train.Saver()

        # ready to run
        print('initialize')
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        for n in tf.all_variables():
            print(n.name)

        print('restore {}'.format(model_path))
        saver.restore(sess, str(model_path))

        # temporal checkpoint
        temp_dir = Path('/tmp/ginkgo_{}'.format(time.time()))
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_pb = 'model.pb'
        temp_ckpt = str(temp_dir/'model')
        print('create temp checkpoint into {}'.format(temp_dir))
        tf.train.write_graph(sess.graph.as_graph_def(), str(temp_dir), temp_pb)
        saver.save(sess, temp_ckpt)

        # run
        print('freeze')

        freeza.freeze_graph(
            input_graph=str(temp_dir/temp_pb), input_saver='', input_binary=False,
            input_checkpoint=temp_ckpt, output_node_names='inference',
            restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
            output_graph=ns.output, clear_devices=False, initializer_nodes=None)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = create_parser()
    namespace = parser.parse_args()
    freeze(namespace)

if __name__ == '__main__':
    main()

