import tensorflow as tf
import numpy as np

def train_read_op(filename_queue):
    reader_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=reader_option)

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
        features={
            'vec': tf.FixedLenFeature([], tf.string),
            'turn_number': tf.FixedLenFeature([], tf.int64),
            'total_turn_number': tf.FixedLenFeature([], tf.int64),
            'side': tf.FixedLenFeature([], tf.int64),
            'winner_side': tf.FixedLenFeature([], tf.int64),
            })

    if True:
        vec = tf.decode_raw(features['vec'], tf.float64)
        vec = tf.cast(tf.reshape(vec, [9,9,360]), tf.float32)
    else:
        vec = tf.zeros(shape=[9,9,360])

    s = tf.cast(features['side'], tf.float32)
    w = tf.cast(features['winner_side'], tf.float32)
    label = tf.cond( tf.equal(s,w),
        lambda: tf.cast(np.array([1.0, 0.0]), tf.float32) ,
        lambda: tf.cast(np.array([0.0, 1.0]), tf.float32) )

    return vec, label

