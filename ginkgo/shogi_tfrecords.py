import tensorflow as tf
import numpy as np

def train_read_op(filename_queue, reader):
    _, sirialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
        feartures={
            'vec': tf.FixedLenFeature([], tf.string),
            'turn_number': tf.FixedLenFeature([], tf.int64),
            'total_turn_number': tf.FixedLenFeature([], tf.int64),
            'side': tf.FixedLenFeature([], tf.int64),
            'winner_side': tf.FixedLenFeature([], tf.int64),
            })

    vec = tf.decode_raw(features['vec'], tf.float32)
    vec = tf.reshape([9,9,360])
    label = np.array([
        1.0 if features['side'] == features['winner_side'] else 0.0,
        1.0 if features['side'] != features['winner_side'] else 0.0])

    return vec, label

