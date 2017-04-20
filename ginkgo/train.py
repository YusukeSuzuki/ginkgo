import tensorflow as tf

def loss(label, inference):
    with tf.name_scope('loss'):
        l = tf.squared_difference(label, inference)
        tf.summary.scalar('loss', tf.reduce_mean(l), collections=['scalar', 'loss'])
        return l

def correct_rate(label, inference):
    with tf.name_scope('correct_rate'):
        corrects = tf.equal(tf.argmax(label,1), tf.argmax(inference,1))
        rate = tf.reduce_mean(tf.cast(corrects, tf.float32))
        tf.summary.scalar('correct_rate', rate)
        return correct_rate

