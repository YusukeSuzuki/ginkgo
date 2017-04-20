import tensorflow as tf
import castanea as cas
import castanea.layers as cal

def inference(minibatch, reuse=False, var_device='/cpu:0'):
    x = minibatch

    p1 = cal.LayerParameter(rectifier=tf.nn.relu, var_device=var_device)
    p2 = cal.LayerParameter(rectifier=tf.nn.relu, var_device=var_device)
    p3 = cal.LayerParameter(rectifier=tf.nn.softmax, var_device=var_device)

    with tf.variable_scope('inference', reuse=reuse):
        x = cal.conv2d(x, 9, 9, 1080, parameter=p1)
        x = cal.conv2d(x, 7, 7, 720, parameter=p1)
        x = cal.conv2d(x, 5, 5, 720, parameter=p1)
        x = cal.conv2d(x, 3, 3, 360, parameter=p1)
        x = cal.conv2d(x, 3, 3, 180, parameter=p1)
        x = cal.conv2d(x, 3, 3, 90, parameter=p1)
        x = cal.conv2d(x, 3, 3, 90, strides=[1,2,2,1], parameter=p1)
        x = cal.conv2d(x, 3, 3, 90, strides=[1,2,2,1], parameter=p1)
        x = cal.linear(x, shape=[-1, 512], parameter=p2)
        x = cal.linear(x, shape=[-1, 2], parameter=p3)

        return x

