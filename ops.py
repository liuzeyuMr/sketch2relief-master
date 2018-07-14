import tensorflow as tf

from tensorflow.python.framework import ops

class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    '''
            初始化
            :param epsilon:    防零极小值
            :param momentum:   滑动平均参数
            :param name:       节点名称
            '''
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        # print("!!!!!!!!@@@@@@@@@",input_.shape)
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
def upscale(x,scale):
    _, h, w, _ = get_conv_shape(x)#获得输入的高和宽
    return resize_nearest_neighbor(x, (h * scale, w * scale))#上采样

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def get_biases(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)
def batch_norm2(x, name='BN', is_training=True):
        decay_rate = 0.99

        shape = x.get_shape().as_list()
        dim = shape[-1]
        if len(shape) == 2:
            mean, var = tf.nn.moments(x, [0], name='moments_bn_{}'.format(name))
        elif len(shape) == 4:
            mean, var = tf.nn.moments(x, [0, 1, 2], name='moments_bn_{}'.format(name))

        avg_mean = get_biases('avg_mean_bn_{}'.format(name), [1, dim], 0.0, False)
        avg_var = get_biases('avg_var_bn_{}'.format(name), [1, dim], 1.0, False)

        beta = get_biases('beta_bn_{}'.format(name), [1, dim], 0.0)
        gamma = get_biases('gamma_bn_{}'.format(name), [1, dim], 1.0)

        if is_training:
            avg_mean_assign_op = tf.assign(avg_mean, decay_rate * avg_mean
                                           + (1 - decay_rate) * mean)
            avg_var_assign_op = tf.assign(avg_var,
                                          decay_rate * avg_var
                                          + (1 - decay_rate) * var)

            with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
                ret = gamma * (x - mean) / tf.sqrt(1e-6 + var) + beta
        else:
            ret = gamma * (x - avg_mean) / tf.sqrt(1e-6 + avg_var) + beta

        return ret

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
       

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
