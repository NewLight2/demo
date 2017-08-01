#coding:utf-8
__author__ = "Tuxun 2017.3.22"

import tensorflow as tf
import numpy as np


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32)
    return(var)


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return(var)


def convnet1(income, conf, nb_filter, batch_size):
    nb_flow, len_seq, map_width, map_height, conf_name = conf
    with tf.variable_scope('conv') as scope:
        kernel1 = _variable_with_weight_decay('weights1',
                                             shape=[3,
                                                    3,
                                                    nb_flow*len_seq,
                                                    nb_flow],
                                             stddev=5e-2,
                                             wd=0.005)
    conv1 = tf.nn.conv2d(income, kernel1, [1,1,1,1], padding='SAME')

    with tf.variable_scope('tanh') as scope:
        biases = _variable_on_cpu('biases', [nb_flow], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv1, biases)
        activation = tf.nn.tanh(pre_activation, name='tanh_activation')

    return(activation)


def convnet2(income, conf, nb_filter, batch_size):
    nb_flow, len_seq, map_width, map_height, conf_name = conf
    with tf.variable_scope('conv1') as scope:
        kernel1 = _variable_with_weight_decay('weights1',
                                             shape=[3,
                                                    3,
                                                    nb_flow*len_seq,
                                                    nb_filter],
                                             stddev=5e-2,
                                             wd=0.005)
        conv1 = tf.nn.conv2d(income, kernel1, [1,1,1,1], padding='SAME')

    with tf.variable_scope('conv2') as scope:
        kernel2 = _variable_with_weight_decay('weights2',
                                             shape=[3,
                                                    3,
                                                    nb_filter,
                                                    nb_flow],
                                             stddev=5e-2,
                                             wd=0.0)
        conv2 = tf.nn.conv2d(conv1, kernel2, [1,1,1,1], padding='SAME')

    with tf.variable_scope('tanh') as scope:
        biases = _variable_on_cpu('biases', [nb_flow], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv2, biases)
        activation = tf.nn.tanh(pre_activation, name='tanh_activation')

    return(activation)


def convnet_linear(income, conf, nb_filter):
    nb_flow, len_seq, map_width, map_height, conf_name = conf
    with tf.variable_scope('conv') as scope:
        kernel1 = _variable_with_weight_decay('weights1',
                                             shape=[3,
                                                    3,
                                                    nb_flow*len_seq,
                                                    nb_flow],
                                             stddev=5e-2,
                                             wd=None)
    conv1 = tf.nn.conv2d(income, kernel1, [1,1,1,1], padding='SAME')

    with tf.variable_scope('linear') as scope:
        w_a = _variable_on_cpu('w_a', [1,1,1,nb_flow], tf.constant_initializer(0.0))
        biases = _variable_on_cpu('biases', [nb_flow], tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(tf.multiply(conv1, w_a), biases)

    return(activation)


def convnet2_linear(income, conf, nb_filter):
    nb_flow, len_seq, map_width, map_height, conf_name = conf
    with tf.variable_scope('conv') as scope:
        kernel1 = _variable_with_weight_decay('weights1',
                                             shape=[3,
                                                    3,
                                                    nb_flow*len_seq,
                                                    nb_filter],
                                             stddev=5e-2,
                                             wd=0.0)
    conv1 = tf.nn.conv2d(income, kernel1, [1,1,1,1], padding='SAME')

    with tf.variable_scope('conv2') as scope:
        kernel2 = _variable_with_weight_decay('weights2',
                                             shape=[4,
                                                    4,
                                                    nb_filter,
                                                    nb_flow],
                                             stddev=5e-2,
                                             wd=0.0)
        conv2 = tf.nn.conv2d(conv1, kernel2, [1,1,1,1], padding='SAME')

    with tf.variable_scope('linear') as scope:
        w_a = _variable_on_cpu('w_a', [1,1,1,nb_flow], tf.constant_initializer(0.0))
        biases = _variable_on_cpu('biases', [nb_flow], tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(tf.multiply(conv2, w_a), biases)

    return(activation)



def _relu_conv2d(incomming, in_size, nb_row, nb_col, nb_filter, bias=True, weight_decay=0.001, name="Relu_conv_unit"):
    with tf.variable_scope(name) as scope:
        biases = _variable_on_cpu('biases', [nb_filter], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(incomming, biases)
        activation = tf.nn.relu(pre_activation)
        kernel = _variable_with_weight_decay('weights',
                                             [3, 3, in_size, nb_filter],
                                             stddev=0.03,
                                             wd=0.005)
        conv = tf.nn.conv2d(activation, kernel, [1,1,1,1], padding='SAME')
    return(conv)


def _res_unit(incomming, nb_filter, name="res_unit"):
    with tf.variable_scope(name) as scope:
        shape = incomming.get_shape()
        (in_size, nb_row, nb_col) = (shape[3].value, shape[1].value, shape[2].value)
        residual = _relu_conv2d(incomming=incomming, in_size=in_size, nb_row=nb_row, nb_col=nb_col, nb_filter=nb_filter, name='relu_conv1')
        residual = _relu_conv2d(incomming=residual, in_size=in_size, nb_row=nb_row, nb_col=nb_col, nb_filter=nb_filter, name='relu_conv2')
        merge = tf.add(residual, incomming)
    return(merge)


def Res_unit(incomming, nb_filter, repetations=1, name="Res_unit"):
    with tf.variable_scope(name) as scope:
        for i in range(repetations):
            incomming = _res_unit(incomming, nb_filter, name='res_unit_'+str(i))
    return(incomming)


def res1_linear(income, conf, nb_filter):
    nb_flow, len_seq, map_width, map_height, conf_name = conf
    with tf.variable_scope('conv') as scope:
        kernel1 = _variable_with_weight_decay('weights1',
                                             shape=[3,
                                                    3,
                                                    nb_flow*len_seq,
                                                    nb_filter],
                                             stddev=0.03,
                                             wd=0.005)
    conv1 = tf.nn.conv2d(income, kernel1, [1,1,1,1], padding='SAME')

    res1 = Res_unit(conv1, nb_filter, repetations=2, name="res1")

    with tf.variable_scope('conv2') as scope:
        kernel2 = _variable_with_weight_decay('weights2',
                                             shape=[3,
                                                    3,
                                                    nb_filter,
                                                    nb_flow],
                                             stddev=0.03,
                                             wd=0.005)
        conv2 = tf.nn.conv2d(res1, kernel2, [1,1,1,1], padding='SAME')

    with tf.variable_scope('linear') as scope:
        w_a = _variable_on_cpu('w_a', [1, 101, 71, nb_flow], tf.truncated_normal_initializer(stddev=0.03, dtype=tf.float32))
        tf.summary.scalar('w_a', tf.reduce_mean(w_a))
        biases = _variable_on_cpu('biases', [1, 101, 71, nb_flow], tf.constant_initializer(0.0))
        tf.summary.scalar('biases', tf.reduce_mean(biases))
        activation = tf.add(tf.multiply(conv2, w_a), biases)

    return(activation)


def linearNet(c_input, p_input, t_input, c_conf, p_conf, t_conf, nb_filter, phase_train, nb_res_unit=1):
    outputs = []
    for conf_pair in [(c_conf, c_input), (p_conf, p_input), (t_conf, t_input)]:
        if conf_pair[0][1] != 0:
            conf = conf_pair[0]
            income = conf_pair[1]
            nb_flow, len_seq, map_width, map_height, conf_name = conf
            with tf.variable_scope(conf_name):
                with tf.variable_scope('conv1'):
                    kernel2 = _variable_with_weight_decay('weights1',
                                                         shape=[3,
                                                                3,
                                                                nb_flow*len_seq,
                                                                nb_flow],
                                                         stddev=5e-2,
                                                         wd=None)
                    conv2 = tf.nn.conv2d(income, kernel2, [1,1,1,1], padding='SAME')
                outputs.append(conv2)

    with tf.variable_scope('merge_three_temporal') as scope:
        merge_outputs = tf.add_n(outputs, name='add_n')

    with tf.variable_scope('linear') as scope:
        w_a = _variable_on_cpu('w_a', [1, 101, 71, nb_flow], tf.truncated_normal_initializer(stddev=0.03, dtype=tf.float32))
        tf.summary.scalar('w_a', tf.reduce_mean(w_a))
        w_b = _variable_on_cpu('w_b', [1, 101, 71, nb_flow], tf.constant_initializer(0.0))
        tf.summary.scalar('w_b', tf.reduce_mean(w_b))
        activation = tf.add(tf.multiply(tf.nn.sigmoid(merge_outputs), w_a), w_b)
    return(activation)
