#coding:utf-8
__author__ = "Tuxun 2017.3.22"

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


def _variable_on_cpu(name, shape, initializer, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
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



def batch_norm(inputs, n_out, is_training):
    with tf.variable_scope('bn'):
        scale = _variable_on_cpu('scale', [n_out], tf.constant_initializer(1.0))
        beta = _variable_on_cpu('beta', [n_out], tf.constant_initializer(0.0))
        pop_mean = _variable_on_cpu('pop_mean', [n_out], tf.constant_initializer(0.0), trainable=False)
        pop_var = _variable_on_cpu('pop_var', [n_out], tf.constant_initializer(1.0), trainable=False)
        epsilon = 1e-3
        decay = 0.99
        def _bn_on_train():
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='moments')
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        def _bn_not_train():
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)

        normed = tf.cond(is_training,
                            _bn_on_train,
                            _bn_not_train)
    return normed



def _relu_conv2d(incomming, in_size, nb_row, nb_col, nb_filter, phase_train, bias=True, weight_decay=0.001, name="Relu_conv_unit"):
    with tf.variable_scope(name) as scope:
        biases = _variable_on_cpu('biases', [nb_filter], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(incomming, biases)
        activation = tf.nn.relu(pre_activation)
        kernel = _variable_with_weight_decay('weights',
                                             [3, 3, in_size, nb_filter],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(activation, kernel, [1,1,1,1], padding='SAME')
#        bn_conv = batch_norm(conv, nb_filter, phase_train)
    return(conv)


def _res_unit(incomming, nb_filter, phase_train, name="res_unit"):
    with tf.variable_scope(name) as scope:
        shape = incomming.get_shape()
        (in_size, nb_row, nb_col) = (shape[3].value, shape[1].value, shape[2].value)
        residual = _relu_conv2d(incomming=incomming, in_size=in_size, nb_row=nb_row, nb_col=nb_col, nb_filter=nb_filter, phase_train=phase_train, name='relu_conv1')
        residual = _relu_conv2d(incomming=residual, in_size=in_size, nb_row=nb_row, nb_col=nb_col, nb_filter=nb_filter, phase_train=phase_train, name='relu_conv2')
        merge = tf.add(residual, incomming)
    return(merge)


def Res_unit(incomming, nb_filter, phase_train, repetations=1, name="Res_unit"):
    with tf.variable_scope(name) as scope:
        for i in range(repetations):
            incomming = _res_unit(incomming, nb_filter, phase_train, name='res_unit_'+str(i))
    return(incomming)


def lstm_resnet(c_input, p_input, t_input, c_conf, p_conf, t_conf, nb_filter, batch_size, phase_train, nb_res_unit=1, layer_num=1):
    outputs = []
    input_seq = []
    n_flow = c_conf[0]
    hidden_size = n_flow * 101 * 71

    with tf.variable_scope('conv_seq'):
        for conf_pair in [(t_conf, t_input), (p_conf, p_input), (c_conf, c_input)]:
            if conf_pair[0][1] != 0:
                conf = conf_pair[0]
                income = conf_pair[1]
                nb_flow, len_seq, map_width, map_height, conf_name = conf
                income_seq = tf.split(income, len_seq, 3)
                for i in range(len_seq, 0):
                    with tf.variable_scope(conf_name + '_' + str(i)):
                        with tf.variable_scope('conv1'):
                            kernel1 = _variable_with_weight_decay('weights1',
                                                                 shape=[3,
                                                                        3,
                                                                        n_flow,
                                                                        nb_filter],
                                                                 stddev=5e-2,
                                                                 wd=None)
                            conv1 = tf.nn.conv2d(income, kernel1, [1,1,1,1], padding='SAME')
                        residual_output = Res_unit(conv1, nb_filter, phase_train, repetations=nb_res_unit)
                        bias = _variable_on_cpu('biases', [nb_filter], tf.constant_initializer(0.0))
                        pre_activition = tf.nn.bias_add(residual_output, bias)
                        activition = tf.nn.relu(pre_activition, name='relu1')
                        with tf.variable_scope('conv2'):
                            kernel2 = _variable_with_weight_decay('weights2',
                                                                 shape=[3,
                                                                        3,
                                                                        nb_filter,
                                                                        n_flow],
                                                                 stddev=5e-2,
                                                                 wd=None)
                            conv2 = tf.nn.conv2d(activition, kernel2, [1,1,1,1], padding='SAME')
                        input_seq.append(conv2)

    with tf.variable_scope('lstm_layer'):
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        state = init_state
        k = 0
        for seq in input_seq:
            if k > 0:
                tf.get_variable_scope().reuse_variables()
            seq1 = tf.reshape(seq, [None, 101 * 71 * n_flow])
            cell_output, state = mlstm_cell(seq1, state)
        lstm_output = tf.reshape(cell_output, [None, 101, 71, n_flow])

#    with tf.variable_scope('merge_three_temporal') as scope:
#        merge_outputs = tf.add_n(outputs, name='add_n')

    with tf.variable_scope('linear') as scope:
        w_a = _variable_on_cpu('w_a', [1, 101, 71, nb_flow], tf.truncated_normal_initializer(stddev=0.03, dtype=tf.float32))
        tf.summary.scalar('w_a', tf.reduce_mean(w_a))
        w_b = _variable_on_cpu('w_b', [1, 101, 71, nb_flow], tf.constant_initializer(0.0))
        tf.summary.scalar('w_b', tf.reduce_mean(w_b))
        activation = tf.add(tf.multiply(tf.nn.sigmoid(lstm_output), w_a), w_b)
#        activation = tf.nn.relu(activation)
    return(activation)
