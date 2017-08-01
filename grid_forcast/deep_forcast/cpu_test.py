#coding:utf-8

import tensorflow as tf
import numpy as np
from models.resnet import *
from models.testnet import *
from data_process.get_time_arrange import *
from data_process.batch_prepare import *
import json
import random
import datetime


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'log/default', """log dir.""")
tf.app.flags.DEFINE_integer('num_gpus', 2, """use gpus""")
tf.app.flags.DEFINE_integer('batch_size', 32, """batch size""")
tf.app.flags.DEFINE_integer('nb_filter', 48, """how many filters in resnet""")
tf.app.flags.DEFINE_integer('steps', 10000, """train steps""")
tf.app.flags.DEFINE_string('lr', 0.001, """learning rate""")


inflow_dir = 'source/inflow'
outflow_dir = 'source/outflow'

c_conf = [2, 4, 101, 71,'c_conf']
p_conf = [2, 3, 101, 71,'p_conf']
t_conf = [2, 0, 101, 71,'t_conf']


def data_ready(inflow_dir, outflow_dir):
    left_bottom_grid = [526, 322]
    top_right_grid = [626, 392]

    All_mat = get_all_output(inflow_dir, outflow_dir, left_bottom_grid, top_right_grid)
    All_batch = get_batch(All_mat, c_conf, p_conf,t_conf, FLAGS.batch_size)

    len_train_batch = int(len(All_batch) * 0.9)

    random.seed(10)
    train_index = random.sample(range(len(All_batch)), len_train_batch)
    test_index = set(range(len(All_batch))) - set(train_index)
    train_batches = [All_batch[i] for i in train_index]
    test_batches = [All_batch[i] for i in test_index]

    print("train_samples: %d" % len(train_batches * FLAGS.batch_size))
    print("train_samples: %d" % len(test_batches * FLAGS.batch_size))

    test_input = []
    for j in range(4):
        try:
            inp = np.concatenate([i[j] for i in test_batches], axis = 0)
        except ValueError:
            inp = None
        test_input.append(inp)

    return(train_batches, test_input, len(train_batches))


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return(average_grads)


with tf.Graph().as_default(), tf.device('/cpu:0'):
    lr = FLAGS.lr
    lr = float(lr)
    print(lr)
    opt = tf.train.AdamOptimizer(lr)
    tower_grads = []
    c_placeholder = tf.placeholder(tf.float32, [None, 101, 71, c_conf[0] * c_conf[1]], name='c_input')
    p_placeholder = tf.placeholder(tf.float32, [None, 101, 71, p_conf[0] * p_conf[1]], name='p_input')
    t_placeholder = tf.placeholder(tf.float32, [None, 101, 71, t_conf[0] * t_conf[1]], name='t_input')
    y_placeholder = tf.placeholder(tf.float32, [None, 101, 71, 2], name='y_input')
    c_places = tf.split(c_placeholder, FLAGS.num_gpus, 0)
    p_places = tf.split(p_placeholder, FLAGS.num_gpus, 0)
    t_places = tf.split(t_placeholder, FLAGS.num_gpus, 0)
    y_places = tf.split(y_placeholder, FLAGS.num_gpus, 0)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    activation = st_resnet(c_places[i], p_places[i], t_places[i], c_conf, p_conf, t_conf, FLAGS.nb_filter, 4)
#                    activation = convnet_linear(c_places[i], c_conf, FLAGS.nb_filter)
                    loss = tf.reduce_mean(tf.square(activation - y_places[i]), name='loss')
                    tf.add_to_collection('losses', loss)
                    losses = tf.get_collection('losses', scope)
                    total_loss = tf.add_n(losses, name='total_loss')
                    for l in losses + [total_loss]:
                      # session. This helps the clarity of presentation on tensorboard.
                        loss_name = l.op.name
                        tf.summary.scalar(loss_name, l)
                    tf.get_variable_scope().reuse_variables()
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = opt.compute_gradients(total_loss)
                    tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    train_op = opt.apply_gradients(grads, global_step=None)
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    init = tf.global_variables_initializer()
#    summary_op = tf.summary.merge(summaries)
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(init)
        train_batches, test_batches, num_samples = data_ready(inflow_dir, outflow_dir)
        saver.restore(sess, 'save/model.ckpt')
        [nb_c_batch_mat, nb_p_batch_mat, nb_t_batch_mat, nb_y_batch_mat] = random.choice(train_batches)
        date_1 = datetime.datetime.now()
        date_str = datetime.datetime.strftime(date_1, '%Y-%m-%d %H:%M:%S')
        print('time %s' % date_str)
        sess.run(train_op, feed_dict = {c_placeholder: nb_c_batch_mat, p_placeholder: nb_p_batch_mat, y_placeholder: nb_y_batch_mat})
        date_2 = datetime.datetime.now()
        time_delta = float((date_2 - date_1).seconds) + (date_2 - date_1).microseconds / 1000000
        date_str = datetime.datetime.strftime(date_2, '%Y-%m-%d %H:%M:%S')
        print('time %s' % date_str)
        print(time_delta)
