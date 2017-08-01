#coding:utf-8

import tensorflow as tf
import numpy as np
from models.resnet import *
from models.testnet import *
from models.lstm_res import lstm_resnet
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
tf.app.flags.DEFINE_string('ma_decay', 0.99, """moving average decay""")


inflow_dir = 'source/inflow'
outflow_dir = 'source/outflow'
all_dir = 'source/gridcount'

c_conf = [3, 3, 101, 71,'c_conf']
p_conf = [3, 2, 101, 71,'p_conf']
t_conf = [3, 0, 101, 71,'t_conf']
y_conf = [3, 1, 101, 71,'y_conf']


def data_ready(inflow_dir, outflow_dir, all_dir):
    left_bottom_grid = [526, 322]
    top_right_grid = [626, 392]

    All_mat, max_num = get_all_output(inflow_dir, outflow_dir, all_dir, left_bottom_grid, top_right_grid)
    print(max_num)
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

    return(train_batches, test_input, len(train_batches), max_num)


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

def div_diff(mat, p, q, numlst):
    if q == 0 or q == 0:
        return(mat)
    slices = tf.split(mat, p*q, 3)
    div_results = []
    for i in range(q):
        for j in range(p):
            sl = slices[i*p + j]
            num = numlst[j]
            div_result = tf.divide(sl, num)
            div_results.append(div_result)
    pre = tf.concat(div_results, 3)
    return(pre)

with tf.Graph().as_default(), tf.device('/cpu:0'):
    lr = FLAGS.lr
    lr = float(lr)
    print(lr)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, 5000, 0.1, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate)
    tower_grads = []
    c_placeholder = tf.placeholder(tf.float32, [None, 101, 71, c_conf[0] * c_conf[1]], name='c_input')
    p_placeholder = tf.placeholder(tf.float32, [None, 101, 71, p_conf[0] * p_conf[1]], name='p_input')
    t_placeholder = tf.placeholder(tf.float32, [None, 101, 71, t_conf[0] * t_conf[1]], name='t_input')
    y_placeholder = tf.placeholder(tf.float32, [None, 101, 71, c_conf[0]], name='y_input')
    c_places = tf.split(c_placeholder, FLAGS.num_gpus, 0)
    p_places = tf.split(p_placeholder, FLAGS.num_gpus, 0)
    t_places = tf.split(t_placeholder, FLAGS.num_gpus, 0)
    y_places = tf.split(y_placeholder, FLAGS.num_gpus, 0)
    is_training = tf.placeholder('bool', [], name='is_training')
    d_num = tf.placeholder(tf.float32, [], name='d_num')
    a_num = tf.placeholder(tf.float32, [], name='a_num')
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    c_pre = div_diff(c_places[i], c_conf[0], c_conf[1], [d_num, d_num, a_num])
                    p_pre = div_diff(p_places[i], p_conf[0], p_conf[1], [d_num, d_num, a_num])
                    t_pre = div_diff(t_places[i], t_conf[0], t_conf[1], [d_num, d_num, a_num])
                    y_pre = div_diff(y_places[i], y_conf[0], y_conf[1], [d_num, d_num, a_num])
                    activation = lstm_resnet(c_pre, p_pre, t_pre,
                            c_conf, p_conf, t_conf, FLAGS.nb_filter, FLAGS.batch_size, is_training, nb_res_unit=4)
#                    activation = convnet_linear(c_places[i], c_conf, FLAGS.nb_filter)
                    loss = tf.sqrt(tf.reduce_mean(tf.square(activation - y_pre)), name='loss')
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
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
                    FLAGS.ma_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)

    init = tf.global_variables_initializer()
#    summary_op = tf.summary.merge(summaries)
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.logdir + '/test')
        train_batches, test_batches, num_samples, max_num = data_ready(inflow_dir, outflow_dir, all_dir)
        date_last = datetime.datetime.now()
        for i in range(FLAGS.steps):
            [nb_c_batch_mat, nb_p_batch_mat, nb_t_batch_mat, nb_y_batch_mat] = random.choice(train_batches)
            sess.run(train_op, feed_dict={c_placeholder: nb_c_batch_mat,
                p_placeholder: nb_p_batch_mat, y_placeholder:nb_y_batch_mat,
                is_training: True, d_num: 5000, a_num: 50000})
            if i % 100 == 0 :
                val_train, train_summary = sess.run([loss, summary_op],
                        feed_dict = {c_placeholder: nb_c_batch_mat,
                            p_placeholder: nb_p_batch_mat, y_placeholder: nb_y_batch_mat,
                            is_training: False, d_num: 5000, a_num: 50000})
                val_test, test_summary = sess.run([loss, summary_op],
                        feed_dict = {c_placeholder: test_batches[0],
                            p_placeholder: test_batches[1], y_placeholder: test_batches[3],
                            is_training: False, d_num: 5000, a_num: 50000})
                date_now = datetime.datetime.now()
                date_str = datetime.datetime.strftime(date_now, '%Y-%m-%d %H:%M:%S')
                time_delta = str(float((date_now - date_last).seconds) + (date_now - date_last).microseconds / 1000000)
                print('%s, step: %d, train_loss: %f, test_loss: %f, time cost: %s' % (date_str, i, val_train, val_test, time_delta))
                if i != 0:
                    train_writer.add_summary(train_summary, i)
                    test_writer.add_summary(test_summary, i)
                date_last = datetime.datetime.now()
        saver_path = saver.save(sess, 'save/model.ckpt')
