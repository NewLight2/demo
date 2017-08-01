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

tf.app.flags.DEFINE_string('log_dir', 'log',
                """log dir.""")


left_bottom_grid = [526, 322]
top_right_grid = [626, 392]

inflow_dir = 'source/inflow'
outflow_dir = 'source/outflow'
#inflow_test_dir = 'source/inflow_test'
#outflow_test_dir = 'source/outflow_test'

All_mat = get_all_output(inflow_dir, outflow_dir, left_bottom_grid, top_right_grid)
#All_mat_test = get_all_output(inflow_test_dir, outflow_test_dir, left_bottom_grid, top_right_grid)

c_conf = [2, 3, 101, 71,'c_conf']
p_conf = [2, 0, 101, 71,'p_conf']
t_conf = [2, 0, 101, 71,'t_conf']
batch_size = 25

All_batch = get_batch(All_mat, c_conf, p_conf,t_conf, batch_size)


c_placeholder = tf.placeholder(tf.float32, [None, 101, 71, c_conf[0] * c_conf[1]], name='c_input')
p_placeholder = tf.placeholder(tf.float32, [None, 101, 71, p_conf[0] * p_conf[1]], name='p_input')
t_placeholder = tf.placeholder(tf.float32, [None, 101, 71, t_conf[0] * t_conf[1]], name='t_input')
y_placeholder = tf.placeholder(tf.float32, [None, 101, 71, 2], name='y_input')

#activation = st_resnet(c_input=c_placeholder, p_input=p_placeholder, t_input=t_placeholder, c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, nb_filter=32, nb_res_unit=2)
activation = res1_linear(c_placeholder, c_conf, 32)

#All_batch_test = get_batch(All_mat_test, c_conf, p_conf, t_conf, batch_size)

#normalized_all_batch = []
#for one_batch in All_batch:
#    norm_one, max_value = maxmin_normalize(one_batch)
#    normalized_all_batch.append(norm_one)


len_train_batch = int(len(All_batch) * 0.9)

random.seed(10)
train_index = random.sample(range(len(All_batch)), len_train_batch)
test_index = set(range(len(All_batch))) - set(train_index)
train_batches = [All_batch[i] for i in train_index]
test_batches = [All_batch[i] for i in test_index]

print("train_samples: %d" % len(train_batches * batch_size))
print("train_samples: %d" % len(test_batches * batch_size))

test_c_input = np.concatenate([i[0] for i in test_batches], axis = 0)
test_y_input = np.concatenate([i[3] for i in test_batches], axis = 0)

loss = tf.reduce_mean(tf.square(activation - y_placeholder))
tf.summary.scalar('loss', loss)
merged_summary = tf.summary.merge_all()


opt = tf.train.AdamOptimizer(0.001)
train_step = opt.minimize(loss)
init = tf.global_variables_initializer()
variables = tf.trainable_variables()
#weight1 = [var for var in variables if 'weights2' in var.name]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    for i in range(100000):
        [nb_c_batch_mat, nb_p_batch_mat, nb_t_batch_mat, nb_y_batch_mat] = random.choice(train_batches)
        _, summary = sess.run([train_step, merged_summary], feed_dict={c_placeholder: nb_c_batch_mat, y_placeholder:nb_y_batch_mat})
        val_train = sess.run(loss, feed_dict={c_placeholder: nb_c_batch_mat, y_placeholder:nb_y_batch_mat})
        train_writer.add_summary(summary, i)
#        sess.run(train_step, feed_dict={c_placeholder:nb_c_batch_mat, p_placeholder:nb_p_batch_mat, t_placeholder:nb_t_batch_mat, y_placeholder: nb_y_batch_mat})
#        val, w1 = sess.run([loss,weight1], feed_dict={c_placeholder:nb_c_batch_mat, p_placeholder:nb_p_batch_mat, t_placeholder:nb_t_batch_mat, y_placeholder: nb_y_batch_mat})
        if i % 50 == 0:
            val_test, summary_test = sess.run([loss, merged_summary], feed_dict={c_placeholder: test_c_input, y_placeholder:test_y_input})
            test_writer.add_summary(summary_test, i)
            print('step '+ str(i+1) + ': train loss = '+str(val_train) + ', test loss = ' + str(val_test))

    inflow_pre_file = open('result/inflow_pre.json', 'w')
    inflow_real_file = open('result/inflow_real.json', 'w')
    inflow_delta_file = open('result/inflow_delta.json', 'w')
    outflow_pre_file = open('result/outflow_pre.json', 'w')
    outflow_real_file = open('result/outflow_real.json', 'w')
    outflow_delta_file = open('result/outflow_delta.json', 'w')
    w_x, w_y = sess.run([activation, y_placeholder], feed_dict={c_placeholder: test_c_input, y_placeholder:test_y_input})
    inflow_pre = w_x[:,:,:,0]
    outflow_pre = w_x[:,:,:,1]
    inflow_real = w_y[:,:,:,0]
    outflow_real = w_y[:,:,:,1]

    inflow_delta = np.abs(inflow_real - inflow_pre)
    outflow_delta = np.abs(outflow_real - outflow_pre)

    print(w_x.shape)
    print(len_test)
    def write_file(mat):
        result = {}
        for k in range(0, 50, 1):
            result[k] = []
            for i in range(101):
                for j in range(71):
                    result[k].append([i, j, max(0, int(mat[k, i, j]))])
        return(result)

    A = write_file(inflow_pre)
    B = write_file(inflow_real)
    C = write_file(outflow_pre)
    D = write_file(outflow_real)
    inflow_pre_file.write(json.dumps(A))
    inflow_real_file.write(json.dumps(B))
    outflow_pre_file.write(json.dumps(C))
    outflow_real_file.write(json.dumps(D))
    outflow_delta_file.write(json.dumps(write_file(outflow_delta)))
    inflow_delta_file.write(json.dumps(write_file(inflow_delta)))

