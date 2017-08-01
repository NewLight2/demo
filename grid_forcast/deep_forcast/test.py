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
tf.app.flags.DEFINE_string('ma_decay', 0.99, """moving average decay""")


inflow_dir = 'source/inflow'
outflow_dir = 'source/outflow'

c_conf = [2, 3, 101, 71,'c_conf']
p_conf = [2, 2, 101, 71,'p_conf']
t_conf = [2, 0, 101, 71,'t_conf']


def data_ready(inflow_dir, outflow_dir):
    left_bottom_grid = [526, 322]
    top_right_grid = [626, 392]

    All_mat, max_num = get_all_output(inflow_dir, outflow_dir, left_bottom_grid, top_right_grid)
    All_batch = get_batch(All_mat, c_conf, p_conf,t_conf, FLAGS.batch_size)

    all_seq, all_seq_date = get_long_seq(All_mat, c_conf, p_conf, t_conf, 12)
    for ds in all_seq:
        for d in ds:
            print(d[0])
            print(d[1])
            print(d[3])
            print()
        print('...')
        print()
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


train_batches, test_batches, num_samples, max_num = data_ready(inflow_dir, outflow_dir)

