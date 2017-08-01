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
tf.app.flags.DEFINE_integer('seq_length', 24, """batch size""")
tf.app.flags.DEFINE_integer('nb_filter', 48, """how many filters in resnet""")
tf.app.flags.DEFINE_integer('steps', 10000, """train steps""")
tf.app.flags.DEFINE_string('lr', 0.001, """learning rate""")


inflow_dir = 'source/inflow'
outflow_dir = 'source/outflow'
all_dir = 'source/gridcount'

c_conf = [3, 3, 101, 71,'c_conf']
p_conf = [3, 2, 101, 71,'p_conf']
t_conf = [3, 0, 101, 71,'t_conf']
y_conf = [3, 1, 101, 71,'t_conf']


def tran180(mat):
    yn = mat.shape[1]
    mat1 = mat.copy()
    for i in range(yn):
        mat1[:, i] = mat[:,yn-1-i]
    mat1[0, 16] = 0
    mat1[2, 12] = 0
    mat1[57, 21] = 0
    mat1[59, 18] = 0
    mat1[46, 11] = 0
    mat1[53, 18] = 0
    return(mat1)


def data_ready(inflow_dir, outflow_dir, all_dir):
    left_bottom_grid = [526, 322]
    top_right_grid = [626, 392]

    All_mat, max_num = get_all_output(inflow_dir, outflow_dir, all_dir, left_bottom_grid, top_right_grid)
    all_seq, all_seq_date = get_long_seq(All_mat, c_conf, p_conf, t_conf, FLAGS.seq_length)

    return(all_seq, all_seq_date)


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


def write_grid_data(kind_dir, seq, real_seq, id_all):
    gridinfo = {}
    gridinfo["status"] = 'success'
    gridinfo["message"] = ''
    gridinfo['data'] = {}
    thisflow = [str(i[lon*71 + lat]) for i in seq[0:3]]
    preflow = [str(i[lon*71 + lat]) for i in seq[3:]]
    realflow = [str(i[lon*71 + lat]) for i in real_seq]
    for i in realflow:
        if int(i) > 10000:
            print(id_all)
            break
    gridinfo['data']["yesterdayFlow"] = realflow
    gridinfo['data']["thisdayFlow"] = thisflow
    gridinfo['data']["predictFlow"] = preflow
    gridinfo['data']["xCategories"] = times_short
    result = json.dumps(gridinfo)
    with open(kind_dir + id_all + '.json', 'w') as fw:
        fw.write(result)


def write_seq(file_dir, seq, times_short):
    with open(file_dir, 'w') as fw:
        head = '''{"status":"success","message":"","data":{"timeList":["'''+ '","'.join(times_short[0:3]) + '","'+times_short[3]+'(P)","'+'","'.join(times_short[4:])+'''"],"valueList":['''
        fw.write(head + '\n')
        all_line = ''
        for lst in seq:
            line = '["' + '","'.join([str(i) for i in lst]) + '"]'
            all_line = all_line + line + ',\n'
        all_line = all_line.rstrip(',\n')
        fw.write(all_line + '\n')
        tail = '''],"predictTime":"''' + times[3] + '''"}}'''
        fw.write(tail)


def write_current(file_dir, seq):
    with open(file_dir, 'w') as fw:
        head = '''{"status":"success","message":"","data":{"record":'''
        fw.write(head + '\n')
        fw.write('["' + '","'.join([str(i) for i in seq[3]]) + '"]' + '\n')
        tail = ''',"predictTime":"''' + times[3] + '''"}}'''
        fw.write(tail)


def reform_mat(mat, num):
    ac1 = mat * num
    ac1[ac1 < 0] = 0
    ac2 = ac1.astype('int')
    return(ac2)


with tf.Graph().as_default(), tf.device('/cpu:0'):
    lr = FLAGS.lr
    lr = float(lr)
    print(lr)
    opt = tf.train.AdamOptimizer(lr)
    tower_grads = []
    c_placeholder = tf.placeholder(tf.float32, [None, 101, 71, c_conf[0] * c_conf[1]], name='c_input')
    p_placeholder = tf.placeholder(tf.float32, [None, 101, 71, p_conf[0] * p_conf[1]], name='p_input')
    t_placeholder = tf.placeholder(tf.float32, [None, 101, 71, t_conf[0] * t_conf[1]], name='t_input')
    y_placeholder = tf.placeholder(tf.float32, [None, 101, 71, y_conf[0] * y_conf[1]], name='y_input')
    is_training = tf.placeholder('bool', [], name='is_training')
    d_num = tf.placeholder(tf.float32, [], name='d_num')
    a_num = tf.placeholder(tf.float32, [], name='a_num')
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device('/cpu:0'):
            with tf.name_scope('tower_0') as scope:
                c_pre = div_diff(c_placeholder, c_conf[0], c_conf[1], [d_num, d_num, a_num])
                p_pre = div_diff(p_placeholder, p_conf[0], p_conf[1], [d_num, d_num, a_num])
                t_pre = div_diff(t_placeholder, t_conf[0], t_conf[1], [d_num, d_num, a_num])
                y_pre = div_diff(y_placeholder, y_conf[0], y_conf[1], [d_num, d_num, a_num])
                activation = st_resnet(c_pre, p_pre, t_pre,
                        c_conf, p_conf, t_conf, FLAGS.nb_filter, is_training, nb_res_unit=4)
                loss = tf.sqrt(tf.reduce_mean(tf.square(activation - y_pre)), name='loss')
#                tf.add_to_collection('losses', loss)
#                losses = tf.get_collection('losses', scope)
#                total_loss = tf.add_n(losses, name='total_loss')
#                for l in losses + [total_loss]:
                  # session. This helps the clarity of presentation on tensorboard.
#                    loss_name = l.op.name
#                    tf.summary.scalar(loss_name, l)
#                tf.get_variable_scope().reuse_variables()
#                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
#                grads = opt.compute_gradients(total_loss)
#                tower_grads.append(grads)
#    grads = average_gradients(tower_grads)
#    train_op = opt.apply_gradients(grads, global_step=None)
#    for grad, var in grads:
#        if grad is not None:
#            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

#    init = tf.global_variables_initializer()
#    summary_op = tf.summary.merge(summaries)
#    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        #sess.run(init)
        all_seq, all_seq_date = data_ready(inflow_dir, outflow_dir, all_dir)
        saver.restore(sess, 'save/model.ckpt')
        for one_seq in range(len(all_seq)):
            date_1 = datetime.datetime.now()
            date_str = datetime.datetime.strftime(date_1, '%Y-%m-%d %H:%M:%S')
            one_loss = []
            k = 0
            close_pre = None
            inflow_seq = []
            outflow_seq = []
            allflow_seq = []
            last_allflow_mat = None
            real_inflow_seq = []
            real_outflow_seq = []
            real_allflow_seq = []
            for d in range(len(all_seq[one_seq])):
                mat = all_seq[one_seq][d]
                if k == 0:
                    loss0, ac0 = sess.run([loss, activation], feed_dict = {c_placeholder: mat[0],
                        p_placeholder: mat[1], y_placeholder: mat[3],
                        is_training: False, d_num: 5000, a_num: 50000})
                    close_pre = mat[0][:,:,:,0:3*(c_conf[1]-1)]
                    for i in range(c_conf[1], 0, -1):
                        inflow_1 = tran180(mat[0][:,:,:,3*(i-1)].squeeze()).ravel()
                        inflow_seq.append(list(inflow_1))
                        real_inflow_seq.append(list(inflow_1))
                        outflow_1 = tran180(mat[0][:,:,:,3*(i-1)+1].squeeze()).ravel()
                        outflow_seq.append(list(outflow_1))
                        real_outflow_seq.append(list(outflow_1))
                        last_allflow_mat = mat[0][:,:,:,3*(i-1)+2]
                        all_1 = tran180(mat[0][:,:,:,3*(i-1)+2].squeeze()).ravel()
                        allflow_seq.append(list(all_1))
                        real_allflow_seq.append(list(all_1))
                else:
                    close_data = np.concatenate([ac2, close_pre], axis=3)
                    loss0, ac0 = sess.run([loss, activation], feed_dict = {c_placeholder: close_data,
                        p_placeholder: mat[1], y_placeholder: mat[3],
                        is_training: False, d_num: 5000, a_num: 50000})
                    close_pre = close_data[:,:,:,0:3*(c_conf[1]-1)]
                k += 1
                inflow_slice = reform_mat(ac0[:,:,:,0], 5000)
                outflow_slice = reform_mat(ac0[:,:,:,1], 5000)
                last_allflow_mat = last_allflow_mat + inflow_slice - outflow_slice
                last_allflow_mat = reform_mat(last_allflow_mat, 1)
                ac2 = np.stack([inflow_slice, outflow_slice, reform_mat(ac0[:,:,:,2], 50000)], axis=3)
                inflow_seq.append(list(tran180(inflow_slice.squeeze()).ravel()))
                outflow_seq.append(list(tran180(outflow_slice.squeeze()).ravel()))
                allflow_seq.append(list(tran180(last_allflow_mat.squeeze()).ravel()))
                real_inflow_seq.append(list(tran180(mat[3][:,:,:,0].squeeze()).ravel()))
                real_outflow_seq.append(list(tran180(mat[3][:,:,:,1].squeeze()).ravel()))
                real_allflow_seq.append(list(tran180(mat[3][:,:,:,2].squeeze()).ravel()))
                print('pre: %d, real: %d, delta: %f' % (np.sum(ac2), np.sum(mat[3]), (np.sum(ac2) - np.sum(mat[3])) / np.sum(ac2)))
                one_loss.append(str(loss0))
            date_2 = datetime.datetime.now()
            time_delta = float((date_2 - date_1).seconds) + (date_2 - date_1).microseconds / 1000000
            date_str = datetime.datetime.strftime(date_2, '%Y-%m-%d %H:%M:%S')
            print('time_cost: %f     one_loss: %s' % (time_delta, ','.join(one_loss)))

            times = []
            p = 0
            for date_lst in all_seq_date[one_seq]:
                if p == 0:
                    times.extend(date_lst[0])
                    times.reverse()
                    p += 1
                times.append(date_lst[3])
            print(times[0])
            times_short = [i[8:10]+':'+i[10:] for i in times]
            if times[0] == '201701151700':
                lonrange = range(101)
                latrange = range(71)
                for lon in lonrange:
                    for lat in latrange:
                        id_lon = '%03d' % lon
                        id_lat = '%03d' % lat
                        id_all = str(id_lon) + '_' + str(id_lat)
                        write_grid_data('/workspace/xwtech/tuxun/testwork/flasktest/data/inflow_grid/',
                                        inflow_seq, real_inflow_seq, id_all)
                        write_grid_data('/workspace/xwtech/tuxun/testwork/flasktest/data/outflow_grid/',
                                        outflow_seq, real_outflow_seq, id_all)
                        write_grid_data('/workspace/xwtech/tuxun/testwork/flasktest/data/allflow_grid/',
                                        allflow_seq, real_allflow_seq, id_all)
                write_seq('/workspace/xwtech/tuxun/testwork/flasktest/data/inflow_seq.json', inflow_seq, times_short)
                write_seq('/workspace/xwtech/tuxun/testwork/flasktest/data/outflow_seq.json', outflow_seq, times_short)
                write_seq('/workspace/xwtech/tuxun/testwork/flasktest/data/allflow_seq.json', allflow_seq, times_short)
                write_current('/workspace/xwtech/tuxun/testwork/flasktest/data/inflow_current.json', inflow_seq)
                write_current('/workspace/xwtech/tuxun/testwork/flasktest/data/outflow_current.json', outflow_seq)
                write_current('/workspace/xwtech/tuxun/testwork/flasktest/data/allflow_current.json', allflow_seq)
                break
