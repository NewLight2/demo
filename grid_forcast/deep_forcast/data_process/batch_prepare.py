#coding:utf-8

import tensorflow as tf
import numpy as np
import json
import random
import datetime

def date_continue(date_str1, date_str2, time_interval):
    date1 = datetime.datetime.strptime(date_str1, '%Y%m%d%H%M')
    date2 = datetime.datetime.strptime(date_str2, '%Y%m%d%H%M')
    if (date2 - date1).seconds//60 == time_interval:
        return(True)
    else:
        return(False)


def get_date_str(date_str, time_interval):
    date1 = datetime.datetime.strptime(date_str, '%Y%m%d%H%M')
    date2 = date1 + datetime.timedelta(seconds = 60 * time_interval)
    date2_str = datetime.datetime.strftime(date2, '%Y%m%d%H%M')
    return(date2_str)


def get_batch(All_mat, c_conf, p_conf, t_conf, batch_size):

    c_seq_length = c_conf[1]
    p_seq_length = p_conf[1]
    t_seq_length = t_conf[1]
    nb_filter = c_conf[0]
    nb_row = c_conf[2]
    nb_col = c_conf[3]

    all_batch = []
    c_batch_lst = []
    p_batch_lst = []
    t_batch_lst = []
    y_batch_lst = []
    batch_iter = 0
    for date in All_mat.keys():
        y_date = date
        c_dates = [get_date_str(y_date, - 15 * (i + 1)) for i in range(c_seq_length)]
        p_dates = [get_date_str(y_date, - 96 * 15 * (i + 1)) for i in range(p_seq_length)]
        t_dates = [get_date_str(y_date, - 7 * 15 * (i + 1)) for i in range(t_seq_length)]
        if np.array([i in All_mat for i in [y_date] + c_dates + p_dates + t_dates]).all() == True:
            c_batch_lst.append(np.concatenate([All_mat[k] for k in c_dates], axis=2))
            y_batch_lst.append(All_mat[y_date])
            if p_seq_length != 0:
                p_batch_lst.append(np.concatenate([All_mat[k] for k in p_dates], axis=2))
            if t_seq_length != 0:
                t_batch_lst.append(np.concatenate([All_mat[k] for k in t_dates], axis=2))
            batch_iter += 1
            if batch_iter == batch_size:
                c_batch_mat = np.stack(c_batch_lst, axis=0)
                y_batch_mat = np.stack(y_batch_lst, axis=0)
                if p_batch_lst != []:
                    p_batch_mat = np.stack(p_batch_lst, axis=0)
                else:
                    p_batch_mat = None
                if t_batch_lst != []:
                    t_batch_mat = np.stack(t_batch_lst, axis=0)
                else:
                    t_batch_mat = None
                if c_batch_mat is not None:
                    all_batch.append([c_batch_mat, p_batch_mat, t_batch_mat, y_batch_mat])
                c_batch_lst = []
                p_batch_lst = []
                t_batch_lst = []
                y_batch_lst = []
                batch_iter = 0
            else:
                pass

    return(all_batch)



def get_long_seq(All_mat, c_conf, p_conf, t_conf, long_seq):

    c_seq_length = c_conf[1]
    p_seq_length = p_conf[1]
    t_seq_length = t_conf[1]
    nb_filter = c_conf[0]
    nb_row = c_conf[2]
    nb_col = c_conf[3]

    all_seqs = []
    all_seqs_date = []
    all_date = sorted(list(All_mat.keys()))
    itern = long_seq
    for date in all_date:
        if itern != 0:
            itern -= 1
            continue
        mark = 0
        one_seq_lst = []
        one_seq_date = []
        for j in range(long_seq):
            y_date = get_date_str(date, 15 * j)
            c_dates = [get_date_str(y_date, - 15 * (i + 1)) for i in range(c_seq_length)]
            p_dates = [get_date_str(y_date, - 96 * 15 * (i + 1)) for i in range(p_seq_length)]
            t_dates = [get_date_str(y_date, - 7 * 15 * (i + 1)) for i in range(t_seq_length)]
            if np.array([i in All_mat for i in [y_date] + c_dates + p_dates + t_dates]).all() == True:
                c_data = np.expand_dims(np.concatenate([All_mat[k] for k in c_dates], axis=2), 0)
                y_data = np.expand_dims(All_mat[y_date], 0)
                if p_seq_length != 0:
                    p_data = np.expand_dims(np.concatenate([All_mat[k] for k in p_dates], axis=2), 0)
                else:
                    p_data = None
                if t_seq_length != 0:
                    t_data = np.expand_dims(np.concatenate([All_mat[k] for k in t_dates], axis=2), 0)
                else:
                    t_data = None
                one_seq_lst.append([c_data, p_data, t_data, y_data])
                one_seq_date.append([c_dates, p_dates, t_dates, y_date])
            else:
                mark = 1
                break
        if mark == 0:
            all_seqs.append(one_seq_lst)
            all_seqs_date.append(one_seq_date)
            itern = long_seq

    return(all_seqs, all_seqs_date)


def maxmin_normalize(mat_list):
    max_value = max([np.max(mat) for mat in mat_list if mat != None])
    min_value = min([np.min(mat) for mat in mat_list if mat != None])
    batch = []
    for mat in mat_list:
        if mat != None:
            batch.append((mat - min_value)/(max_value - min_value) * 2 -1)
        else:
            batch.append(None)
    return(batch, max_value)


