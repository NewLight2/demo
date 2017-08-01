#coding:utf-8
'''
读取网格人数统计，进行网格归并且作图
python3服务器版本
涂迅 2017/3/14
'''
import numpy as np
import tensorflow as tf
from data_process.detect_abnormal import abnormal
import sys
import os
import re

date_patten = re.compile("\d+")

def read_data(file_name, left_bottom_grid, top_right_grid):
    #读取数据源，格式为"网格，人数"，输出矩阵，经度刻度，纬度刻度
    source_data = []
    with open(file_name) as f:
        for line in f:
            date, rest = line.strip().split('\t')
            cols = rest.split('|')
            for col in cols:
                pair = col.split(',')
                lonlat_str = pair[0]
                num = int(pair[1])
                lon = int(lonlat_str[0:4])
                lat = int(lonlat_str[4:])
                source_data.append([date, lon, lat, num])

    low_lon = left_bottom_grid[0]
    low_lat = left_bottom_grid[1]
    high_lon = top_right_grid[0]
    high_lat = top_right_grid[1]

    length_lon = high_lon - low_lon + 1
    length_lat = high_lat - low_lat + 1

    xlab = [i for i in range(low_lon, high_lon + 1)]
    ylab = [i for i in range(low_lat, high_lat + 1)]
    ylab.reverse()

    All_Mat = {}
    for data in source_data:
        if data[0] not in All_Mat:
            All_Mat[data[0]] = np.zeros([length_lon, length_lat], dtype='int32')
        else:
            continue

    for data in source_data:
        lat_pos = high_lat - data[2]
        lon_pos = data[1] - low_lon
        All_Mat[data[0]][lon_pos, lat_pos] = int(data[3])

    Amat = []
    day_sum = []
    for date in All_Mat:
        Amat.append([date, All_Mat[date]])
        day_sum.append([date, np.sum(All_Mat[date])])
    Amat.sort()
    day_sum.sort()
    return(Amat, day_sum, xlab, ylab)


def filter_data(mat_seq):
    '''
    对每日的数据进行过滤，去掉有问题的数据
    '''
    date_today = mat_seq[len(mat_seq) // 2][0][0:8]
    new_mat_seq = []
    for data in mat_seq:
        if data[0][0:8] == date_today:
            new_mat_seq.append(data)
    return(new_mat_seq)


def merge_in_out_flow(Amat_in, Amat_out, Amat_all):
    Amat_merge = []
    for pair_in in Amat_in:
        [date_in, mat_in] = pair_in
        for pair_out in Amat_out:
            [date_out, mat_out] = pair_out
            k = 0
            for pair_all in Amat_all:
                [date_all, mat_all] = pair_all
                if date_out == date_in and date_all == date_in:
                    mat_merge = np.stack((mat_in, mat_out, mat_all), axis=2)
                    Amat_merge.append([date_in, mat_merge])
                    k = 1
                    break
            if k == 1:
                break
    return(Amat_merge)


def get_all_output(file_in_dir, file_out_dir, file_all_dir, left_bottom_grid, top_right_grid):
    filenames_in = os.listdir(file_in_dir)
    filenames_out = os.listdir(file_out_dir)
    filenames_all = os.listdir(file_all_dir)
    filenames_in.sort()
    filenames_out.sort()
    filenames_all.sort()
    All_result = []
    All_in_sum = []
    All_out_sum = []
    All_all_sum = []
    for file_in in filenames_in:
        file_in_day = date_patten.search(file_in).group()
        Amat_in, day_in_sum, _, _ = read_data(os.path.join(file_in_dir, file_in), left_bottom_grid,top_right_grid)
        for file_out in filenames_out:
            file_out_day = date_patten.search(file_out).group()
            for file_all in filenames_all:
                file_all_day = date_patten.search(file_all).group()
                if file_in_day == file_out_day and file_in_day == file_all_day:
                    Amat_in, day_in_sum, _, _ = read_data(os.path.join(file_in_dir, file_in), left_bottom_grid,top_right_grid)
                    Amat_out, day_out_sum, _, _ = read_data(os.path.join(file_out_dir, file_out), left_bottom_grid, top_right_grid)
                    Amat_all, day_all_sum, _, _ = read_data(os.path.join(file_all_dir, file_all), left_bottom_grid, top_right_grid)
                    All_in_sum.extend(day_in_sum)
                    All_out_sum.extend(day_out_sum)
                    All_all_sum.extend(day_all_sum)
                    Amat_merge = merge_in_out_flow(Amat_in, Amat_out, Amat_all)
                    All_result.extend(Amat_merge)

    ab_in = abnormal(15, 20000, 4)
    ab_in.read_data(All_in_sum)
    abnormal_in = ab_in.gather_abnormal()
    ab_out = abnormal(15, 20000, 4)
    ab_out.read_data(All_out_sum)
    abnormal_out = ab_in.gather_abnormal()
    abnormal_set = abnormal_in | abnormal_out
    filter_result = {}
    max_num = 0
    for seq in All_result:
        if seq[0] not in abnormal_set:
            filter_result[seq[0]] = seq[1]
            max_seq = np.max(seq[1])
            if max_seq > max_num:
                max_num = max_seq

    return(filter_result, max_num)


def write_binary(file_dir, output_filename, left_bottom_grid, top_right_grid):
    filenames = os.listdir(file_dir)
    filenames.sort()
    writer = tf.python_io.TFRecordWriter(output_filename)
    for filename in filenames:
        Amat, _, _ = read_data(os.path.join(file_dir, filename), left_bottom_grid, top_right_grid)
        new_mat_seq = filter_data(Amat)
        for pair in new_mat_seq:
            date = pair[0].encode('utf-8')
            mat = pair[1]
            mat_raw = mat.tostring()
            example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'date': tf.train.Feature(
                                bytes_list = tf.train.BytesList(value=[date])),
                            'mat': tf.train.Feature(
                                bytes_list = tf.train.BytesList(value=[mat_raw]))
                            }
                        )
                    )
            serialized = example.SerializeToString()
            writer.write(serialized)
    writer.close()


if __name__ == '__main__':
    file_dir = sys.argv[1]
    output_filename = sys.argv[2]
    left_bottom_grid = [526, 322]
    top_right_grid = [626, 392]
    write_binary(file_dir, output_filename, left_bottom_grid, top_right_grid)
