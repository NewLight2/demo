#coding:utf-8
import numpy as np
from scipy import stats
import datetime
import sys


class abnormal():
    #计算异常时间段集合

    def __init__(self, interval, low_bound, z_bound):
        #interval:时间间隔,low_bound:最小人数界限,z_bound:zscore阈值
        self.count_abnormal = set()
        self.seq_collect_left = {}
        self.seq_collect_right = {}
        self.seq_white = set()
        self.interval = interval
        self.low_bound = low_bound
        self.z_bound = z_bound


    def time_continue(self, date_str1, date_str2, time_interval):
        #判断时间是否连续
        date1 = datetime.datetime.strptime(date_str1, '%Y%m%d%H%M')
        date2 = datetime.datetime.strptime(date_str2, '%Y%m%d%H%M')
        if (date2 - date1).seconds//60 == time_interval:
            return(True)
        else:
            return(False)


    def median(self, lst):
        #求中位数
        lst = sorted(lst)
        if len(lst) % 2 == 1:
            return(lst[len(lst)//2])
        else:
            return((lst[len(lst)//2-1]+lst[len(lst)//2])/2.0)


    def MAD_zscore(self, list_pair):
        #求中位数版本的zscore
        d = [i[1] for i in list_pair]
        med = self.median(d)
        MAD = self.median([abs(j - med) for j in d])
        zscore_pair = [[k[0], 0.6745 * (k[1] - med) / MAD] for k in list_pair]
        return(zscore_pair)


    def read_data(self, lst):

        def add_collect(collect, time, seq):
            if time in collect:
                collect[time].append(seq)
            else:
                collect[time] = [seq]

        for seq in range(len(lst)):
            if seq == 0 or seq == len(lst) - 1:
                continue
            dat, num = lst[seq]
            time = dat[8:]
            if num < self.low_bound:
                #判断小于最小界限的异常值
                self.count_abnormal.add(dat)
            left_continue = self.time_continue(lst[seq-1][0], lst[seq][0], self.interval)
            right_continue = self.time_continue(lst[seq][0], lst[seq+1][0], self.interval)
            if left_continue and right_continue:
                if (lst[seq-1][1] - lst[seq][1]) * (lst[seq][1] - lst[seq+1][1]) > 0:
                    #若连贯，加入序列判断白名单
                    self.seq_white.add(dat)
            if left_continue:
                #加入左连续字典
                left_delta = abs(lst[seq-1][1] - lst[seq][1])
                add_collect(self.seq_collect_left, time, [dat, left_delta])
            if right_continue:
                #加入右连续字典
                right_delta = abs(lst[seq][1] - lst[seq+1][1])
                add_collect(self.seq_collect_right, time, [dat, right_delta])


    def get_abnormal_set(self, collect, bound):
        abnormal_set = set()
        for t in collect:
            zscore_pair = self.MAD_zscore(collect[t])
            for p in zscore_pair:
                if abs(p[1]) > bound:
                    abnormal_set.add(p[0])
        return(abnormal_set)


    def gather_abnormal(self):
        #计算最终异常结果
        left_seq_abnormal = self.get_abnormal_set(self.seq_collect_left, self.z_bound)
        right_seq_abnormal = self.get_abnormal_set(self.seq_collect_right, self.z_bound)
        gather_set = self.count_abnormal | ((left_seq_abnormal & right_seq_abnormal) \
                - self.seq_white)
        return(gather_set)


if __name__ == '__main__':
    filename = sys.argv[1]
    lst = []
    with open(filename) as f:
        for line in f:
            dat, num = line.strip().split(',')
            num = int(num)
            lst.append([dat, num])
    ab = abnormal(15, 20000, 4)
    ab.read_data(lst)
    result_set = ab.gather_abnormal()
    for i in result_set:
        print(i)
