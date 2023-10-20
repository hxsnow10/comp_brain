#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       receall_task_gen
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/07/20
#   Description:    ---
"""script to autoly generate short-memory recall task data.

"""

import os
import sys

import argparse
import random
from enum import Enum

class QType(Enum):
    #recall_type
    whether_one_occur = 0
    whether_one_semantic_same_occur = 1;
    whether_one_next_one = 2;
    whether_one_pre_one = 3;

    recall_next_of_item = 4;
    recall_pre_of_item = 5;
    recall_full_list = 6;
    recall_full_set = 7;
    
    count_one = 8;
    count_all_num = 9;
    count_all_uniq_num =   10;

class QTemplate(object):    
    templates = [None for _ in range(12)]
    templates[0] = "whether {} occur?"
    templates[1] = "whether some item similar to {} occur?"
    templates[2] = "whether {} occur next {}?"
    templates[3] = "whether {} occur before {}?"
    templates[4] = "what occur next {}?"
    templates[5] = "what occur before {}?"
    templates[6] = "recall the items with order?"
    templates[7] = "recall the items?"
    templates[8] = "how many time of {} occur?"
    templates[9] = "how many items occur?"
    templates[10] = "how many different items occur?"

def gen_qu(question_type,  item0, item1):
    """ return a sentence question"""
    sent = "Question:"+QTemplate.templates[question_type.value].format(item0, item1)
    #TODO: consider word level
    return sent

def gen_answer(seq, question_type,  item0, item1):
    pass

def sampling(a):
    f = random.random()
    inx = int(f*len(a))
    return a[inx]

def generate_stm_sample(
                                   item_size = 1000,
                                   time_count = 10000,
                                   item_occur_count = 10,
                                   item_cocur_duration = 1,
                                   question_occur_begin = False,
                                   question_types = [],
                                number_black = 5
        ):
    """生成一个sample数据，list(<t,char>)
    """
    # TODO 有没有可能把item设计成一些尽量无关的对象呢
    items = list(range(1,item_size+1))
    random.shuffle(items)
    occur_prob = item_occur_count*1.0/time_count
    t, seq = 0, "Sequence:"
    while t<time_count:
        f= random.random()
        if f<occur_prob:
            item = sampling(items)
            seq+=item_cocur_duration*str(item)
            seq+=number_black*" "
            t+=(item_cocur_duration+number_black)
        else:
            seq+=" "
            t+=1
    rval = seq
    for qtype in question_types:
        item1, item2 = sampling(items), sampling(items) 
        q = gen_qu(qtype, item1, item2)
        a = gen_answer(seq, qtype, item1, item2)
        if question_occur_begin:
            rval = q + a + rval 
        else:
            rval = rval+q+a
    return rval

def generate_stm_task(total_num, data_path):
    """
    需要考虑一个数据集中每个样本参数的分布，从分布采样生成
    因为要泛化STM，所以item_type也不局限，枚举item_type
    size为[4，8，16，32，..]
    item_occur_count = []
    item_cocur_duration=[1, 10, 50] 固定几个值
    question_type 均匀分布采样
    question_num [1,2,5,10] 均匀分布采样

    只要采样的足够多，就会出现不同难度的task。
    给一个难度值。
    """
    item_size_list = list(range(4,32)) # TODO: 考虑logN均匀分布
    item_occur_count_list = list(range(4,32))
    time_count_list = list(range(8,640))
    item_cocur_duration_list = [1]
    question_type_list = [ a for a in QType ]
    question_num_list = list(range(1,10))
    oo = open(data_path,'w')
    for k in range(total_num):
        item_size = sampling( item_size_list)
        item_occur_count = sampling(item_occur_count_list)
        time_count = max(sampling(time_count_list),item_occur_count)
        item_cocur_duration = sampling(item_cocur_duration_list)
        question_num = sampling(question_num_list)
        question_types = [sampling(question_type_list) for _ in range(question_num)]
        sample = generate_stm_sample(item_size, time_count, item_occur_count, item_cocur_duration,
                                     False, question_types)
        oo.write(sample+'\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--total_num", default=10)
    parser.add_argument("-o", "--out_path", default="recall_data.txt")
    args = parser.parse_args()
    generate_stm_task(args.total_num, args.out_path)

