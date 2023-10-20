#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       env_from_data
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/07/22
#   Description:    ---
"""基于监督或者非监督的静态数据的环境

任意的监督数据与非监督数学都可以转化为环境的形式。

监督数据：
* 传统数据：{x:tenstor, y: tensor, ...}
* 方法1：Env[{x_1, r_1=Null}, ...{x_n, r_n=Null}, Question] , Agent[y_predict]  , Env[r=loss]
* 方法2：Env[x, question, y] 转化为无监督问题
* 方法3：how about clamp based

自监督数据：
* 传统数据：{x_t}
* 方法1：Env[{x_1, r_1=Null}, ...{x_n, r_n=Null}] , Agent[a_1 = x_2']  , Env[r=], Agent[],R=, ...
* 方法2：上述的predictive - coding 也可以是内部的机制，而不用对外输出。不仅包括temporal pc,也包括 space pc.

"""

import os
import sys

import argparse
import gym

def get_time_sequence(source_tensor, time_split_index):
    tr = [time_split_index] + range(source_tensor.ndim)
    del tr[time_split_index+1]
    new_sample = np.transpose(source_tensor,tr)
    target = [new_sample[i] for i in range(sample.shape[0])]
    return target

class DataBasedEnv(gym.Env):

    def __init__(self):
        pass

    def reset(self, seed = None, return_info = True):
        self.reset_buffer()
        obs, reward, done, info = self.step(None)
        return obs, info

    def get_reward(self, action):
        return 0
    
    def step(self, action):
        """这里action与输入同质, 语言的输入与输出。
        """
        obs, reward, done, info = None, 0, False, None
        if self.t>=len(self.env_buffer):
            done = True
        else:
            obs = self.env_buffer[self.t]
            reward = self.get_reward(action)
            self.t+=1
        return obs, reward, done, info

    def __iter__(self):
        """内建遍历接口"""
        while True:
            done = False
            obs, info = self.reset()
            yield obs
            while not done:
                obs, reward, done, info = self.step(None)
                yield obs

class SimpleSupervisedEnv(DataBasedEnv):
    """简单监督数据{x,y}的env形式
    1. 分为2次数据，先给x缺失y，让agent自己预测，然后给x,y 希望构建形成error去学习 1) x 2)x,y
    2. x,y 一次性给x,y其实也是可以,就是收敛路径不同，结果是一致的。
     2.* 一次性给x,y 在某些error影响x（即y_true影响x）算法中，最后收敛的error会有gap
    """
    def __init__(self, ori_data, sample_split = None):
        self.ori_data = ori_data
        self.sample_split = sample_split

    def reset_buffer(self):
        if sample_split:
            self.env_buffer = self.sample_split(next(self.ori_data))
        else:
            self.env_buffer = [next(self.ori_data)]
        self.t = 0

class SeqEnv(DataBasedEnv):
    """监督数据{x,y}涉及到Seq展开的env形态
    x,y 都可能涉及展开。
    """
    def __init__(self,
                 ori_data,
                 input_name,
                 time_process_def,
                 output_question,
                 output_teacher_reward,
                 max_output_t
                ):
        """
        ori_data: [{name:val}...]
        time_process_def: [ {name1:[time_dim]}, {},... ]
        输出为[  f[ori[name1]], ... ], f 为对原env的time_dim展开
        并且涉及到某些对象的后移。
        这里约束f的形式，更复杂的f通过重载。
        Out: actions by agent
        Reward: reward by teacher(Out)
        """
        self.ori_data = ori_data
        self.input_name = input_name
        self.output_names = self.output_names
        self.input_time_dim = input_time_dim
        # consider one-question first
        self.task_questions = task_questions
        self.output_teacher_reward = output_teacher_reward
        self.max_output_t = max_output_t

    def reset_buffer(self):
        self.ori_sample = self.ori_data.next()
        self.env_buffer = []
        for stage,stage_def in enumerate(self.time_process_def):
            stage_buf = []
            for name,name_process in stage_def:
                if name_process[0]>=0:
                    stage_buf += get_time_seqence(self.ori_sample[name], name_process[0])
                else:
                    stage_buf += self.ori_sample[name]
            stage_len = max([len(x) for x in stage_buf])
            stage_buf = [ [x[i] for x in stage_buf] for i in range(stage_len)]
            self.env_buffer+=stage_buf
        self.env_buffer+=self.task_questions[0]
        self.input_t = len(env_buffer)
        self.t = 0

    def get_reward(self):
        # TODO
        action_tensor = line2tensor(self.history_actions)
        beliefe = None 
        # compare action_tensor and true_tensor

# 基于符号序列的数据的环境
## 通用的设计模式最好是与传统的使用基本的组件，而不是经过转化
## 但目前最方便的方式还是基于转化

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

