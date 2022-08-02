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

def get_time_sequence(source_tensor, time_split_index):
    tr = [time_split_index] + range(source_tensor.ndim)
    del tr[time_split_index+1]
    new_sample = np.transpose(source_tensor,tr)
    target = [new_sample[i] for i in range(sample.shape[0])]
    return target

class SeqSelfSupervisedEnv(object):

    def __init__(self,
                 ori_data,
                 input_name,
                 input_time_dim):
        self.ori_data = ori_data
        self.input_name = input_name
        self.input_time_dim = input_time_dim

    def reset(self):
        self.ori_sample = self.ori_data.next()
        self.env_buffer = get_time_seqence(self.ori_sample[self.input_name], self.input_time_dim)
        self.input_t = len(env_buffer)
        self.t = 0
    
    def step(self, action):
        obs, reward, done, info = None, 0, False, None
        if self.t>self.max_output_t+self.input_t:
            done = True
        else:
            obs = self.env_input[self.t]
            reward = self.loss(obs, action)
            self.t+=1
        return obs, reward, done, info

# 基于符号序列的数据的环境
## 通用的设计模式最好是与传统的使用基本的组件，而不是经过转化
## 但目前最方便的方式还是基于转化
class SeqSupervisedEnv(object):

    def __init__(self,
                 ori_data,
                 input_name,
                 output_names,
                 input_time_dim,
                 output_questions,
                 output_teacher_reward,
                 max_output_t,
                 line2tensor
                ):
        """
        In: input_tran
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

    def reset(self):
        self.ori_sample = self.ori_data.next()
        self.env_input = []
        self.env_buffer = get_time_seqence(self.ori_sample[self.input_name], self.input_time_dim)
        self.env_buffer+=self.task_questions[0]
        self.input_t = len(env_buffer)
        self.t = 0

    def step(self, action):
        """这里action与输入同质, 语言的输入与输出。
        """
        obs, reward, done, info = None, 0, False, None
        if self.t>self.max_output_t+self.input_t:
            done = True
        else:
            obs = self.env_input[self.t]
            if self.t>self.input_t:
                self.history_actions+=action
                reward = self.get_reward()
            self.t+=1
        return obs, reward, done, info

    def get_reward(self):
        action_tensor = line2tensor(self.history_actions)
        beliefe = 
        compare action_tensor and true_tensor

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

