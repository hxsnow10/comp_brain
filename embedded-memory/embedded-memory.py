#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       embeded-memory
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         17/04/2022
#   Description:    ---
"""核心的目的是：实现短期记忆与长期记忆的融洽架构。

短期/工作记忆的特点是：有限性，注意力，可操作，不限于内容，流式
具体理论参考 wolai:记忆模型的思考。
目前的理论难点在于：生物上注意怎么实现准入、全局的分配的。（竞争性网络？）

embeded-model: 
* Nets: Various Long-Term Memory/System(e.g episodic memory， semantic memory，vision-system, nlp-system)
** 数学形式：RNN与MLP的结合，也包括CNN,ATTENTION; 还是动力学网络的形式。
** 具体子系统：
*** 感觉系统(包括低级与高级-语义等；语音，图像，。。)
*** 感觉想象系统
*** 语言自述系统
*** 强化学习与决策
** 应当注意到不同的子系统存在独立的短期记忆或者注意力控制器
** 应当注意到不同的系统存在结构的区别，比如说事件性记忆的稀疏性与语义网络的重叠性；
* Centeral Excetive Process:  for attention management, attentined states pointer management, rehearasal
** 所以这里的核心是attention的理论。
"""

import os
import sys

import argparse

class AbstractLimitedAttention(object):

    def __init__(self, size, th):
        self.size = size
        self.pointer_list = []
        self.th = th

    def work(self, system):
        while True:
            for pointer in self.pointer_list:
                # do forget
            # 获取下一个注意到的激发态
            under_attention_activations, attention_activations = system.next_step(th) # 变化是不是也受attention影响呢？
            for states in attention_activations:
                # 注意力只能关注到1-2个对象
                pointer,state_size = get_pointer(states)
                pointer_list.insert(pointer,state_size)
                # pointer_list不仅要存储一个吸引子的指针，也要存储序！
                pointer_list.resize(self.size)




    def set_size(self, size):
        self.size = size


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

