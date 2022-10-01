#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       temporal_predicitive_coding
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/20
#   Description:    ---
"""one line of summary

description
"""

import os
import sys

import argparse

# temporal predictive coding
# 本身就是一个复杂神经群
# 从一次输入的最优解释，拓展到对历史、当前、未来的最优解释、预测。
# * 一个rnn预测未来，未来是多步伐的
# * 一个rnn保存历史
# * 直接保存历史
# 附近的神经元群对此有连接

# [neuron, neighbors]->[predict_next] predict_next->last_predict, neuron,last_predict->targetNeuron
class ThisType(object):

    def __init__(self,):
        self


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

