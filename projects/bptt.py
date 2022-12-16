#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       bptt
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/08/29
#   Description:    ---
"""Experiemnt about BPTT.
本实验关注传统的RNN（计算图）的训练。

for k,inp in enumerate(input):
    network.process(inp,mode="inference")
    network.process(inp,mode="train")
# 先不考虑与agent统一接口的问题，封装罢了，先构建核心

主要核心包括:
    BPTT
    TBPTT
    RARL
    Eprop
    sparse rarl
    predictional gradient
    (UORO)
"""

import os
import sys

import argparse

class ThisType(object):

    def __init__(self,):
        self


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

