#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       rnn_step
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/11/30
#   Description:    ---
"""rnn step modules.

"""

import os
import sys
import random

import argparse
import torch
from torch import nn

class MyOneStep(torch.nn.Module):
    def __init__(self, layer_size = 50):
        super(MyOneStep, self).__init__()
        self.layer_size = layer_size
        self.lin_hh = nn.Linear(2*layer_size, layer_size)
        self.lin_hy = nn.Linear(layer_size, layer_size)
        self.params = [self.lin_hh.weight, self.lin_hy.weight, self.lin_hh.bias, self.lin_hy.bias]

    def forward(self, inp, state, idx):
        inp = torch.cat([inp, state], 1)
        # out, new_state = full_out.chunk(2, dim=1)
        new_state = self.lin_hh(inp)
        out = self.lin_hy(new_state)
        def get_pr(idx_val, name):
            def pr(*args):
                print("{} doing backward {}".format(name, idx_val))
            return pr
        new_state.register_hook(get_pr(idx, "state_tran"))
        out.register_hook(get_pr(idx,"output"))
        print("doing fw {}".format(idx))
        return out, new_state

class SparseRNN(torch.nn.Module):
    """ 稀疏的简单RNN，通过一个度数分布函数来采样出一个网络。
    常见的度数分布包括：正太分布，均值分布，指数分布、幂律分布等
    """
    def __init__(self, n, degree_sampling):
        edge_indicies = set([])
        for i in range(n):
            k = min(max(int(degree_sampling()),0),n-1)
            edge_indicies.add((i,i))
            js = set([i])
            while len(js)<k:
                j = random.randint(0,n-1)
                if j not in js:
                    js.add(j)
                    edge_indicies.append((i,j))
                    edge_indicies.append((j,i))
                    # 双向连接 保证相互影响
        edge_indicies = list(edge_indicies)
        values = torch.rand(len(edge_indicies))
        self.weight = torch.sparse_coo_tensor(edge_indicies, values)
        self.bias = torch.zeros(n)

    def forward(self, inp):
        return torch.sparse.mm(self.weight,inp)+self.bias

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

