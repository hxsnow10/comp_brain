#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       synpase_inference
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/12/22
#   Description:    ---
"""关于inference的synpase的类。

"""

import os
import sys
import logging

import argparse
from .synpase_base import Synpase
from .neuron import Neurons, ErrorNeurons
import numpy as np

sys.path.append("..")
from op import *
from util.log import log_info, log_debug
import torch

class LinearSynpase(Synpase):
    """标准突触：单向线性动力学的突触
    """
    def __init__(self, neurons, name = None, error=True):
        synpase_inits = np.full([neurons[0].shape[-1], neurons[1].shape[-1]], 0.01)
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.error = error
    
    def inference_neuron_states_impact(self):
        states1, states2 = self.neurons[0].out_states, self.neurons[1].states
        impact_2 = get_matmul(states1, self.weights)
        return [0, impact_2]

class CompDefSynpase(Synpase):
    
    def __init__(self, neurons, comp_def, error = True):
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = None)
        self.error = error
        self.comp_def = comp_def
        # TODO weights
    
    def inference_neuron_states_impact(self):
        states1, states2 = self.neurons[0].out_states, self.neurons[1].states
        impact_2 = self.comp_def(states1)
        return [0, impact_2]

class RecurrentSynpase(Synpase):
    """标准突触：简单的单神经元群的反馈连接突触
    """
    def __init__(self, neurons, name = None, error=True):
        synpase_inits = np.full([neurons[0].shape[-1], neurons[0].shape[-1]], 0.01)
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.error = error

    def inference_neuron_states_impact(self):
        states, states = self.neurons[0].states
        impact_2 = get_matmul(states1, self.weights)
        return [impact_2]

class GRURecurrentSynpase(RecurrentSynpase):
    
    def __init__(self, neurons, name = None, error=True):
        synpase_inits = np.full([neurons[0].shape[-1], neurons[0].shape[-1]], 0.01)
        #TODO 涉及多个参数
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.error = error

    def inference_neuron_states_impact(self):
        states, states = self.neurons[0].states
        impact_2 = get_matmul(states1, self.weights)
        return [impact_2]

class BiLinearSynpase(Synpase):
    """双向线性动力学的突触
    """
    def __init__(self, neurons, name = None, 
                 go_factor=1, back_factor=1):
        synpase_inits = synpase_inits = get_variable(
             [neurons[0].shape[1], neurons[1].shape[1]])
        super(BiLinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)

    def inference_neuron_states_impact(self):
        states1, states2 = self.neurons[0].out_states, self.neurons[1].states
        impact_2 = self.go_factor * get_matmul(states1, self.weights)
        impact_1 = self.back_factor * \
            get_matmul(states2, self.weights, transpose_b=True)
        return [impact_1, impact_2]


class RNNLearningSynpase(Synpase):
    """RNN有一些学习的方法，核心是求一些中间变量。
    
    """


class SparseSynpase(RecurrentSynpase):
    """稀疏反馈突触。
    不具体设定计算，泛化成任意的突触计算，方便应用到BILSTM, GRU上。
    1. 把参数替换成稀疏矩阵
    2. 矩阵乘法根据是否稀疏灵活处理
    初始化一般是根据度分布随机采样连接；但是学习的时候有2种策略1）不修改连接拓扑  2）修改连接拓扑
    """
    def __init__(self, n, degree_sampling, *args, **xargs):
        super().__init__(*args, **xargs)
        edge_indicies = self.sampling_indicies()
        values = torch.rand(len(edge_indicies))
        # 如果是bilstm这种多参数的，则类似地生成多个稀疏矩阵。
        # TODO: for 循环处理下
        self.weights = torch.sparse_coo_tensor(edge_indicies, values)

    def sampling_indicies(self, n, degree_sampling):
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
        return edge_indicies

    def inference_neuron_states_impact(self, inp):
        states= self.neurons[0].out_states
        impact = mm(self.weights,states)
        # impact = torch.sparse.mm(self.weights,states)
        return impact


def general_mlp(
        x,y,
        synpase_type,
        hidden_layer_sizes = [],
        stop_th=0.1,
        neuron_inter_synpase_type = None,
        states2error = 0
    ):
    """通用的MLP"""
    synpases = []
    h_layer_num = len(hidden_layer_sizes)
    x_neurons = [x]
    batch_size = x.shape[0]
    for l in range(h_layer_num):
        x_neurons.append(ErrorNeurons(shape = [batch_size,hidden_layer_sizes[l]], name = "mlp_{}_{}_{}".format(x.name, y.name,l),rnn_synpase_type = neuron_inter_synpase_type, error=True,
                                bp2states_ratio = states2error))
        synpase = synpase_type([x_neurons[l], x_neurons[l+1]])
        synpases.append(synpase)
    synpase = synpase_type([x_neurons[-1], y])
    synpases.append(synpase)
    return synpases

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

