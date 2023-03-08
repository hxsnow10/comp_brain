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

def linear_one_step(weights,out_states):
    impact = get_matmul(out_states[0], weights)
    return [0, impact]

class LinearSynpase(Synpase):
    """标准突触：单向线性动力学的突触
    """
    def __init__(self, neurons, name = None, error=True):
        synpase_inits = np.full([neurons[0].shape[-1], neurons[1].shape[-1]], 0.01)
        TODO: init random
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.error = error
    
    def inference_neuron_states_impact(self):
        impact_2 = get_matmul(self.neurons[0].out_states, self.weights)
        return [0, impact_2]

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

class SimpleRecurrentSynpase(Synpase):
    """标准突触：简单的单神经元群的反馈连接突触
    与一般的synpase并无区别，输入与输出都是同一组神经元的就是recuurrent。
    """
    def __init__(self, neurons, name = None, error=True):
        assert len(neurons)==1
        synpase_inits = np.full([neurons[0].shape[-1], neurons[0].shape[-1]], 0.01)
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.error = error

    def inference_neuron_states_impact(self):
        impact = get_matmul(self.neurons[0].out_states, self.weights)
        return [impact]

class GRURecurrentSynpase(RecurrentSynpase):
    """ TODO： 直接泛化到任意的cell上
    nn.RNNcell
    nn.LSTMCell
    """
    def __init__(self, neurons, name = None, error=True):
        # synpase_inits = np.full([neurons[0].shape[-1], neurons[0].shape[-1]], 0.01)
        # TODO 涉及多个参数
        # super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.gru_cell = torch.nn.GRUCell()

    def inference_neuron_states_impact(self):
        inputs = self.neurons[0].output_states
        states = self.neurons[1].states
        new_states = self.gru_cell(inputs, states)
        # 或者直接neuron leak=1 或者把gru拆了，把reset gates传过来
        # 这里为了简单性，就把neuron leak=1
        # 这样不行，如果某个神经元群与多个输入连接，就做了多次的reset
        # 把forget也作为neuron的某种状态，接受synpase的输入
        return [new_states]

class SparseSynpase(RecurrentSynpase):
    """稀疏反馈突触。

    sparseSimpleRNN
    sparseGRU
    sparseLSTM


    TODO
    不具体设定计算，泛化成任意的突触计算，方便应用到BILSTM, GRU上。
    1. 把参数替换成稀疏矩阵；理想上可行，但又可能稀疏矩阵与一般矩阵用的不用的函数，那就完了。
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

