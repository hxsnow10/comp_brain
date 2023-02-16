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

import argparse

class MseSynpase(TargetBPSynpase):
    """3个神经元群的突触:x,y,mse目标
    """
   
    target_ne_idx = 2
    def init_more(self):
        self.max_target = 0

    def learning_dynamic(self):
        """无参数，不需要自我学习"""
        pass

    def inference_neuron_states_impact(self):
        print("MSE ", sys._getframe().f_lineno, self.__str__())
        input("start states dynamic"+self.__str__())
        for n in self.neurons:
            print(n)
        # 需要考虑输入y_true缺失与为0的情况，输入 error = 0, 输入 为0,自然计算
        # 本质上不是y_true的问题，而是任意信道区分0与空的问题
        # 物理上不存在在这个问题，0就是空。所以我们应该尽量避免0的编码，比如在word2vec中。
        # 方法一：假设全为0即为空。
        # 方法二：输入额外的信号。
        # 这里我们选择方法一。
        y_true_on = get_sum(self.neurons[1].states)>0.01
        y,y_true = self.neurons[0].states, self.neurons[1].states
        print("mse xxx", self.neurons[0].name, y.shape, self.neurons[1].name, y_true.shape)
        target = 0
        # y.requires_grad = True
        # y_true.requires_grad = True
        if y_true_on:
            target = get_mse(y, y_true)
            # self.max_target = max(self.max_target, target)
        print("MSE ", sys._getframe().f_lineno, self.__str__())
        for n in self.neurons:
            print(n)
        input("finish states dynamic")
        return [0,0,target]

class LinearSynpase(Synpase):
    """标准突触：单向线性动力学的突触
    """
    def __init__(self, neurons, name = None, error=True):
        synpase_inits = np.full([neurons[0].shape[-1], neurons[1].shape[-1]], 0.01)
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.error = error
    
    def inference_neuron_states_impact(self):
        states1, states2 = self.neurons[0].states, self.neurons[1].states
        impact_2 = get_matmul(states1, self.weights)
        return [0, impact_2]

    def inference_neuron_error_impact(self):
        """标准的error反向传播"""
        error1, error2 = self.neurons[0].error, self.neurons[1].error
        impact_2 = 0
        if self.error:
            impact_1 = get_matmul(error2, self.weights, transpose_b=True)
        else:
            impact_1 = 0
        return [impact_1, impact_2]

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
        states1, states2 = self.neurons[0].states, self.neurons[1].states
        impact_2 = self.go_factor * get_matmul(states1, self.weights)
        impact_1 = self.back_factor * \
            get_matmul(states2, self.weights, transpose_b=True)
        return [impact_1, impact_2]

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
        states= self.neurons[0].states
        impact = mm(self.weights,states)
        # impact = torch.sparse.mm(self.weights,states)
        return impact

# 使用type(Name, baseclasses)可以动态创建class
# BP 类型的连接
LinearBPSynpase = type("PSynpase",(LinearSynpase, ErrorBPSynpase), {})

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
    for l in range(h_layer_num):
        x_neurons.append(Neurons(shape = [64,hidden_layer_sizes[l]], rnn_synpase_type = neuron_inter_synpase_type, error=True,
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

