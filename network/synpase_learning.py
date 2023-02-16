!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       synpase
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/20
#   Description:    ---
"""synpases.

核心2个函数 
    * inference_dynamic: 前向相关的动力学, 对相关neuron的更新
    * learning_dyanmic: 学习相关的动力学，对synpase自身的更新

实际上机器学习由3阶段组成 1）前向 2）梯度反向 3）参数更新。
2从功能上支持learning，不影响inference;但从发生的阶段上看，2与1往往是同时进行的，哪怕目标信号还未产生，在RTRL类算法中也可以存储中间变量dh/dw。
这里把它归结到learning_dynamic，是learning的隐变量的提前更新环节。
"""

import os
import sys

import argparse
import numpy as np

from .neuron import Neurons

sys.path.append("..")
from op import *
import torch

class TargetBPSynpase(Synpase):
    """使用标准BP来求梯度,neuron.error = grad(target, neuron.states)
    """
    def set_traget_neuron_index(self, idx):
        self.target_ne_idx = idx

    def inference_neuron_error_impact(self):
        input("start error dynamic"+self.__str__())
        print(sys._getframe().f_lineno, self.__str__())
        rval = []
        target = self.neurons[self.target_ne_idx].states
        for ne in self.neurons:
            print("name = {}, requires_grad = {}, shape = {}", ne.name, ne.states.requires_grad, ne.states.shape)
            # ne.states.requires_grad = True
            if ne.states.requires_grad:
                ne.states.retain_grad()
        sum_target = target.sum()
        try:
            sum_target.backward()
        except:
            pass
        for ne in self.neurons:
            # make ne's require_grad = True before inference
            grad = ne.states.grad
            if grad is None: grad = 0
            rval.append(grad)
        print("FUCK error of BP", rval)
        return rval

class HebbianSynpase(Synpase):
    """基于hebbian来学习的突触
    """

    def learning_synpase_impact(self):
        """
        shape of states1/2=  [batch, size]
        shape of return = [[size1, size2], [size2, size3], ...]
        """
        states1, states2 = self.neurons[0].states, self.neurons[1].states
        rval = get_matmul(get_expand_dims(states1,2),get_expand_dims(states2,1))
        return rval

class STDPSynpase(Synpase):
    """基于STDP来学习的突触
    here we implement STDP : \delta(W)=lr*pre_state*\devirate(post_state)
    post_devirate_{t+1} =  (post_states_{t}-post_states_{t-1})/(1*dt)

    when state, next_state, we get new 
    here k*dt should could ot be very big.
    """
    def init(self, post_devirate_decay=1):
        self.post_devirate = get_variable(post_neuron.shape())
        self.post_devirate_decay = post_devirate_decay

    def learning_synpase_impact(self):
        # TODO divide timestamp
        self.post_devirate = (1-self.post_devirate_decay)*synpase.neuron_impacts + \
                self.post_devirate_decay*self.post_devirate
        pre_states =  self.neurons[0].states
        synpase_update = self.post_devirate*pre_states
        return synpase_update

class ErrorBPSynpase(Synpase):
    """基于反向传播来传播梯度的突触"""
    
    def learning_synpase_impact(self):
        states1 = self.neurons[0].out_states
        # 按公式是要用out_states
        error2 = self.neurons[1].error
        rval = get_matmul(get_expand_dims(states1,2),get_expand_dims(error2,1))
        rval = torch.sum(rval, dim = 0)
        return rval

class RTRLLearning(Synpase):
    pass

class SnapKSynpase(SparseSynpase):
    def __init__(self, k = 1, * args, ** xargs)
        # 这里是考虑泛化的形式one_step_module
        # Snap 把不同rnn的相关函数实现切割出来，让结构突触自己去定义， sparse(朴素rnn, gru, bilstm)。
        # 问题是不同的计算结构对稀疏产生了挑战
        # 这里我们假设：one_step_module(h_t)->h_{t+1} 如果使用了多个W[n*n]，他们都保持一样的稀疏性
        # 即计算在本地视角分解后，保持类似的稀疏性。如果不是那也是稀疏RNN内部的问题。
        self.k = k
        self.one_step_module = one_step_module
        self.build_state_weight_grad() 
        super().__init__(*args, **xargs)
    
    def build_state_weight_grad(self):
        """根据链接拓扑获取dh/dw的稀疏矩阵
        这里先假设k=1
        """
        self.grad_of_state_param = []
        def filter_sparse_indices(a,select_indicies):
            indicies = a.indices().permute(1,0)
            values = s.values()
            set_select_indicies = set(select_indicies.permute(1,0))
            new_indicies, values = [], []
            for k,ind in enumerate(indicies):
                if ind in set_select_indicies:
                    new_indicies.append(ind)
                    new_values.append(values[k])
            new_indicies.permute(1,0)
            rval = torch.sparse_coo_tensor(new_indicies, values)
            return rval
        for k,param in enumerate(self.one_step_module.params):
            state_param_connect_indices = self.one_step_module.state_param_connect_indices[k]
            values = torch.zeros(state_param_connect_indices.shape[1])
            grad_of_state_param_this = torch.sparse_coo_tensor(state_param_connect_indices, values, [state_size, * param.shape])
            self.grad_of_state_param.append(grad_of_state_param_this)

    def inference_neuron_error_impact(self):
        # update forward gradient
        neurons = self.neurons[0]
        state = neurons.last_state
        new_state = neurons.state
        grad_of_state_param_step = torch.autograd.grad(new_state, self.one_step_module.params, retain_graph=True)
        grad_of_state_state = torch.autograd.grad(new_state, state, retain_graph=True)
        new_grad_of_state_param = []
        for k,grad in enumerate(grad_of_state_param):
            grad = grad_of_state_param_step[i] + torch.sparse.mm(grad_of_state_state,self.grad_of_state_param[i])
            grad = filter_sparse_indices(grad, self.one_step_module.state_param_connect_indices[k])
            new_grad_of_state_param.append(grad)
        self.grad_of_state_param = new_grad_of_state_param
        # 数学上是替换。但在动力学中，是否使用某种累加来收敛更好呢？
        # 这里有许多考虑。总的来说：float网络，使用非动力学更高效。
        # 通过synpase内部状态来更新，不需要存储在neuron-error

        # update inside loss_state_grad
        # RNN的dT/dh通过一次反向传播完成。
        # grad_of_loss_state = torch.autograd.grad(loss,new_state, retain_graph=True)
        # 假设网络是前馈的，进行反向传播并进行累加。
        # 按理是计算一次比较高效。生物系统计算多次，是因为它的效率与累加值成正比，可以拆分，然后进行稀疏的传播。

        return None

    def learning_synpase_impact(self): 
        # compute step gradient
        # 这里的grad是外部以及内部共同形成的。
        for i in range(len(self.grad_of_state_param)):
            self.one_step_module.params[i]+=self.error*self.grad_of_state_param[i]

