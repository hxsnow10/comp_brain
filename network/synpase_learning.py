#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       synpase
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/20
#   Description:    ---
"""synpases.

learning核心2个模块：
    隐变量（梯度）的传播
    根据梯度更新weights

关于算法的通用性：实际上learning就是对Inference（实际上式各函数）的梯度传播与参数的学习。
通用性来自在实现learning的时候，不需要知道inference函数具体内容。

所以算法本质上一定具备通用性；虽然实现上未必可以直接适配任意的函数,大多数是可以的。

实际上涉及2种函数:前馈x-y; 自反馈x-x

"""

import os
import sys

import argparse
import numpy as np
import logging

from .neuron import Neurons
from .synpase_base import Synpase
from .synpase_inference import SparseSynpase

sys.path.append("..")
from op import *
from util.log import log_info, log_debug
import torch

class TargetBPSynpase(Synpase):
    """使用标准BP来求局部梯度,neuron.error = grad(target, neuron.states)
    可以不仅用在target，任意的局部算子u = f(v)，都可以使用BP计算v.error = BP_f(u.error) 
    哪怕在BPTT中，框架中对任意局部算子进行求导以及梯度传播的功能还是需要的。
    """
    def inference_neuron_states_impact(self):
        pass

    def inference_neuron_error_impact(self):
        logging.debug(sys._getframe().f_lineno, self.__str__())
        rval = []
        target = self.neurons[1].states
        for ne in self.neurons:
            logging.debug("name = {}, requires_grad = {}, shape = {}".format(ne.name, ne.states.requires_grad, ne.states.shape))
            # ne.states.requires_grad = True
            if ne.states.requires_grad:
                ne.states.retain_grad()
        sum_target = target.sum()
        try:
            sum_target.backward()
            # TODO: 可能导致太长的回溯
        except:
            pass
        for ne in self.neurons:
            # make ne's require_grad = True before inference
            # 使用bp需要保证变量之间存在计算依赖关系。
            # mse(y_t,y_true_t)->target_{t+1}
            # grad(target_{t+1},y_t)
            grad = ne.last_states.grad
            if grad is None: grad = 0
            rval.append(grad)
        logging.debug("FUCK error of BP, rval = {}".format(rval))
        return rval

    def learning_synpase_impact(self):
        # 没有参数
        pass

class MseSynpase(TargetBPSynpase):
    """计算target的突触。3个神经元群的突触:x,y,mse目标
    """
   
    target_ne_idx = 2
    def init_more(self):
        self.max_target = 0

    def learning_dynamic(self):
        """无参数，不需要自我学习"""
        pass

    def inference_neuron_states_impact(self):
        # logging.debug("MSE ", sys._getframe().f_lineno, self.__str__())
        for n in self.neurons:
            logging.debug(n)
        # 需要考虑输入y_true缺失与为0的情况，输入 error = 0, 输入 为0,自然计算
        # 本质上不是y_true的问题，而是任意信道区分0与空的问题
        # 物理上不存在在这个问题，0就是空。所以我们应该尽量避免0的编码，比如在word2vec中。
        # 方法一：假设全为0即为空。
        # 方法二：输入额外的信号。
        # 这里我们选择方法一。
        y_true_on = get_sum(self.neurons[1].states)>0.01
        y,y_true = self.neurons[0].states, self.neurons[1].states
        logging.debug("mse xxx".format(self.neurons[0].name, y.shape, self.neurons[1].name, y_true.shape))
        target = 0
        # y.requires_grad = True
        # y_true.requires_grad = True
        if y_true_on:
            target = get_mse(y, y_true)
            # self.max_target = max(self.max_target, target)
        logging.debug("MSE ".format(sys._getframe().f_lineno, self.__str__()))
        for n in self.neurons:
            logging.debug(n)
        logging.debug("finish states dynamic")
        logging.debug("mse get target = {}".format(target))
        return [0,0,target]

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

class LinearErrorBPSynpase(Synpase):
    # TODO: 把这个泛化到任意函数的Infernce上
    
    def inference_neuron_error_impact(self):
        """标准的error反向传播"""
        error1, error2 = self.neurons[0].error, self.neurons[1].error
        impact_2 = 0
        if self.error:
            impact_1 = get_matmul(error2, self.weights, transpose_b=True)
        else:
            impact_1 = 0
        return [impact_1, impact_2]

class GeneralForwardErrorBPSynpase(Synpase):

    def inference_neuron_error_impact(self):
        """
        """
        error1, error2 = self.neurons[0].error, self.neurons[1].error
        impact_2 = 0
        # TODO: 这里也可以考虑直接backwrad()
        d21 = jacobian(self.neurons[1].states, self.neuron[0].last_states)
        if self.error:
            impact_1 = get_matmul(error2, d21, transpose_b=True)
        else:
            impact_1 = 0
        return [impact_1, impact_2]
        

class GradientDescentSynpase(Synpase):
    """基于最大负梯度方向提升参数的突触"""
    
    def learning_synpase_impact(self):
        # TODO: 定义错了, BP有2个环节，一个反向传播梯度，一个是用梯度修正参数。
        states1 = self.neurons[0].out_states
        error2 = self.neurons[1].error
        rval = get_matmul(get_expand_dims(states1,2),get_expand_dims(error2,1))
        rval = torch.sum(rval, dim = 0)
        return rval

class RTRLLearning(Synpase):
    pass

class SnapKSynpase(SparseSynpase):
    def __init__(self, k = 1, * args, ** xargs):
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

# 使用type(Name, baseclasses)可以动态创建class
# BP 类型的连接
from .synpase_inference import LinearSynpase
LinearBPSynpase = type("PSynpase",(LinearSynpase, LinearErrorBPSynpase, GradientDescentSynpase), {})
