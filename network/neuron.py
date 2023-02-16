#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       neuron
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         16/04/2022
#   Description:    ---
"""Neuron Class

Neuron，ForwardNeuron， SpikingNeuron。
"""
import argparse
import os
import sys

import numpy as np

sys.path.append("..")
from op import *

all_neuron_names = set([])

"""
区别于基于计算图的编程方式:y = f(x)，
这里使用基于神经元动力学的编程方式:link(x,y,f) 或者 y.add(f(x))
动力学方式结构上更清晰，属于RNN，契合存算一体。
"""

class Neurons(object):
    """
    神经元类，包括状态变量、激发函数、历史影响（泄露比例或者内部rnn）等要素
    """
    def __init__(self, states_init = 0.00 ,
                 shape=None,
                 name = None,
                 requires_grad = False,
                 activation = "relu",
                 leak = 0.1,
                 self_dynamics = None,
                 rnn_synpase_type = None,
                 visiable = False,
                 error = False,
                 extend_vars = []):
        """
        ## Args
        name: name string
        shape: 一般是[batch_size, neuron_size]
        init_states: 初始值
        leak: 默认值 TODO: 进一步 leak也可以是一个函数.
        TODO: add LTF activation
        self_dynamics:
            * "rnn_train"
            * "leak_train"
            * "leak_static"
            self-dynamics实际上是synpase, 那么work的时候也要触发这里的动力学,更复杂的动力学还是通过synpase去管理?
        visiable: only visiable neurons could be set manualy
        error: with error node
        extend_vars: more neuron vars
        TODO: add neuron -error dynamics
        """
        if shape: self.states_init = np.full(shape,states_init)
        else:self.states_init = states_init
        self.shape = self.states_init.shape
        if name == None:
            name = "neruon"
        if name in all_neuron_names:
            for i in range(1000):
                if name+"_"+str(i) not in all_neuron_names:
                    name = name+"_"+str(i)
                    break
        all_neuron_names.add(name)
        self.name = name
        self.requires_grad = requires_grad
        self.states = get_variable(self.states_init, name=name, requires_grad = requires_grad)
        self.leak = get_variable(leak_init)
        self.synpase_impact = []
        self.clamped = False
        self.activation = get_activation(activation)
        self.out_states = self.activation(self.states)
        self.sum_states_impacts = 0

        # 内置自反馈RNN突触
        self.inter_rnn_synpase = None
        if rnn_synpase_type!=None:
            self.inter_rnn_synpase = rnn_synpase_type([self, self])
        
        self.init_more()
        print("new created", self.__str__())

    def __str__(self):
        s = "neuron, name = {} , shape = {}, dtype = {}, requires_grad = {}, id  = {}".format(
            self.name, self.states.shape, self.states.dtype, self.states.requires_grad, id(self.states))
        return s

    def init_states(self):
        self.states = get_variable(self.states_init, name=self.name, requires_grad = self.requires_grad)

    def init_more(self):
        pass

    def get_val(self):
        """return value of states"""

    def set_val(self, val, clamped = False):
        """set value of states of visiable neurons"""
        # TODO: how to set without create an replace
        # assign_add(self.states, val)
        if val.shape!=self.states.shape:
            raise ValueError("neuron set val , dim mismatch {} -> {}".format(val.shape, self.states.shape))
        self.states = val
        self.clamped = clamped
    
    def add_error(self, val, clampled):
        """外部直接提供error信号
        按道理不是提供target信号吗？
        可以使用TF内部的能力，target=F(states), error = dtarget/dstates
        """
        assign_add(self.error, val)

    def add_impact(self, impact):
        print("add impact", impact)
        if not impact: return None
        if self.clamped: return None
            self.sum_states_impacts = self.sum_states_impacts + impact[0]
        self.sum_states_impacts = self.sum_states_impacts + impact
        return None

    def dynamic(self):
        self.last_states = self.states
        print("before forward", self.__str__())
        input("start forward"+self.__str__())
        if self.inter_rnn_synpase:
            self.inter_rnn_synpase.neuron_dynamic()
        self.states = (1-self.leak)*self.states+self.sum_states_impacts
        self.states = self.states + self.update_states_by_error()
        self.out_states = self.activation(self.states)
        self.sum_states_impacts = 0
    
        print("after forward", self.__str__())

class ErrorNeuons(Neurons):
    """
    包含error项的neuron, error项表示neuron表征距离期望状态的偏离或者梯度。
    学习系统的神经元往往包括以下几项error：
        * target关于神经元状态的梯度：1）通过简单的反向传播来实现 2）是否可能涉及对未来目标产生的梯度的预测
        * 神经元状态关于(连接)参数的梯度：1）时间层面，需要累积对参数的历史影响  2）空间层面，是否需要考虑较远的路径的连接
    在前向网络BP中，第二项直接本地求导即可；
    在反馈网络BP中，前向梯度的算法中，第二项是关键。朴素的算法需要存储N*M的梯度矩阵，N是神经元状态数，M是参数量；优化的算法存储M的状态数。
    """

    def __init__(self, bp2states_ratio = 0， error_leak = 0.1, *args, **xargs):
        # 内置Error项
        # 这里目前只考虑前向网络的梯度 TODO
        super().__init__(*args, **xargs)
        self.error = get_zeros_like(self.states) 
        self.error_leak = error_leak
        # error_implact 通过突触去修改
        self.sum_error_impacts = 0
        self.bp2states_ratio = bp2states_ratio
    
    def update_states_by_error(self):
        return -self.bp2states_ratio*self.error

    def dynamic(self):
        super().dynamic()
        self.error = (1-self.error_leak)*self.error+self.sum_error_impacts
        # TODO: add tbptt error
        # Consider different activation
        # self.out_error = self.activation(self.error)
        self.sum_error_impacts = 0
    
    def add_error_impact(self, impact):
        print("add impact", impact)
        if not impact: return None
        if self.clamped: return None
        self.sum_error_impacts = self.sum_error_impacts + impact
        return None

class ForwardNeurons(Neurons):
    """无历史因素"""
    def init_more(self):
        self.leak = 1

class SpikingNeurons(Neurons):
    """输入输出为spike(0/1值)的神经元类。"""

    def __init__(self, init_states = 0 ,
                 shape=None,
                 name = None,
                 activation = "relu",
                 leak_init = 0.1,
                 self_dynamics = None,
                 rnn_synpase_type = None,
                 visiable = False,
                 error = False,
                 bp2states_ratio=0,
                 trigger_th = 10,
                 trigger_reset = 5,
                 surrogate = True
                ):
        super(SpikingNeurons, self).__init__(init_states, shape, name, activation, leak_init, self_dynamics, rnn_synpase_type,
                                            visiable, error, bp2states_ratio)
        self.trigger_th = trigger_th
        self.trigger_reset = trigger_reset
        self.surrogate = surrogate

    def dynamic(self):
        # 可变的地方1：在这个next的式子
        self.states = (1-self.leak)*self.states+self.sum_impacts+self.leak*self.trigger_reset
        # TODO: why not add nonlinear: self.states = self.activation(self.states)
        self.out_states = cast_type(self.states>self.trigger, "int32")
        # 可变的地方2：out_states的值域 self.out_states = cast_type(self.states/self.trigger, "int32")
        # 如果允许发超过1的，下式子要变
        # what about, trigeer many times in a time-step
        self.states = (1-self.out_states)*self.states+self.out_states*self.trigger_reset

if __name__ == "__main__":
    a = Neurons(0.01, [100, 100])
    a = ForwardNeurons(0.01, [100, 100])
    a = SpikingNeurons(0.01, [100, 100])
