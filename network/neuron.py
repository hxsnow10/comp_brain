#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       neuron
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         16/04/2022
#   Description:    ---
"""Dynamic Network Kernel Class Design.

We Induce the design from following main characteristics/assumes:
    1、Structure inclues (states, weights) , or say (neurons, synpases)
    2、Dynamics include neuron-dynamics and synpase-dynamics, or predict and learning .
    3、All Dynamics within structure all local (except some phase control signal)

So we can easily deduce an interface which's similar to brian2.

Neurons: states
    * tensor
    * neurons may with self-dynamics, like leak or rnn-dynamics
    * high-order neurons-network include visiable-neurons, inside structure, inside dynamics

Synpase: link between neurons neurons
    * tensor
    * neuron dynamics: define computation to change postsynpase neuron
    * synpase dynamics: define computation to change synpase self
    * time: define time to compute. default equal to timestep of model
    * high order synpase-network include above too.

Uasge:
    a = Neurons()
    b = Neurons()
    link = MlpSynpase(a,b)
    network = NetWork([a,b], [link], ...)

    data = ...

    control = ...

    network.fit(data, control)
"""

import argparse
import os
import sys

import numpy as np

sys.path.append("..")
from op import *

all_neuron_names = set([])

class Neurons(object):

    def __init__(self, states_init = 0.00 ,
                 shape=None,
                 name = None,
                 requires_grad = False,
                 activation = "relu",
                 leak_init = 0.1,
                 self_dynamics = None,
                 rnn_synpase_type = None,
                 visiable = False,
                 error = False,
                 bp2states_ratio=0,
                 extend_vars = []):
        """
        ## Args
        name: name string
        shape: 一般是[batch_size, neuron_size]
        init_states: 初始值
        leak_init: 默认值  TODO: 进一步 leak也可以是一个函数
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
        # TODO: consider use tensor name
        if name == None:
            name = "neruon_"+str(shape)
        if name in all_neuron_names:
            for i in range(1000):
                if name+"_"+str(i) not in all_neuron_names:
                    name = name+"_"+str(i)
                    break
        all_neuron_names.add(name)
        self.name = name
        # 实际上torch输入本身就是tensor
        self.requires_grad = requires_grad
        self.states = get_variable(self.states_init, name=name, requires_grad = requires_grad)
        self.leak = get_variable(leak_init)
        self.synpase_impact = []
        self.clamped = False
        self.activation = get_activation(activation)
        self.out_states = self.activation(self.states)
        self.error = get_zeros_like(self.states) 
        self.all_states = [ self.states, self.error]
        self.bp2states_ratio = bp2states_ratio
        self.inter_rnn_synpase = None
        if rnn_synpase_type!=None:
            self.inter_rnn_synpase = rnn_synpase_type([self, self])
        self.init_more()
        self.sum_states_impacts = 0
        self.sum_error_impacts = 0
        self.error_leak = 0.1 # TODO: INIT
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
        if self.clamped: return None
        if type(impact)==type([1,2,3]):
            self.sum_states_impacts = self.sum_states_impacts + impact[0]
            self.sum_error_impacts = self.sum_error_impacts + impact[1]
        else:
            self.sum_states_impacts = self.sum_states_impacts + impact
        return None

    def update_states_by_error(self):
        return -self.bp2states_ratio*self.error

    def forward(self):
        """
        得先把所有神经元的下个状态算出来，保证仿真不产生时间不合法的依赖
        """
        print("before forward", self.__str__())
        input("start forward"+self.__str__())
        if self.inter_rnn_synpase:
            self.inter_rnn_synpase.neuron_dynamic()
        self.states = (1-self.leak)*self.states+self.sum_states_impacts
        self.states = self.states + self.update_states_by_error()
        self.out_states = self.activation(self.states)
        self.sum_states_impacts = 0

        self.error = (1-self.error_leak)*self.error+self.sum_error_impacts
        # TODO: add tbptt error
        # Consider different activation
        self.out_error = self.activation(self.error)
        self.sum_error_impacts = 0
        print("after forward", self.__str__())

class ForwardNeurons(Neurons):
    """无历史因素"""
    def init_more(self):
        self.leak_init = 1

class SpikingNeurons(Neurons):

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

    def forward(self):
        """
        得先把所有神经元的下个状态算出来，保证仿真不产生时间不合法的依赖
        """
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
