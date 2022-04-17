#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       base
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         16/04/2022
#   Description:    ---
"""Dynamic Network Kernel Class Design.

We Induce the design from following main characteristics/assumes:
    1、Structure inclues (states, weights) , or say (neurons, synpases)
    2、Dynamics include neuron-dynamics and synpase-dynamics, or predict and learning .
    3、All Dynamics within structure all local (except some phase control signal)

So we can easily deduce an anterface which's similar to brian2.

Neurons: states
    tf.tensor
    all states should a inside tensor,init once then update by dynamics,
        only visiable(input,output) neurons could be write.

Synpase: kink between neurons and neurons
    which neurons to link
    neuron dynamics: a tf graph which finally change states
    synpase dynamics: a tf graph which finally change syapase
    synpase chould be simple W, either a complex network.only it can define two dynamics(change out, change in)

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


class Neurons(object):

    def __init__(self, name, shape, init_satets, leakage_factor=1):
        """
        shape: 一般是[batch_size, neuron_size]
        init_states: 初始值
        leakage_factor: 自身每个时间片泄露的比例,[0,1],1就是一般的人工神经网络
        """
        self.shape = shape
        self.name = name
        self.init_states = init_states
        self.states = tf.Variable(init_states, name=name)
        self.synpase_implact = []
        self.next_states = (1 - self.leakage_factor) * self.states
        self.update_op = tf.compat.v1.assign_add(self.states, self.next_states)
        self.update_val = self.next_states - self.states

    def get_val(self):
        """return value of states"""

    def set_val(self, val, clampled):
        """set value of states of visiable neurons"""
        tf.compat.v1.assign_add(self.states, val)

    def add_synpase_implact(self, implact):
        self.synpase_implact.append(implact)
        self.next_states = self.next_states + impact
        self.update_op = tf.compat.v1.assign_add(self.states, self.next_states)
        self.update_val = self.next_states - self.states


class Synpase(object):

    def __init__(self,
                 name,
                 neurons,
                 neuron_dynamic_fn=None,
                 synpase_dynamic_fn=None):
        """Abstract Multi-Head Synapase

        Args:
            neurons: a list of linked neurons
            neuron_dynamic_fn:
                fn(neurons = [ne0, ne1, ...], synpases) -> [up0, up2, up3, ..]
                up_i has same shape of ne_i.states

            synpase_dynamic_fn:
                fn(neurons = [ne0, ne1, ...], synpases) -> synpase_up

        因为这里synpase内部的数据结构都未定义，基于此的函数也是没有约束的
        """
        self.name = name
        self.neurons = self.neurons
        self.weights = None
        self.next_weights = None
        self.weights_update_op = None

        self.neuron_dynamic()
        self.synpase_dynamic()

    def neuron_dynamic(self):
        self.neurons_implacts = self.get_neuron_implacts()
        for implact in self.neurons_implacts:
            self.neurons.add_synpase_implact(implact)

    def synpase_dynamic(self):
        """abstract method"""
        pass


class TargetProgagtion(object):

    def __init__(self, layer_num):
        for l in range(layer_num):
            neurons =
        for l in range(layer_num):
            synpase_l =
            synpase_l_back =

""" how to do bp  synpase dynamic

方法1：显式的错误神经群，通过错误神经元群与之前的Heebian
方法2：类似bp构建前向-后向网络，但共享部分参数。结果就是上边。

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)
