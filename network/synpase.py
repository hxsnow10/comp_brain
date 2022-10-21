!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       synpase
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/20
#   Description:    ---
"""synpases.
"""

import os
import sys

import argparse
import numpy as np

from .neuron import Neurons

sys.path.append("..")
from op import *
import torch

class Synpase(object):

    def __init__(self,
                 neurons,
                 name = None,
                 synpase_shapes = None,
                 synpase_inits = None,
                 learning_rate_init = 0.01,
                 synpase_dynamic_fn_past_order = 0,
                ):
        """Multi-Head Synapase

        Args:
            neurons: a list of linked neurons
            synpase_shapes: list of shape
            synpase_inits: list of init val
            dynamic mode:
                * y_t += f(x_t)
                ** get global acctractor
                ** only once
                * y_{t+1} += f(x_t)
        注意到fn在tf1与tf2是2种行为，前者是定义，后者是直接执行。我希望是前者，可是世界。。
        注意到这里的fn是具备灵活性的，可以使用任意的计算图的函数;
        """
        #if shape: self.states_init = np.full(shape,states_init)
        #else:self.states_init = states_init
        self.name = name
        if self.name==None:
            self.name = "synpase_"+"->".join([n.name for n in neurons])
        self.neurons = neurons
        if synpase_inits is not None:
            self.weights = get_variable(synpase_inits)
            self.shape = self.weights.shape
            print(synpase_inits.shape, self.weights.shape)
        else: 
            self.weights = None
            self.shape = None
        self.next_weights = None
        self.learning_rate = get_variable(learning_rate_init)
        # Lr 一般是这常数，如果是一个与weights一样大小的tensor，可以控制到每个突触的粒度
        self.init_more()
        self.allow_weights_update = True
        print("new created", self.__str__())
    
    def __str__(self):
        if self.weights is not None:
            s = "synpase, name = {} , shape = {}, dtype = {}".format(self.name, self.weights.shape, self.weights.dtype)
        else:
            s = "synpase, name = {} , weights = None"
        return s

    def init_more(self):
        pass

    def init_states(self):
        pass

    def freeze(self):
        self.allow_weights_update = False

    def unfreeze(self):
        self.allow_weights_update = True

    def neuron_states_dynamic_imp(self):
        """神经元的states变化"""
        return 0
    
    def neuron_error_dynamic_imp(self):
        """神经元的error变化"""
        return 0

    def neuron_dynamic_imp(self):
        """神经元的变化,包括states、error2项
        fn(neurons, synpase) -> [ [ne1_var1_up, ne1_var2_up, ...], ... ]
        """
        states_impact = self.neuron_states_dynamic_imp()
        # 这样导致error存在一个step的延迟
        error_impact = self.neuron_error_dynamic_imp()
        print(self.name, "states_impact", lstr(states_impact))
        print(self.name, "error_impact", lstr(error_impact))
        rval = [[states_impact[i], error_impact[i]] for i in range(len(states_impact))]
        print(self.name, "neuron_impact", lstr(rval))
        return rval

    def synpase_dynmaic_imp(self):
        """突触的权重变化
        fn(neurons, synpase) -> [syn0_up, syn1_up, ...]
        实际上他是pre_neuron(t), post(t), synpase(t)的函数,换言之，需要内部变量记录过去信息。
        因为要涉及到结构与突触动力学的解耦，但同时提供突触动力学的灵活性，所以fn传入了synpase自身：
        比如说涉及高阶states的更新与使用；
        """
        raise("Not Implemented")

    def neuron_dynamic(self):
        # TODO impact could be tensor of list_of_tensor
        # 这个lr 是全局的还是局部的，应该是可选的。如果是局部的，就把全局的设置为1
        # 除了学习率，还有个freeze的概念。如果freeze就要临时把lr设置为0
        self.neurons_impacts = self.neuron_dynamic_imp()
        print(lstr(self.neurons_impacts))
        for k,impact in enumerate(self.neurons_impacts):
            print(self.name, "add impact", k, impact)
            self.neurons[k].add_impact(impact)

    def learning_rate_dynamic(self):
        pass

    def synpase_dynamic(self):
        # 先假设所有synpase只有一个weight
        if not self.allow_weights_update: return
        synpase_impacts = self.synpase_dynamic_imp()
        self.learning_rate_dynamic()
        print(self.__str__())
        print("weights", self.weights.shape)
        print("impacts", synpase_impacts.shape)
        self.weights+=synpase_impacts*self.learning_rate

class TargetBPSynpase(Synpase):
    """考虑一种真实bp形成的突触，它的动力学是1)fn(neurons) update  2）计算其中一个neuron对其他neuron的梯度，反馈error
    TODO: 如果使用内置的bp，要在target的计算过程neuron_dynamic_imp加了gradient_tape； bptt类似，在恰当的地方加上，避免tape爆炸
    """

    def set_traget_neuron_index(self, idx):
        self.target_ne_idx = idx

    def neuron_error_dynamic_imp(self):
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
        input("xxxx")
        return rval

class MseSynpase(TargetBPSynpase):
   
    target_ne_idx = 2
    def init_more(self):
        self.max_target = 0

    def synpase_dynamic(self):
        """无参数，不需要自我学习"""
        pass

    def neuron_states_dynamic_imp(self):
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

class HebbianSynpase(Synpase):

    def synpase_dynmaic_imp(self):
        """
        shape of states1/2=  [batch, size]
        shape of return = [[size1, size2], [size2, size3], ...]
        """
        states1, states2 = self.neurons[0].states, self.neurons[1].states
        rval = get_matmul(get_expand_dims(states1,2),get_expand_dims(states2,1))
        return rval

class STDPSynpase(Synpase):
    """STDP fn
    here we implement STDP : \delta(W)=lr*pre_state*\devirate(post_state)
    post_devirate_{t+1} =  (post_states_{t}-post_states_{t-1})/(1*dt)

    when state, next_state, we get new 
    here k*dt should could ot be very big.
    """
    def init(self, post_devirate_decay=1):
        self.post_devirate = get_variable(post_neuron.shape())
        self.post_devirate_decay = post_devirate_decay

    def synpase_dynamic_imp(self):
        # TODO divide timestamp
        self.post_devirate = (1-self.post_devirate_decay)*synpase.neuron_impacts + \
                self.post_devirate_decay*self.post_devirate
        pre_states =  self.neurons[0].states
        synpase_update = self.post_devirate*pre_states
        return synpase_update

class ErrorBPSynpase(Synpase):
    
    def synpase_dynamic_imp(self):
        states1 = self.neurons[0].out_states
        # 按公式是要用out_states
        error2 = self.neurons[1].error
        rval = get_matmul(get_expand_dims(states1,2),get_expand_dims(error2,1))
        rval = torch.sum(rval, dim = 0)
        return rval

class BiLinearSynpase(Synpase):
    """Bi Linear Synpase.

    """
    def __init__(self, neurons, name = None, 
                 go_factor=1, back_factor=1):
        synpase_inits = synpase_inits = get_variable(
             [neurons[0].shape[1], neurons[1].shape[1]])
        super(BiLinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)

    def neuron_states_dynamic_imp(self):
        states1, states2 = self.neurons[0].states, self.neurons[1].states
        impact_2 = self.go_factor * get_matmul(states1, self.weights)
        impact_1 = self.back_factor * \
            get_matmul(states2, self.weights, transpose_b=True)
        return [impact_1, impact_2]

class LinearSynpase(Synpase):
    def __init__(self, neurons, name = None, error=True):
        synpase_inits = np.full([neurons[0].shape[-1], neurons[1].shape[-1]], 0.01)
        print("fuck", synpase_inits.shape)
        super(LinearSynpase, self).__init__(neurons, name, synpase_inits = synpase_inits)
        self.error = error

    def neuron_error_dynamic_imp(self):
        error1, error2 = self.neurons[0].error, self.neurons[1].error
        impact_2 = 0
        if self.error:
            impact_1 = get_matmul(error2, self.weights, transpose_b=True)
        else:
            impact_1 = 0
        return [impact_1, impact_2]

    def neuron_states_dynamic_imp(self):
        states1, states2 = self.neurons[0].states, self.neurons[1].states
        impact_2 = get_matmul(states1, self.weights)
        return [0, impact_2]

class MLPMetaSynpase(Synpase):

    def __init__(self, meta_synpase_type = None, mlp_srags):
        # 注意到这里weights变成的neurons，与其他对象形成网络！
        # 方法是共享变量生成一个新的neuron（这里使用）
        # 即这时候，synpase一方面weights变成了具有动力学的neuron
        # 另一方面weights的更新动力学形成了网络
        self.meta_synpase_type = meta_synpase_type
        self.w_neuron = Neurons(states_ref = self.weights, leaky=1)
        # 不设置leaky，就是把历史存在内部，设置leaky相当于引入预设
        input = [neurons.states. w_neuron, neurons.error]
        self.delta_w = Neurons(w.states())
        self.delta_error = Neurons()
        # 实际上是多层rnn
        # 最好把这里mlp计算延时设置为0
        # 显式地考虑对error变化的建模
        self.meta_synpase = general_mlp(input, [delta_w,error_implact], **mlp_args)
        # 这里delta_w与梯度的关系也可以是预设的，而把元学习约束到梯度传播上 TODO

    def synpase_dynamic_imp(self):
        self.meta_synpase.neuron_dynamic()
        # 何时运行self.synpase.synpase_dynamic()呢
        # 这里meta_synpase作为本synpase的子系统，需要本synpase来触发子synpase的动力学。
        # 另一种方案是把子synpase暴露到外边，异步执行，这会导致延时，这里使用过程式的形式
        self.meta_synpase.synpase_dynamic()
        # 这里依旧需要外层synpase传导梯度支持元网络的BP
        
        return self.delta_w.states

    def neuron_error_dynamic_imp(self):
        # do error implact
        self.meta_synpase.neuron_dynamic()
        # TODO 与上边只计算一次
        return self.delta_error.state

class TBPTTSynpase(Synapse):

    def neuron_error_dynamic_imp(self):
        pass

class FPTTSynpase(Synpase):

    pass

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
