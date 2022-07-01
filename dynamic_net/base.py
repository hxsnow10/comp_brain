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

So we can easily deduce an interface which's similar to brian2.

Neurons: states
    * tf.tensor
    * neurons may with self-dynamics, like leak or rnn-dynamics
    * high-order neurons-network include visiable-neurons, inside structure, inside dynamics

Synpase: link between neurons neurons
    * tf.tensor
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

class Neurons(object):

    def __init__(self, name, shape=None, init_satets=None, 
                 activation = None,
                 leak_init = 0.1, 
                 self_dynamics = None,
                 visiable = False
                )
        """
        ## Args
        name: name string
        shape: 一般是[batch_size, neuron_size]
        init_states: 初始值
        leak_init: 默认值  TODO: 进一步 leak也可以是一个函数
        TODO: consider spike mode
        self_dynamics:
            * "rnn_train"
            * "leak_train"
            * "leak_static"
            self-dynamics实际上是synpase, 那么work的时候也要触发这里的动力学,更复杂的动力学还是通过synpase去管理?
        visiable: only visiable neurons could be set manualy
        """
        self.shape = shape
        self.name = name
        self.init_states = init_states
        self.states = tf.Variable(init_states, name=name)
        self.leak = tf.Variable(leak_init)
        self.synpase_implact = []
        self.next_states = (1 - self.leak) * self.states
        self.clamped = False
        if activation = None:
            activation = tf.relu
        self.activation = activation

    def get_val(self):
        """return value of states"""

    def set_val(self, val, clampled):
        """set value of states of visiable neurons"""
        tf.compat.v1.assign_add(self.states, val)

    def clamp(self, val):
        self.states = val
        self.clamped = True

    def add_synpase_implact(self, implact):
        if self.clampled: return None
        self.synpase_implact.append(implact)
        self.next_states = self.next_states + impact

    def forward(self):
        """
        得先把所有神经元的下个状态算出来，保证仿真不产生时间不合法的依赖
        """
        self.states = self.next_states

class Synpase(object):

    def __init__(self,
                 name,
                 neurons,
                 synpase_shapes,
                 synpase_inits,
                 neuron_dynamic_fn=None,
                 synpase_dynamic_fn=None,
                 synpase_dynamic_fn_past_order = 0):
        """Multi-Head Synapase

        Args:
            neurons: a list of linked neurons
            synpase_shapes: list of shape
            synpase_inits: list of init val
            neuron_dynamic_fn:
                fn(neuron_tensors = [ne0, ne1, ...], synpase_tensors, synpase) -> [ne_up0, ne_up1, ...]
                up_i has same shape of ne_i.states
                如果需要高阶信息，会存在synpase中;
            synpase_dynamic_fn:
                fn(neuron_tensors = [ne0, ne1, ...], synpase_tensors, synpse) -> [syn0_up, syn1_up, ...]
                实际上他是pre_neuron(t), post(t), synpase(t)的函数,换言之，需要内部变量记录过去信息。
                因为要涉及到结构与突触动力学的解耦，但同时提供突触动力学的灵活性，所以fn传入了synpase自身：
                比如说涉及高阶states的更新与使用；
            dynamic mode:
                * y_t += f(x_t)
                ** get global acctractor
                ** only once
                * y_{t+1} += f(x_t)
        注意到fn在tf1与tf2是2种行为，前者是定义，后者是直接执行。我希望是前者，可是世界。。
        注意到这里的fn是具备灵活性的，可以使用任意的计算图的函数;
        """
        self.name = name
        self.neurons = self.neurons
        self.weights = None
        self.next_weights = None
        self.weights_update_op = None
        self.neuron_dynamic_fn = neuron_dynamic_fn
        self.synpase_dynamic_fn = synpase_dynamic_fn
        self.learnin_rate = tf.Variable(learning_rate)
        #self.neuron_dynamic()
        #self.synpase_dynamic()

    def neuron_dynamic(self):
        self.neurons_implacts = control_lr*self.neuron_dynamic_fn([ne.states for ne in self.neurons], self.weights, self)
        for k,implact in enumerate(self.neurons_implacts):
            self.neurons[k].add_synpase_implact(implact)

    def synpase_dynamic(self, control_lr=1):
        self.synpase_implacts = control_lr*self.learning_rate * self.synpase_dynamic_fn([ne.states for ne in self.neurons], self.weights, self)
        for k,weight in self.synpase_implacts:
            self.weights[k]+=self.synpase_implacts[k]

def hebbian_synpase_fn(stetes, weights, synpase):
    """
    shape of states1/2=  [batch, size]
    shape of return = [[size1, size2], [size2, size3], ...]
    """
    states1, states2 = states
    rval = if.matmul(tf.expand_dims(states1,2),tf.expand_dims(states2,1))
    return rval

class stdp_synpase_fn(object):
    """STDP fn
    here we implement STDP : \delta(W)=lr*pre_state*\devirate(post_state)
    post_devirate_{t+1} =  (post_states_{t}-post_states_{t-1})/(1*dt)

    when state, next_state, we get new 
    here k*dt should could ot be very big.
    """
    def __init__(self, post_devirate_decay=1):
        self.post_devirate = tf.Variable(post_neuron.shape())
        self.post_devirate_decay = post_devirate_decay

    def __call__(self, states, weights, synpase):
        # TODO divide timestamp
        self.post_devirate = (1-self.post_devirate_decay)*synpase.neuron_implacts + \
                self.post_devirate_decay*self.post_devirate
        pre_states =  states[0]
        synpase_update = self.post_devirate*pre_states
        return [synpase_update]

def bilinear_neuron_fn(states, weights, synpase):
    states1, states2 = states
    implact_2 = go_factor * tf.matmul(states1, self.weights)
    implact_1 = back_factor * \
        tf.matmul(states2, self.weights, transpose_b=True)
    return [implact_1, implact_2]

class BiLinearSynpase(Synpase):
    """Bi Linear Synpase.

    """
    def __init__(self, name, pre_neuron, post_neuron, 
                 neuron_dynamic_fn = bilinear_neuron_fn,
                 synpase_dyamic_fn = hebbian_synpase_fn,
                 go_factor=1, back_factor=1):
        neurons = [pre_neuron, post_neuron]
        self.weights = tf.Variable(
             [self.neurons_1.shape[1], self.neurons_2.shape[1]], name=name)
        super(BiLinearSynpase, self).__init__(name, neurons, shape, init,
                                              neuron_dynamic_fn,
                                             synpase_dynamic_fn)

class LinearSynpase(BiLinearSynpase):
    def __init__(self, name, pre_neuron, post_neuron,
                 neuron_dynamic_fn = bilinear_neuron_fn,
                 synpase_dyamic_fn = hebbian_synpase_fn)
        super(LinearSynpase, self).__init__(name, name, pre_neuron, post_neuron,
                                              neuron_dynamic_fn,
                                             synpase_dynamic_fn)
                                            go_facto r=1, back_factor=0)

def link(pre_neron, post_neuron, layer = "mlp", layer_sizes = []):
    """
    mode: how to link two neurons
    * linear
    * multi-layer
    * cnn
    * attention
    """
    if type(layer)!=type("mlp"):
        def neuron_dynamic_fn([pre_state, post_state]):
            update = layer(pre_state)
            return [0, update]
        synpase = Synpase(pre_state, post_state, neuron_dynamic_fn, None)
    elif layer=="linear":
        synpases = [ LinearSynpase(pre_neuron, post_neuron)]
    elif layer[:3]=="mlp":
        """ high order synpase consist multi layer 
        pre_neuron(n)->[ iter_neruon[n0] -> ... iter_neuron[nk] ] -> post_neuron[m]

        有2种思路：第一种fn实现为多层mlp即可； 第二种实现为多层的neurons与synpnase
        
        前者mlp 预测时会很简洁，但是训练参数时需要考虑特殊的学习，相当于脱离了底层一致的原则。
        为了更快复用AI社区的技术，也考虑直接使用计算图的方式。训练的时候，引入模块的bp学习。
        注意到y=F(x)，在Neuron里有一些本地的方法把error传递过来，那么就可以用来训练F。

        这里考虑更一致的方法，就是拆解成多个Linearsynpase.
        TODO: 其他模块,attention，cnn
        * leak=1, 其他网络连接与原本一致
        * 是否某项场景需要leak!=1呢。考虑对象本身是考虑时间的！的cnn与rnn
        * 本地化分析
        """
        # TODO auto generate layer size
        neurons = [pre_neuron]
        synpases = []
        for k, layer_size in enumerate(layer_sizes):
            nname = name+"iter_neuron_k"
            neuron = Neurons(nname, layer_size)
            synpases.append(link(neurons[-1], neuron))
            neurons.append(neuron)

    return synpases

class network():
    """ a class to manage neurons, synpases, forward, backword
    """

    def __init__(self, neurons, synpases):
        self.neuron_names = []
        self.neurons = neurons
        self.synpases = synpases

    def link(self, pre_neron, post_neuron, mode = "mlp", layer_sizes = []):)
        if pre_neuron.name not in self.neuron_names:
            self.neurons.append(pre_neuron)
        if post_neuron.name not in self.neuron_names:
            self.neurons.append(post_neuron)
        neurons, synpases = link(pre_neron, post_neuron, mode, layer_sizes)
        self.neurons.extend(neurons)
        self.synpases.extend(synpases)

    def update_neuron(self):
        for synpase in self.synpases:
            synpase.neuron_dynamic(control_lr)
            # TODO collect all neurons and finaly update states to next
        for neuron in self.neurons:
            self.neuron.forwad(mode)
    
    def clamp_input(self, inputs, input_inxs):
        for inx, inp in zip(input_inxs, inputs):
            self.neurons[inx].clamp(inp)

    def update_synpase(self, control_lr):
        for synpase in self.synpase:
            synpase.sypase_dynamic( control_lr )

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)
