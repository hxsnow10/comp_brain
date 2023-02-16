#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       network
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/20
#   Description:    ---
"""network of neurons and synpases.

network由于用neuron与synpase组成，性质与一般函数y=F(x)的计算图有区别，netowrk主要描述的是：
    h_{t+1}, param_{t+1} =F (h_{t}, param_{t}, x_{t})
其中，h由neuron动力学以及突触（传播）动力学定义，param由突触（提升）动力学定义。
这里我们进一步约束单层的synpase-neuron的F是类MAU的结构，而不拓展到任意的计算图。
纯NN结构就是activation(WX)，但表达力会较弱，比如BILSTM，GRU的表达会困难。
类MAU即g(f(synpase1, neurons)+f(synpase2, neurons)+....),是为了在类脑的原子性与表达能力中取得一个折中。
绝大多数，这里涉及的算法都是纯NN（前馈或者反馈）。但是因为类MAU设计，可以直接拓展到GRU， BILSTM更复杂的网络上。

如果希望这种网络与一般的动力学结合：
1. 动力学转化为计算图，与其他计算图连接：h中选取部分作为y，那么x_t->y_{t+1}（或者y_t，但实际依赖的是x_{t-1})存在计算图关系，很容易得到。
但又一定区别：neuron实际上表示的是变量的序列，而计算图中都是明确单一时间的变量。
这就隐含了：y=F(x) 1）如果x是时序，y也是时序 2）如果x是固定的，那么y也是固定的。

2. 动力学内部可以包含任意的计算图
因为内部实现与外部接口都可以转化为计算图，即这种网络也可以使用标准的BP去训练。

We Induce the design from following main characteristics/assumes:
    1、Structure inclues (states, weights) , or say (neurons, synpases)
    2、Dynamics include neuron-dynamics and synpase-dynamics.
    3、All Dynamics within structure all local (except some phase control signal)

So we can easily deduce an interface which's similar to brian2.

Neurons: states
    * tensor
    * neurons may with self-dynamics, like leak or rnn-dynamics
    * high-order neurons-network include visiable-neurons, inside structure, inside dynamics

Synpase: link between neurons neurons
    * tensor
    * tr dynamics: define computation to change postsynpase neuron
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

import os
import sys

import argparse

class Network():
    """ a class to manage neurons, synpases, forward, backword
    """

    def __init__(self, neurons=[], synpases=[], activation = None, leak = 0.1, error = None, 
                    synpase_type=None, 
                    input_idxs=[],
                    input_names = [],
                    output_names = []):
        self.name2neurons = {n.name:n for n in neurons}
        self.name2synpases = {s.name:s for s in synpases}
        self.activation = activation # default activation
        self.leak = leak # default activation
        self.error = None
        self.synpase_tye = synpase_type
        self.input_names = input_names
        self.output_names = output_names

    def clamp_inputs(self, inputs, input_names):
        if not inputs:
            return
        for name, inp in zip(input_names, inputs):
            self.name2neurons[name].set_val(inp, clamped = True)
            print("after clamp", self.name2neurons[name])

    def set_io_names(self, input_names, output_names):
        self.input_names = input_names
        self.output_names = output_names

    def apply(self, inputs = None, input_names=None, output_names=None, time_step_num=1, neuron_on = True, synpase_on = True):
        """根据输入运行，返回输出。
        Args
            inputs: list of input tensor to clamp
            input_names: list of input tensor name
            output_names: list of output tensor name
            time_step_num: time step to run
        """
        input_names = input_names or self.input_names
        output_names = output_names or self.output_names
        for t in range(time_step_num):
            self.clamp_inputs(inputs, input_names)
            for name,synpase in self.name2synpases.items():
                synpase.inference_dynamic()
            for neuron in self.name2neurons.values():
                neuron.dynamic()
            for name,synpase in self.name2synpases.items():
                synpase.learning_dynamic()
        return self.get_val(output_names)

    def init_states(self):
        """网络的状态初始化，对不同样本输入时进行，不修改参数
        """
        for synpase in self.name2synpases.values():
            synpase.init_states()
        for neuron in self.name2neurons.values():
            neuron.init_states()

    def get_val(self, name):
        # TODO consider list
        return self.name2neurons[name].get_val()

    def add_synpase(self, synpases):
        if type(synpases)!=type([1,2,3]):
            synpases = [synpases]
        for synpase in synpases:
            self.name2synpases[synpase.name] = synpase
            for neuron in synpase.neurons:
                if neuron.name not in self.name2neurons:
                    print("network add neuron", neuron.name)
                    if self.activation and not neuron.activation:
                        neuron.activation = self.activation
                    if self.leak and not neuron.leak:
                        neuron.activation = leak
                    self.name2neurons[neuron.name] = neuron
        print("after add synpase")
    
    def report(self):
        for neuron in self.name2neurons.values():
            print(neuron.name, neuron.states.shape)
    
    def check_neurons(self, process_name):
        print("start {}!!!!!!!!!".format(process_name))
        for neuron in self.name2neurons.values():
            print(neuron)

if __name__=='__main__':
    from neuron import *
    from synpase import *
    a = Neurons(0.01, [100,100])
    b = Neurons(0.01, [100,100])
    c = SpikingNeurons(0.01, [100,100])
    n = network()
    n.add_synpase( LinearSynpase([a,b]) )
    n.add_synpase( LinearSynpase([b,c]) )
    n.add_synpase( BiLinearSynpase([a,b]) )
    n.add_synpase( LinearBPSynpase([a,b]) )
    n.add_synpase( MseSynpase([a,b]) )

