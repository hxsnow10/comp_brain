#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       bp_model
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         17/04/2022
#   Description:    ---
""" Predictive Coding
PC有2个设计动机：
* 从近似BP出发，需要构建与传播error项
* 从提前预测出发，top-down预测与bottom-up预测一起构成了真实的图景

另外，不管是哪个设计思想，具体到实现上，都会因为通用性、任务不同，
表现出不同的形式。相关理论的研究依旧有许多空白，尤其是期望的最通用、高效的形式。

"""

"""Predicitive Network.
Song, Yuhang, Thomas Lukasiewicz, Zhenghua Xu, and Rafal Bogacz. 2020. ‘Can the Brain Do Backpropagation? —Exact Implementation of Backpropagation in Predictive Coding Networks’. Advances in Neural Information Processing Systems 33: 22566–79.

Lillicrap, Timothy P., Adam Santoro, Luke Marris, Colin J. Akerman, and Geoffrey Hinton. 2020. ‘Backpropagation and the Brain’. Nature Reviews Neuroscience 21 (6): 335–46. https://doi.org/10.1038/s41583-020-0277-3.


### error收敛
给每个神经元配一个error，设计动力学，当收敛时error的形式恰好符合error传播。
有几个工作：
* 通过error来传播，Backpropagation and the Brain, 实现为mlp_predicitive
* 不影响前馈，error按反向传播变成对应的动力学，实现为mlp_bp_predicitive
* 关于反馈网络的近似bp算法(需要解决bptt的难点)
* target产生: 对未来的预测产生mse-target
"""

import argparse
import os
import sys

def mlp_predicitive(
    # 标准的predicitive-coding mlp
    # 通过error来进行上下层的沟通，具体参考Backpropagation and the Brain
        layer_num,
        neurons,
        synpases,
        input_inxs,
        output_inxs,
        stop_th=0.1):
    def error_neuron_fn(states):
        states1, states2 = states
        error = states2 - weights*states1
        update1 = tf.transpose(weights)*error
        update2 = -error
        return [update1, update2]
    x_neurons = []
    x_neurons.append(Neusons())
    synpases = []
    for l in range(1, layer_num):
        x_neurons.append(Neurons(0))
        synpase = BiLinearSynpase("synpase"+str(l),x_neurons[l-1], x_neurons[l],
                                 error_neuron_fn, hebbian_synpase_fn)
        synpase.append(synpase)
    network = Network(x_neurons, synpases)

def mlp_bp_predicitive(
        layer_num,
        neurons,
        synpases,
        input_inxs,
        output_inxs,
        stop_th=0.1):
    """ 一种与bp更相似的网络。每个神经元包含(states, error)，前向不变，后向每个神经元配个error按bp方式进行传播。
    只有当target真正触发了，非0-error才会传递到整个网络。
    注意到这里一定得是前向网络，并且输入保持一段时间的稳定才行。"""
    x_neurons = []
    x_neurons.append(Neusons(error=True))
    synpases = []
    PSynpase = type("PSynpase",(LinearSynpase, ErrorBPSynpase))
    for l in range(1, layer_num):
        x_neurons.append(Neurons(error=True))
        synpase = PSynpase("synpase"+str(l),x_neurons[l-1], x_neurons[l])
        synpase.append(synpase)
    target = ForwardNeurons(error=True)
    y_true = Neurons(error=True)
    # error项是target对神经元的微分。target可以是分布式多个来源的。
    # 我们只要描述目标神经元与目标的关系,然后采取TargetBPSynpase，把target-states转化为error
    Tsynpase = type("TSynpase",(TargetBPSynpase, MseSynpase))
    synpase = TSynpase("tsynpase", [x_neurons[layer_num-1], y_true, target])
    synpase.append(synpase)
    x_neurons+=[y_true, target]
    network = Network(x_neurons, synpases)

# 反馈神经网络存在困难，需要特殊处理：
# * 或者使用短期记忆保存历史状态，然后使用BP
# * 或者使用前向梯度，复杂度太高，需要考虑近似策略
# 思考记在知识库，TODO

# local_predicitve_coding  局部可以产生自我预测的error
# 可以是空间或者时间的
# 时间涉及与到预测时间窗口的问题。TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)
