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


"""

import argparse
import os
import sys

"""
### error收敛
给每个神经元配一个error，设计动力学，当收敛时error的形式恰好符合error传播。
error传播即反向传播，所以就是在原有网络连接上，每个连接配的；y=f(z=Wx1+Wx2+...)
dC/dx1 = dC/dy*dy/dx1 = error_y*f'(z)*W error_x=W^T*error_y
* 原网路可以保持不变；
* 也可以受error影响转到新状态
* 最后的学习依赖error与原状态(或者新状态)

整体的过程是：
* 预测，产生error
* error传播，训练

以下分别实现
* 基于多层MLP
* 基于一般计算图的，包括CNN, RNN
* 基于反馈神经网络
* 基于一般动力学神经网络
"""

def error_neuron_fn(object):
    states1, states2 = states
    error = states2 - weights*states1
    update1 = tf.transpose(weights)*error
    update2 = -error
    return [update1, update2]

def main(
        layer_num,
        neurons,
        synpases,
        input_inxs,
        output_inxs,
        stop_th=0.1):
    x_neurons = []
    x_neurons.append(Neusons())
    synpases = []
    for l in range(1, layer_num):
        x_neurons.append(Neurons(0))
        synpase = BiLinearSynpase("synpase"+str(l),x_neurons[l-1], x_neurons[l],
                                 error_neuron_fn, hebbian_synpase_fn)
        synpase.append(synpase)

    network = Network(x_neurons, synpases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)
