#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       mnist
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/08/17
#   Description:    ---
"""Mnist task to test various bp-like learning algorithms.

Algos inccluds: 
    1) Bp-d: Bp with dynamics
    2) Bp-da: Bp dynamics and error change states
    3) Pc: Predictive Coding
    4) TP: target Prop
    5) TP-
    6) EP
"""

import os
import sys

import argparse

import tensorflow_datasets as tfds

def get_mnist_agent(model_name):
    x = Neurons(error=True)
    target = ForwardNeurons(error=True)
    y = Neurons(error=True)
    y_true = Neurons(error=True)
    network = Network([x, target, y, y_true], activation = "relu")
    hidden_layer_sizes = [int(x.size/2), int(x.size/4)]
    if model_name == "Bp-d":
        synpase_type = BPSynpase
        error2states = 0
    elif model_name =="Bp-da":
        synpase_type = BPSynpase
        error2states = 0.1
    elif model_name == "PC":
        synpase_type = PCSynpase
        error2states = 0
        #需要考虑PC是clamp-y类型，而不是loss(y,y_true)类型
        # 把clamp设计为一种输入影响的网络！
    elif model_name == "TP":
        pass # TODO
    network.add_synpase(MseTSynpase("tsynpase", [x_neurons[layer_num-1], y_true, target], target_name ="target"))
    network.add_synpase(general_mlp(x,y, synpase_type, error2states = 0.1))
    agnet = Agent(network)
    return agent

def test_mnist(agent, test_env):
    """ 动力学网络与一般的的计算图不同的地方在于：y 是一个时间序列，而不是一个值。
    因此，loss也不是一个值，而是一个时间序列。
    在测试的时候，我们与训练时候类似的，在第二阶段，给予y，观察loss
    即测试与训练的接口一致：观察当y出现时loss的平均值的变化。
    """
    agent.work(test_sample) # without train
    get loss

def main():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    mnist_train_env = SupervisedEnv(ds_train)
    mnist_test_env = SupervisedEnv(ds_test)

    for model_name in model_names:
        agent = get_mnist_agent(model_name)
        for epoch in range(10):
            agent_work(agent, mnist_train_env, epoch=1, neuron_on=True, synpase_on=True)
            agent_work(agent, mnist_test_env, epoch=1, neuron_on=True, synpase_on=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_synpase_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

