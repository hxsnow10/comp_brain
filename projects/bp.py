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

# import tensorflow_datasets as tfds
from torchvision import datasets, transforms
sys.path.append("..")
from network import *
from agent_env import *
from utils import *

def get_mnist_agent(model_name):
    batch_size = 64
    in_size = 28*28
    x = Neurons(name = "x", states_init = np.full([batch_size, in_size],0), error=True)
    target = ForwardNeurons(name="target", states_init = np.full([batch_size, 1],0), error=True)
    y = Neurons(name = "y", states_init = np.full([batch_size, 10],0), error=True, requires_grad = True)
    y_true = Neurons(name = "y_true", states_init = np.full([batch_size, 10],0), error=True)
    network = Network([x, target, y, y_true], activation = "relu")
    hidden_layer_sizes = [int(in_size/2), int(in_size/4)]
    if model_name == "Bp-d":
        synpase_type = LinearBPSynpase
        error2states = 0
    elif model_name =="Bp-da":
        synpase_type = LinearBPSynpase
        error2states = 0.1
    elif model_name == "PC":
        synpase_type = PCSynpase
        error2states = 0
        #需要考虑PC是clamp-y类型，而不是loss(y,y_true)类型
        # 把clamp设计为一种输入影响的网络！
    elif model_name == "TP":
        pass # TODO
    network.add_synpase(MseSynpase(name = "tsynpase", neurons = [y, y_true, target]))
    network.add_synpase(general_mlp(x,y, synpase_type,hidden_layer_sizes = hidden_layer_sizes,  states2error = 0.1))
    agent = Agent(network)
    agent.network.check_neurons("222222")
    print(agent)
    return agent

def test_mnist(agent, test_env):
    """ 动力学网络与一般的的计算图不同的地方在于：y 是一个时间序列，而不是一个值。
    因此，loss也不是一个值，而是一个时间序列。
    在测试的时候，我们与训练时候类似的，在第二阶段，给予y，观察loss
    即测试与训练的接口一致：观察当y出现时loss的平均值的变化。
    """
    agent.work(test_sample) # without train
    # get loss

def get_tf_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True)

def get_pytorch_minist_data():
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
    ds_train = datasets.MNIST(root = "~/Codes/pytorch_data/",
                                transform=transform,
                                train = True,
                                download = True)

    ds_test = datasets.MNIST(root="~/Codes/pytorch_data/",
                               transform = transform,
                               train = False)

    ds_train = torch.utils.data.DataLoader(dataset=ds_train,
                                                batch_size = 64,
                                                shuffle = True)

    ds_test = torch.utils.data.DataLoader(dataset=ds_test,
                                               batch_size = 64,
                                               shuffle = True)
    
    def transform(d):
        x,y = d
        x = torch.reshape(x,[64,-1]).type(torch.float32)
        y = torch.nn.functional.one_hot(y,10).type(torch.float32)
        return x,y
    ds_train = PytorchData(ds_train, transform)
    ds_test = PytorchData(ds_test, transform)
    
    return ds_train, ds_test

def main():
    ds_train, ds_test = get_pytorch_minist_data()
    train_env = SimpleSupervisedEnv(ds_train)
    test_env = SimpleSupervisedEnv(ds_test)
    model_names = ["Bp-d"]
    for model_name in model_names:
        agent = get_mnist_agent(model_name)
        agent.network.check_neurons("333333")
        for epoch in range(10):
            agent_work(agent, train_env, ["x", "y_true"], max_step=10000, neuron_on=True, synpase_on=True)
            agent_work(agent, test_env, ["x", "y_true"], max_step=10000, neuron_on=True, synpase_on=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()

