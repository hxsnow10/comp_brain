#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       synpase_base
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/12/22
#   Description:    ---
"""突触基类

description
"""

import os
import sys
import logging

import argparse
import numpy as np
import torch

from .neuron import Neurons

sys.path.append("..")
from op import *
from util.log import log_info, log_debug

class Synpase(object):
    """Synpase基类, 定义了Neuron之间的通信动力学(inference), 以及自身学习的动力学(learning)
    """

    def __init__(self,
                 neurons,
                 name = None,
                 synpase_shapes = None,
                 synpase_inits = None,
                 learning_rate_init = 0.01,
                 learning_dynamic_fn_past_order = 0,
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
            logging.debug((synpase_inits.shape, self.weights.shape))
        else:
            self.weights = None
            self.shape = None
        self.next_weights = None
        self.learning_rate = get_variable(learning_rate_init)
        # Lr 一般是这常数，如果是一个与weights一样大小的tensor，可以控制到每个突触的粒度
        self.init_more()
        self.allow_weights_update = True
        self.sub_synpases = []
        logging.info("new created synpase = {}".format(self.__str__()))

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

    def inference_dynamic(self):
        # TODO impact could be tensor of list_of_tensor
        # 这个lr 是全局的还是局部的，应该是可选的。如果是局部的，就把全局的设置为1
        # 除了学习率，还有个freeze的概念。如果freeze就要临时把lr设置为0
        self.neurons_impacts = self.inference_neuron_states_impact()
        logging.debug("inference_dynamic, neuron_impacts = {}".format(lstr(self.neurons_impacts)))
        if self.neurons_impacts is not None:
            for k,impact in enumerate(self.neurons_impacts):
                logging.debug("{} add impact {}".format(self.name,impact))
                self.neurons[k].add_impact(impact)
    
    def inference_neuron_states_impact(self):
        """神经元的显式states变化"""
        raise("Not Implemented")

    def inference_neuron_error_impact(self):
        """神经元的隐式error变化"""
        raise("Not Implemented")

    def learning_rate_dynamic(self):
        pass

    def learning_dynamic(self):
        # TODO：增加一个开关
        # 这个环节往往依赖突触2侧的状态或者 RNN前后的状态
        # 最好是存在计算依赖的2个状态 y = f(x), 或者 x_t = f(x_t-1)
        # 所以把这个环节放在inference后边
        logging.info("synpase {} start learning".format(self.name))
        self.error_impacts = self.inference_neuron_error_impact()
        logging.info("error impact = ".format(self.error_impacts))
        if self.error_impacts is not None:
            for k,impact in enumerate(self.error_impacts):
                logging.debug("neuron {} add error impact {} {}".format(self.name,k, impact))
                print(self.name)
                print(impact)
                if impact is not None and impact is not 0:
                    print(impact.sum())
                    if self.neurons[k].error is not None:
                        self.neurons[k].add_error_impact(impact)

        # 先假设所有synpase只有一个weight
        if self.allow_weights_update:
            synpase_impacts = self.learning_synpase_impact()
            self.learning_rate_dynamic()
            logging.debug(self.__str__())
            if synpase_impacts is not None:
                logging.debug("weights shape = {}".format(self.weights.shape))
                logging.debug("impacts shape = {}".format(synpase_impacts.shape))
                self.weights+=synpase_impacts*self.learning_rate
            # 元学习中子突触学习
            for synpase in self.sub_synpases:
                synpase.learning_dynamic()
    
    def learning_synpase_impact(self):
        """突触的权重变化
        fn(neurons, synpase) -> [syn0_up, syn1_up, ...]
        实际上他是pre_neuron(t), post(t), synpase(t)的函数,换言之，需要内部变量记录过去信息。
        因为要涉及到结构与突触动力学的解耦，但同时提供突触动力学的灵活性，所以fn传入了synpase自身：
        比如说涉及高阶states的更新与使用；
        """
        raise("Not Implemented")

    def freeze(self):
        self.allow_weights_update = False

    def unfreeze(self):
        self.allow_weights_update = True

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

