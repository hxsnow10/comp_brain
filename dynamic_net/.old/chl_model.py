#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       chl_model
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         12/04/2022
#   Description:    ---

"""Constrative Hebbian Learning Model.

Model: Multi recurrent Layer with symmetry weight: x_l-W_l-x_{l+1}  
Dynamics:  d{x}_{l}=W_{l-1} x_{l-1}+W_{l}^{T} x_{l+1}-x_{l}

1. clamp x0=x, run until converge
2. camp c0=x, xl=y, run until converge
3. update parameter: 
train Dynamics: \Delta W_{2} \sim (t-x_{3}) x_{2}^{T}=-x_{3} x_{2}^{T}+{t} x_{2}^{T}

TODO: while no bias?
while no no more factor?
TODO: feedback random weights 
"""


"""
 i = Neurons()
 j = Neurons()
 global_controler_variable =  
 Synase(i,j, dynamics, update_rule)
 
 input data(with controler)

 special dynamic: w:i->j just update j onetime.
 special update: update by bp

 CHL ok
 Predicitive Coding ok
 target progagtation ok
 EP
"""

import os
import sys

import argparse

from hebbian import hebbian


class FeedwardLinearCHLModel(object):
    """ Feedward Linear 
    """

    def __init__(self, batch_size, layer_num, layer_sizes):
        shapes = 
        self.states = []
        self.weights = []
        for l in range(layer_um):
            # 输入层也转为float64
            self.states.apped(
                tf.placeholder(tf.float64, [batch_size, layer_sizes[l]], name="layer"+str(i)))
            if l+1<layer_um:
                self.weights.append(
                    tf.Variable(tf.float64,
                                [layer_sizes[l],layer_sizes[l+1]],
                                name = "weight"_+str(l)))
        self.next_states = []
        self.on_paramter = []
        # 考虑加快收敛的方法与参数控制
        for l in range(1, layer):
            dXl = tf.mul(self.states[l-1], self.weigts[l-1])+\
                    tf.mul(self.states[l+1], tf.transpose(tf.weights[l]))-\
                    self.pre_states
            self.next_states[i] =  self.states[i] + dXl
            # shape of states, next_states =  [[batch, size1], [ batch, size2] ,...]
            # shape of on parameter = [[size1, size2], [size2, size3], ...]
        
        self.on_paramter = hebbian(self.next_states)
        self.parameter_updates = []
        for l in range(len(weights)):
            update = tf.compat.v1.assign(weights[l], weights[l]+on_paramter[l])
            self.parameter_updates.append(update)


