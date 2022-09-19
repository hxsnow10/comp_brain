#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       kernel
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/19
#   Description:    ---
"""wrap tensorflow、pytorch

"""

import tensorflow as tf


import os
import sys

import argparse

kernel = "torch"

def set_kernel( nkernel):
    global kernel
    kernel = nkernel

def get_variable(init_states = 0, name = None):
    if kernel=="tf":
        val = tf.Variavle(init_states, name = name)
    elif kernel=="torch":
        val = torch.tensor(init_states, requires_grad = False)
    return val

def get_activation(name):
    if name=="relu" and kernel=="tf":
        return tf.relu
    if name=="relu" and kernel=="torch":
        return torch.nn.relu

def get_zeros_like(states)
    if kernel=="tf":
        return tf.zeros_like(states)
    else:
        return torch.zeros_like(states)

def assign_add(val_a, val_b):
    if kernel=="tf":
        tf.compat.v1.assign_add(self.states, val)
    else:
        val_a+=val_b

def cast_type(val, ttype = "int32")
    if ttype="int32":
        if kernel == "tf"
            return tf.cast(val. tf.int32)
        else:
            return val.type(torch.IntTensor)

def get_expand_dims(val, idx)
    if kernel =="tf":
        return tf.expand_dims(val , idx)
    else:
        return val.unsqueeze(idx)

def get_gradient(val_a, val_b):
    if kernel =="tf":
        return tf.gradient(val_a, val_b)
    else:
        return torch(val_a, val_b)

def get_sum(val)
    if kernel =="tf":
        return tf.sum(val)
    else:
        return torch.sum(val)

def get_matmul(a, b, transpose_a=True, transpose_b = True):
    # TODO: transpose
    if kernel == "tf":
        return tf.matmul(a,b, transpose_a = transpose_a, transpose_b = transpose_b)
    else:
        aa = a
        bb = b 
        if transpose_a: aa=torch.transpose(a)
        if transpose_b: bb=torch.transpose(b)
        return torch.matmul(aa,bb)

def get_transpose(a):
    if kernel =="tf":
        return tf.transpose(a)
    else:
        return torch.transpose(a)

def get_mse(a, b):
    if kernel =="tf":
        return tf.keras.metrics.mean_squared_error(a,b)
    else:
        return torch.nn.functional.mse_loss(a,b)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

