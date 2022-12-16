#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       rarl_variants
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/11/23
#   Description:    ---
"""Real Time Recurrent Learning Variants
include  sparse-rtrl, uoro, eprop, etc.

"""

import os
import sys

import random
import argparse

import torch
from torch import nn
from rnn_step import MyOneStep
from torch.autograd import grad
from util import jacobian1 as jacobian 
def flatten_size(param):
    n=1
    for x in param.shape:
        n = n*x
    return n

class RTRL(object):

    def __init__(self, one_step_module, loss_module, optimier):
        self.one_step_module = one_step_module
        self.loss_module = loss_module
        self.lr = 0.001
   
    def apply(self, input_sequence, init_state):
        states = [(None, init_state)]
        state_size = init_state.shape[1]
        grad_of_state_param = [torch.zeros(*init_state.shape, flatten_size(param)) for param in self.one_step_module.params]
        for i, (inp, target) in enumerate(input_sequence):
            state = states[-1][1].detach()
            state.requires_grad=True
            output, new_state = self.one_step_module(inp, state, i)
            loss = self.loss_module(output, target)
            print(inp.shape, state.shape, new_state.shape, output.shape, loss.shape)
            states.append((state,new_state))
            # update forward gradient
            print(new_state)
            # 算jabian 如果使用grad 需要调N次，最好把问题调整下？ 调整不了，这个涉及核心的中间变量，不能跳过
            #torch.autograd.grad(new_state, self.one_step_module.params, grad_outputs = torch.ones_like(new_state), 
            #                                               allow_unused = True, retain_graph=True)
            # 先用random跑通
            grad_of_state_param_step = [ jacobian(new_state, param).reshape(grad_of_state_param[k].shape) for k,param in enumerate(self.one_step_module.params)]
            grad_of_state_state = jacobian(new_state, state, batch_size = 200)
            for k in range(len(grad_of_state_param)):
                print("param_shape =", self.one_step_module.params[k].shape)
                print(grad_of_state_param_step[k].shape, grad_of_state_state.shape, grad_of_state_param[k].shape)
                grad_of_state_param[k]  = grad_of_state_param_step[k] + torch.matmul(grad_of_state_state,grad_of_state_param[k])
            # compute step gradient
            grad_of_loss_state = torch.unsqueeze(torch.autograd.grad(loss,new_state, retain_graph=True)[0],1)
            print(grad_of_loss_state)
            for k in range(len(grad_of_state_param)):
                print(grad_of_loss_state.shape)
                print(grad_of_state_param[k].shape)
                gradient = torch.sum(torch.matmul(grad_of_loss_state,grad_of_state_param[k]), dim=0).reshape(self.one_step_module.params[k].shape)
                self.one_step_module.params[k].data.sub_(self.lr*gradient)

class SparseRARL(RTRL):
    def apply(self, input_sequence, init_state):
        states = [(None, init_state)]
        storch.sparse_coo_tensor(new_indicies, values)
        state_size = init_state.shape[0]
        grad_of_state_param = []
        def filter_sparse_indices(a,select_indicies):
            indicies = a.indices().permute(1,0)
            values = s.values()
            set_select_indicies = set(select_indicies.permute(1,0))
            new_indicies, values = [], []
            for k,ind in enumerate(indicies):
                if ind in set_select_indicies:
                    new_indicies.append(ind)
                    new_values.append(values[k])
            new_indicies.permute(1,0)
            rval = torch.sparse_coo_tensor(new_indicies, values)
            return rval
        for k,param in enumerate(self.one_step_module.params):
            state_param_connect_indices = self.one_step_module.state_param_connect_indices[k]
            values = torch.zeros(state_param_connect_indices.shape[1])
            grad_of_state_param_this = torch.sparse_coo_tensor(state_param_connect_indices, values, [state_size, * param.shape])
            grad_of_state_param.append(grad_of_state_param_this)
        for i, (inp, target) in enumerate(input_sequence):
            state = states[-1][1].detach()
            state.requires_grad=True
            output, new_state = self.one_step_module(inp, state, i)
            loss = self.loss_module(output, target)
            states.append((state,new_state))
            # update forward gradient
            grad_of_state_param_step = torch.autograd.grad(new_state, self.one_step_module.params, retain_graph=True)
            grad_of_state_state = torch.autograd.grad(new_state, state, retain_graph=True)
            new_grad_of_state_param = []
            for k,grad in enumerate(grad_of_state_param):
                grad = grad_of_state_param_step[i] + torch.sparse.mm(grad_of_state_state,grad_of_state_param[i])
                grad = filter_sparse_indices(grad, self.one_step_module.state_param_connect_indices[k])
                new_grad_of_state_param.append(grad)
            grad_of_state_param = new_grad_of_state_param
            # compute step gradient
            grad_of_loss_state = torch.autograd.grad(loss,new_state, retain_graph=True)
            for i in range(len(grad_of_state_param)):
                self.one_step_module.params[i]+=grad_of_loss_state*grad_of_state_param[i]

class EProp(RTRL):
    def apply(self, input_sequence, init_state):
        states = [(None, init_state)]
        state_size = init_state.shape[1]
        # 应该把所有参数连接起来，假设是一个纯粹的RNN？
        # 核心变量是dparam_out/dparam = dstate/dstate*dstate/dparam
        # 原算法假设，每个param输出一定在某个state上
        self.one_step_module.params = [param for param in self.one_step_module.params if flatten_size(param)==state_size*state_size ]
        grad_of_param_out_param = [torch.zeros(init_state.shape[0], *param.shape) for param in self.one_step_module.params] 
        for i, (inp, target) in enumerate(input_sequence):
            state = states[-1][1].detach()
            state.requires_grad=True
            output, new_state = self.one_step_module(inp, state, i)
            loss = self.loss_module(output, target)
            states.append((state,new_state))
            # update forward gradient
            grad_of_param_out_step = [ jacobian(torch.sum(new_state, dim=1), param).reshape(grad_of_param_out_param[k].shape)
                        for k,param in enumerate(self.one_step_module.params) ]
            # 这个实际上只有|param|非0
            grad_of_state_state = jacobian(new_state, state, batch_size = 200)
            # 这里可以把jacobian优化成jvp
            for k in range(len(grad_of_param_out_param)):
                print(grad_of_param_out_step[k].shape, grad_of_state_state.shape, grad_of_param_out_param[k].shape)
                grad_of_param_out_param[k] = grad_of_param_out_step[k] + torch.matmul(grad_of_state_state, grad_of_param_out_param[k].reshape(
                    init_state.shape[0], *self.one_step_module.params[k].shape))

            # compute step gradient
            grad_of_loss_state = torch.autograd.grad(loss,new_state, retain_graph=True)
            for k in range(len(grad_of_param_out_param)):
                print(grad_of_loss_state[0].shape)
                print(grad_of_param_out_param[k].shape)
                # 使用elelment-wise mul
                gradient = torch.sum(torch.unsqueeze(grad_of_loss_state[0],1).expand(*grad_of_param_out_param[k].shape)*grad_of_param_out_param[k], dim=0)
                print(gradient.shape)
                gradient = gradient.reshape(self.one_step_module.params[k].shape)
                self.one_step_module.params[k].data.sub_(self.lr*gradient)

class DirectionalGradient(RTRL):

    def apply(self, input_sequence, init_state):
        states = [(None, init_state)]
        state_size = init_state.shape[1]
        v_param = [torch.rand(*param.shape) for param in self.one_step_module.params]
        grad_of_param_pv = torch.zeros(init_state.shape[0], state_size)
        for i, (inp, target) in enumerate(input_sequence):
            state = states[-1][1].detach()
            state.requires_grad=True
            output, new_state = self.one_step_module(inp, state, i)
            loss = self.loss_module(output, target)
            states.append((state,new_state))
            # update forward gradient
            print(state.shape, new_state.shape)
            print(jacobian(new_state, state, batch_size = 200).shape)
            new_grad_of_param_pv = torch.matmul(jacobian(new_state, state, batch_size = 200),torch.unsqueeze(grad_of_param_pv,2))
            for k in range(len(v_param)):
                new_grad_of_param_pv  += jpv(param,new_state,v)# TO IMPL
            grad_of_output_state = torch.autograd.grad(loss, new_state, retain_graph=True)[0]
            for k in range(len(v_param)):
                print(grad_of_output_state.shape,grad_of_param_pv.shape,v_param[k].shape)
                gradient = torch.sum(torch.matmul(grad_of_output_state*grad_of_param_pv,v_param[k]), dim = 0)
                self.one_step_module.params[k].data.sub_(self.lr*gradient)

def main(data="", model="", train_method="directional"):
    one_step_module = MyOneStep()
    loss_module = nn.MSELoss()
    seq_len = 20
    layer_size = 50
    input_sequence = [(torch.rand(200, layer_size), torch.rand(200, layer_size))] * seq_len

    optimizer = torch.optim.SGD(one_step_module.parameters(), lr=1e-3)
    train_class = RTRL
    if train_method == "sparse_rtrl":
        train_class = SparseRTRL
    elif train_method == "eprop":
        train_class = EProp
    elif train_method == "directional":
        train_class = DirectionalGradient
    trainer = train_class(one_step_module, loss_module, optimizer)

    trainer.apply(input_sequence, torch.zeros(200, layer_size))
    print("done")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main()

