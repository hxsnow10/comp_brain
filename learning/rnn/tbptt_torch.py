#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       tbptt
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/11/22
#   Description:    original code is from https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/13

"""TBPTT(Truncated Backpropagation Through Time) by pytorch 
TBPTT: forward every k1 steps ， taking k2 backprop steps of sum of k1 losses
TBPTT: 2train  train是一个序列修改一次参数； train_online 是每个k1修改一次参数，但需要重复计算k2-k1个前向。
"""

import os
import sys

import argparse

import torch
from torch import nn
import time
torch.autograd.set_detect_anomaly(True)

class TBPTT():
    def __init__(self, one_step_module, loss_module, k1, k2, optimizer):
        self.one_step_module = one_step_module
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2
        # You can also remove all the optimizer code here, and the
        # train function will just accumulate all the gradients in
        # one_step_module parameters
        self.optimizer = optimizer

    def train(self, input_sequence, init_state):
            states = [(None, init_state)]

            outputs = []
            targets = []

            optimizer.zero_grad()
            for i, (inp, target) in enumerate(input_sequence):

                state = states[-1][1].detach()
                state.requires_grad=True
                output, new_state = self.one_step_module(inp, state, i)

                outputs.append(output)
                targets.append(target)
                while len(outputs) > self.k1:
                    # Delete stuff that is too old
                    del outputs[0]
                    del targets[0]

                states.append((state, new_state))
                while len(states) > self.k2:
                    # Delete stuff that is too old
                    del states[0]
                if (i+1)%self.k1 == 0:
                    # loss = self.loss_module(output, target)

                    # backprop last module (keep graph only if they ever overlap)
                    # 还是有问题这里
                    start = time.time()
                    # loss.backward(retain_graph=self.retain_graph)
                    for j in range(self.k2-1):

                        # print("j = ", j)
                        if j < self.k1:
                            loss = self.loss_module(outputs[-j-1], targets[-j-1])
                            loss.backward(retain_graph=True)
                        if states[-j-2][0] is None:
                            break

                    for j in range(self.k2-1):
                        # if we get all the way back to the "init_state", stop
                        if states[-j-2][0] is None:
                            break
                        curr_grad = states[-j-1][0].grad
                        states[-j-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                    
                    for j in range(self.k2-1):
                        if states[-j-2][0] is None:
                            break
                        print("grad of state {} = {}".format(i-j, states[-j-2][0].grad[0][0]))
                    print("bw: {}".format(time.time()-start))
            # 目前opt.step放在循环里会报错，因为计算图中的变量被修改后就不能继续求原计算的反向梯度
            # （k1=5, k2=7) 换言之，前向与反向时梯度要保持一致性。 
            # https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/54
            # 这样导致了一个问题： 参数没法实时更新！这本质上也是反向梯度的一个问题。
            # 这是因为重叠处计算了2次反向梯度。有没有可能复用第一次结果呢？

            # 另一种方法是重算一次前向！从-k2时开始。
            optimizer.step()

    def train_online(self, input_sequence, init_state):
            states = [(None, init_state)]

            outputs = []
            targets = []
            k1,k2 = self.k1, self.k2
            optimizer.zero_grad()
            i = 0
            next_bp_i = k1-1
            last_i = -1
            while i<len(input_sequence):
                print("i={}, last_i = {}".format(i, last_i))
                inp, target = input_sequence[i]

                state = states[last_i][1].detach()
                state.requires_grad=True
                output, new_state = self.one_step_module(inp, state, i)
                if last_i==-1:
                    outputs.append(output)
                    targets.append(target)
                else:
                    outputs[last_i+1] = output
                    targets[last_i+1] = target
                while len(outputs) > self.k1:
                    # Delete stuff that is too old
                    del outputs[0]
                    del targets[0]
                if last_i == -1:
                    states.append((state, new_state))
                else:
                    states[last_i+1] = (state, new_state)
                while len(states) > self.k2:
                    # Delete stuff that is too old
                    del states[0]
                if i == next_bp_i:
                    # loss = self.loss_module(output, target)

                    # backprop last module (keep graph only if they ever overlap)
                    # 还是有问题这里
                    start = time.time()
                    # loss.backward(retain_graph=self.retain_graph)
                    for j in range(self.k2-1):

                        # print("j = ", j)
                        if j < self.k1:
                            loss = self.loss_module(outputs[-j-1], targets[-j-1])
                            loss.backward(retain_graph=True)
                        if states[-j-2][0] is None:
                            break

                    for j in range(self.k2-1):
                        # if we get all the way back to the "init_state", stop
                        if states[-j-2][0] is None:
                            break
                        curr_grad = states[-j-1][0].grad
                        states[-j-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                    """
                    for j in range(self.k2-1):
                        if states[-j-2][0] is None:
                            break
                        print("grad of state {} = {}".format(i-j, states[-j-2][0].grad[0][0]))
                    """
                    print("bw: {}".format(time.time()-start))
                    next_bp_i+=k1
                    if k2<=k1:
                        i = i+1
                        last_i = -1
                    else:
                        i = next_bp_i+1-k2
                        last_i = -1-(k2-k1)
                    print("i = , last_i = ",i, last_i)
                    # 另一种方法是重算一次前向！从-k2时开始。
                    optimizer.step()
                else:
                    i = i+1
                    if last_i!=-1:
                        last_i+=1

def tbptt1_by_torch():
    """pseudocode"""
    h = init_states
    hs = [h]
    losses = []
    while True:
        h = lstm(h)
        hs.append(h)
        if len(hs)>k2:
            del(hs[0])
        loss = env_get_loss(h)
        losses.append(loss)
        if len(losses) == k1:
            loss = sum(losses)
            if len(hs)>=k2:
                # better k2>k1
                # hs[-k1-k2].disable_backward() 
                # disable some tensor backward
                # 如果k2=1, k1>k2, 这里就是动态的
                # 不能使用detach提前修改计算图
                hs[-k2].disable_backward() 
            loss.backward()
            hs[-k2].enabe_backward()
            losses = []

def tbptt2_by_torch():
    """k1=k2=K"""
    h = init_states
    k = 0
    while True:
        h = lstm(h)
        hs.append(h)
        loss = env_get_loss(h)
        loss.backward()
        k = (k+1)%K
        # disable some tensor backward static
        if k==0:
            h = h.detach()


seq_len = 20
layer_size = 50
k1 = 5
k2 = 7

class MyMod(torch.nn.Module):
    def __init__(self):
        super(MyMod, self).__init__()
        self.lin = nn.Linear(2*layer_size, 2*layer_size)

    def forward(self, inp, state, idx):
        full_out = self.lin(torch.cat([inp, state], 1))
        # out, new_state = full_out.chunk(2, dim=1)
        out = full_out.narrow(1, 0, layer_size)
        new_state = full_out.narrow(1, layer_size, layer_size)
        def get_pr(idx_val, name):
            def pr(*args):
                print("{} doing backward {}".format(name, idx_val))
            return pr
        new_state.register_hook(get_pr(idx, "state_tran"))
        out.register_hook(get_pr(idx,"output"))
        print("doing fw {}".format(idx))
        return out, new_state


one_step_module = MyMod()
loss_module = nn.MSELoss()
input_sequence = [(torch.rand(200, layer_size), torch.rand(200, layer_size))] * seq_len

optimizer = torch.optim.SGD(one_step_module.parameters(), lr=1e-3)

runner = TBPTT(one_step_module, loss_module, k1, k2, optimizer)

runner.train_online(input_sequence, torch.zeros(200, layer_size))
print("done")
