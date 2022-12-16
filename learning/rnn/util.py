#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       util
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/12/01
#   Description:    ---
"""one line of summary

description
"""

import os
import sys

import argparse
import torch

def jacobian1(outputs, inputs, batch_size = None):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    if not batch_size:
        return torch.rand(*outputs.shape, *inputs.shape)
    else:
        return torch.rand(batch_size, *outputs.shape[1:], *inputs.shape[1:])

    assert inputs.requires_grad
    print(outputs.shape, inputs.shape)

    num_classes = outputs.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*outputs.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        if inputs.grad is not None:
            inputs.grad.zero_()
        grad_output.zero_()
        grad_output[:, i] = 1
        #outputs.backward(grad_output, retain_graph=True)
        #jacobian[i] = inputs.grad.data
        jacobian[i] = grad()

    return torch.transpose(jacobian, dim0=0, dim1=1)


"""
def jacobian2(outputs, inputs, create_graph=False):
    jac = outputs.new_zeros(outputs.size() + inputs.size()).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

     return jac.view(outputs.size() + inputs.size())
 """
