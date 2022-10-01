#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       utils
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/21
#   Description:    ---
"""one line of summary

description
"""
import torch

class PytorchData(object):

    def __init__(self, data, transform = None):
        self.data = data
        self.size_of_epoch = len(data)
        self.iter = iter(self.data)
        self.transform = transform

    def __iter__(self):
        while True:
            return self__next__()

    def __next__(self):
        try:
            item = next(self.iter)
        except Exception as e:
            self.iter = iter(self.data)
            item = next(self.iter)
        if self.transform:
            item = self.transform(item)
        return item


def lstr(x):
    s = ""
    if type(x)==list:
        ll = [lstr(v) for v in x]
        s = "["+",".join(ll)+"]"
    elif issubclass(type(x),torch.Tensor):
        s = "tensor("+str(x.shape)+")"
    else:
        s = str(x)
    return s

