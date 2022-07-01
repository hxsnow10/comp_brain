#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       chl_model
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         12/04/2022
#   Description:    ---
"""Equilibrium Propagation.

EP看起来就是CHL的一个变形，β-无穷-clamp变成了β-Weakly Clamped。
但是这种变形不是说直接对y进行变化，而是对能量函数进行改变，F := E + βC，C是对y-mse，
E是hopefild能量函数。E实际上表明了自身的融洽性，稳定性，在这个基础上再考虑可预测性。

强烈怀疑E的稳定与hebbian这种学习是等价的。把E拆解到局部上，E的方向必定等价某种动力学。
"""

import os
import sys

import argparse

class CHLModel(object):

    def __init__():



