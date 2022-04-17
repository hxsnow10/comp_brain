#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       chl_model
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         12/04/2022
#   Description:    ---
"""Predicitive Coding Freamwork.(explicit-error nodes)

Dynamics:
    d_{l} =x_{l}-W_{l-1} x_{l-1}
    {x}_{l} =-d_{l}+W_{l}^{T} d_{l+1}

Predict: x0 clamp to x0
Train: x0 clamp to x0, xl clamp to y. Learn by hebbian, ~ bp. 
"""

import os
import sys

import argparse

class CHLModel(object):

    def __init__():



