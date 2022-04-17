#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       hebbian
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         16/04/2022
#   Description:    ---
"""Hebbian Learning

"""

import os
import sys

import argparse


def hebbian(states1, states2):
    """
    shape of states1/2=  [batch, size]
    shape of return = [[size1, size2], [size2, size3], ...]
    """
    rval = if.matmul(tf.expand_dims(states1,2),tf.expand_dims(states2,1))
    return rval


