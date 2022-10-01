#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       bptt
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/07/01
#   Description:    ---
"""BPTT variation.

* TBPTT
* BPTT with memory and attention
"""

import os
import sys

import argparse

class ThisType(object):

    def __init__(self,):
        self


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

