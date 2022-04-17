#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       short_long_memory
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         17/04/2022
#   Description:    ---
"""核心的目的是：实现短期记忆与长期记忆的融洽架构。

短期/工作记忆的特点是：有限性，注意力，可操作，不限于内容，流式
长期记忆



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

