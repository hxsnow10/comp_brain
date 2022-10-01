#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       env
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/06/18
#   Description:    ---
"""High Level Interface of Env and Task.

"""

import os
import sys

import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

