#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       log
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2023/02/17
#   Description:    ---
"""util of logging
"""

import os
import sys

import argparse
import logging

def log_info( * args, ** xargs):
    rstr = ""
    for arg in args:
        rstr+=str(arg)+" "
    logging.info(rstr)

def log_debug( * args, ** xargs):
    rstr = ""
    for arg in args:
        rstr+=str(arg)+" "
    logging.debug(rstr)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

