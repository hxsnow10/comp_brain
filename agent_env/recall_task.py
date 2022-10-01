#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       recall_task
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/07/22
#   Description:    ---
"""recall_env

复用以前的dataset？ 并不太能，多了个时间维度。
数据形式sample [(t,s_t),...]
"""

import os
import sys

import argparse

class RecallEnv(object):

    def __init__(self, data_path, batch_size=50):
        self.data_path = data_path
        self.input = open(self.data_path)
        # TODO add shuffle
        self.t = -1
        self.k = 0
        self.batch_size = self.batch_size

    def reset(self):
        try:
            sents = self.input.readline(self.batch_size)
        except:
            self.input = open(self.data_path)
            sents = self.input.readline(self.batch_size)
        self.sents = sents

    def step(self, action = None):
        self.t=self.t+1
        rval = []
        for k in range(len(self.sents)):
            if self.t<len(self.sents[k]):
                rval.append(self.sents[k][self.t])
            else:
                rval.append(None)
        return rval
        # TODO： return like gem

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

