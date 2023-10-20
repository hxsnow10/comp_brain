#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       env
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/07/26
#   Description:    ---
"""Util about Env

"""

import os
import sys

import argparse
import random

def sampling_idx(sampling_prob, n):
    if not sampling_prob:
        return random.randint(n)
    else:
        s = sum(sampling_prob)
        sampling = [x/s for x in sampling_prob]
        f = random.random()
        ff,idx = 0,-1
        while ff<f:
            idx+=1
            ff+=sampling_prob[idx]
        return idx

class MergeEnv(object):

    def __init__(self, envs, sampling_prob):
        self.envs = envs
        all_names = list(set(sum([env.names for env in envs])))
        self.env_index_in_all = [[all_names.index(name) for name in env.names] for env in envs]
        self.env_on_idx = None
        self.sampling_prob = sampling_prob

    def reset(self):
        self.env_on_idx = sampling_idx(sampling_prob, len(envs))
        self.envs[self.env_on_idx].reset()

    def step(self, action):
        result = self.envs[self.env_on_idx].step(action)
        # TODO:update by all_names
        return result

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

