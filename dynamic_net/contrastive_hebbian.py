#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       chl
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         17/04/2022
#   Description:    ---
"""Constrative Hebbian Learning Model.
Xie, Xiaohui, and H. Sebastian Seung. 2003. ‘Equivalence of Backpropagation and Contrastive Hebbian Learning in a Layered Network’. Neural Computation 15 (2): 441–54. https://doi.org/10.1162/089976603762552988.

Model: Multi biconnect Layer with symmetry weight: x_l,W_l,x_{l+1}
Neuron Dynamics:
    d{x}_{l}=W_{l-1} x_{l-1}+W_{l}^{T} x_{l+1}-x_{l}

Synpase Dynamics:
    1. clamp x0=x, run until converge
    2. update by anti-hebbian
    3. camp c0=x, xl=y, run until converge
    4. update by hebbian

TODO:
    * while no bias and nolinear ?
    * why symmetry?feedback random weights
    * how about recurrent?
"""

import argparse
import os
import sys

from base import Neurons, Synpase

class CHLFreamwork(object):

    def __init__(
            self,
            neurons,
            synpases,
            input_inxs,
            output_inxs,
            stop_th=0.1):
        self.net = Network(neurons, synpases)

    def process(self, inputs, outputs, input_inxs, output_inxs):
        self.net.clamp_inputs(inputs, input_inxs)
        while True:
            net.update_neurons() 
            if out_v[0].amax() < self.stop_th:
                break
        net.update_synpase(-1) 

        self.net.clamp_inputs(outputs, output_inxs)
        while True:
            net.update_neurons() 
            if out_v[0].amax() < self.stop_th:
                break
        sess.run(self.synpase_dynamic_ops)
        net.update_synpase(1) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)
