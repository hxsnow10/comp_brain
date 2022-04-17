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
from hebbian import hebian


class CHLFreamwork(object):

    def __init__(
            self,
            neurons,
            synpases,
            input_inxs,
            output_inxs,
            stop_th=0.1):
        self.neurons = neurons
        self.synpases = synpases
        self.neuron_dynamic_ops = [neuron.update_op for neuron in neurons]
        self.neuron_dynamic_vals = [neuron.update_val for neuron in neurons]
        self.synpase_dynamic_ops = [synpase.update_op for synpase in synpases]
        self.anti_synpase_dynamic_ops = [
            synpase.anti_update_op for synpase in synpases]
        self.input_inxs = input_inxs
        self.output_inxs = output_inxs
        self.clamped_inxs = set([])

    def clamp_input(self, inputs):
        for inx, inp in zip(self.input_inxs, inputs):
            self.neurons[inx].set_val(inp)
        self.clamped_inxs = self.clamped_inxs | set(self.input_inxs)
        self.neuron_dynamic_ops = [
            neuron.update_op for k,
            neuron in enumerate(neurons) if k not in self.clamped_inxs]

    def clamp_output(self, outputs):
        for inx, out in zip(self.output_inxs, outputs):
            self.neurons[inx].set_val(out)
        self.clamped_inxs = self.clamped_inxs | set(self.output_inxs)
        self.neuron_dynamic_ops = [
            neuron.update_op for k,
            neuron in enumerate(neurons) if k not in self.clamped_inxs]

    def process(self, inputs, outputs):
        self.clamped_inxs = set([])
        self.clamp_inputs(inputs)
        while True:
            out_v = sess.run(
                [self.neuron_dynamic_vals, self.neuron_dynamic_ops])
            if out_v[0].amax() < self.stop_th:
                break
        sess.run(self.anit_synpase_dynamic_ops)

        self.clamp_inputs(outputs)
        while True:
            out_v = sess.run(
                [self.neuron_dynamic_vals, self.neuron_dynamic_ops])
            if out_v[0].amax() < self.stop_th:
                break
        sess.run(self.synpase_dynamic_ops)


class BiLinearSynpase(AbstractSynpase):
    def __init__(self, name, neurons, go_factor=1, back_factor=1):
        self.neurons_1, self.neurons_2 = neurons
        self.weights = tf.Variable(
            [self.neurons_1.shape[1], self.neurons_2.shape[1]], name=name)
        super(BiLinearSynpase, self).__init__(name, neurons)

    def get_neuron_implacts(self):
        implact_2 = go_factor * tf.matmul(self.neurons_1.states, self.weights)
        implact_1 = back_factor * \
            tf.matmul(self.neurons_2.states, self.weights, transpose_b=True)
        return [implact_1, implact_2]

    def synpase_dynamic(self):
        # TODO: factor and synpase_dynamic
        self.synpase_implact = hebbian(self.neurons1, self.neurons2)
        self.next_weights = self.weights + self.synpase_implact
        self.update_op = tf.assign(self.weghts, self.next_weights)

        self.anti_next_weights = self.weights - self.synpase_implact
        self.anti_update_op = tf.assign(self.weghts, self.anti_next_weights)


class CHLMLPNetwork(CHLFreamwork):

    def __init__(self, batch_size, layer_sizes):
        neurons = []
        for size in layer_sizes:
            neurons.append(Neurons(batch_size, layer_size))
        for l in range(1, len(layer_sizes)):
            synpases.append(BiLinearSynpase(neurons[l - 1], neurons[l]))
        super(CHLMLPNetwork, self).__init__(neurons,
                                            synpases,
                                            [0],
                                            [len(layer_sizes) - 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)
