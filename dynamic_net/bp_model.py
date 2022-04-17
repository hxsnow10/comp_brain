#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       bp_model
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         17/04/2022
#   Description:    ---
"""Predicitive Network.
Song, Yuhang, Thomas Lukasiewicz, Zhenghua Xu, and Rafal Bogacz. 2020. ‘Can the Brain Do Backpropagation? —Exact Implementation of Backpropagation in Predictive Coding Networks’. Advances in Neural Information Processing Systems 33: 22566–79.

Lillicrap, Timothy P., Adam Santoro, Luke Marris, Colin J. Akerman, and Geoffrey Hinton. 2020. ‘Backpropagation and the Brain’. Nature Reviews Neuroscience 21 (6): 335–46. https://doi.org/10.1038/s41583-020-0277-3.


"""

import argparse
import os
import sys


class ErrorSynpase(AbastractSynpase):

    def __init__(self, name, neurons):
        self.neurons_1, self.neurons_2 = neurons
        super(ErrorSynpase, self).__init__(name, neurons)

    def get_neuron_implacts(self):
        implact_2 = -self.neurons_1.states
        implact_1 = self.neurons_2.states
        return [implact_1, implact_2]

    def synpase_dynamics(self):
        pass


class PredicitiveNetwork(CHLFreamwork):

    def __init__(
            self,
            layer_num,
            neurons,
            synpases,
            input_inxs,
            output_inxs,
            stop_th=0.1):
        x_neurons = []
        error_neurons = []
        x_neurons.append(Neusons())
        synpases = []
        for l in range(1, layer_num):
            error_neurons.append(Neurons(1))
            x_neuronsa.append(Neurons(0))
        for l in range(1, layer_num):
            synapse = BiLinearSynpase(
                [x_neurons[l - 1], error_neurons[l]], go_factor=-1)
            synpase.append(synpase)
            synpase = ErrorSynpase([error_neurons[l], x_neurons[l]])
            synpase.append(synpase)

        neurons = x_neurons + error_neurons
        input_idxs = [0]
        output_idxs = [len(neurons) - 1]
        super(PredicitiveNetwork, self).__init__(self, neurons, synpases, input_inxs, output_inxs, stop_th=0.1):


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)
