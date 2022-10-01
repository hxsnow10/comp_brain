#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       network
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/09/20
#   Description:    ---
"""one line of summary

description
"""

import os
import sys

import argparse

class Network():
    """ a class to manage neurons, synpases, forward, backword
    """

    def __init__(self, neurons=[], synpases=[], activation = None, leak = 0.1, error = None, 
                 synpase_type=None, 
                 input_idxs=[]):
        self.name2neurons = {n.name:n for n in neurons}
        self.name2synpases = {s.name:s for s in synpases}
        self.activation = activation # default activation
        self.leak = leak # default activation
        self.error = None
        self.synpase_tye = synpase_type

    def check_neurons(self, process_name):
        print("start {}!!!!!!!!!".format(process_name))
        for neuron in self.name2neurons.values():
            print(neuron)


    def update_neuron(self):
        for name,synpase in self.name2synpases.items():
            print("start work synpase", name)
            synpase.neuron_dynamic()
            # TODO collect all neurons and finaly update states to next
        for neuron in self.name2neurons.values():
            neuron.forward()

    def clamp_inputs(self, inputs, input_names):
        for name, inp in zip(input_names, inputs):
            self.name2neurons[name].set_val(inp, clamped = True)
            print("after clamp", self.name2neurons[name])

    def update_synpase(self, control_lr = 1):
        for synpase in self.name2synpases.values():
            synpase.synpase_dynamic()
        for neuron in self.name2neurons.values():
            if neuron.inter_rnn_synpase:
                neuron.inter_rnn_synpase.synpase_dynamic()

    def apply(self, inputs, input_names, time_step_num, neuron_on = True, synpase_on = True):
        for t in range(time_step_num):
            self.clamp_inputs(inputs, input_names)
            if neuron_on:
                self.update_neuron()
            if synpase_on:
                self.update_synpase()

    def init_states(self):
        for synpase in self.name2synpases.values():
            synpase.init_states()
        for neuron in self.name2neurons.values():
            neuron.init_states()

    def get_val(self, name):
        return self.name2neurons[name].get_val()

    def add_synpase(self, synpases):
        if type(synpases)!=type([1,2,3]):
            synpases = [synpases]
        for synpase in synpases:
            self.name2synpases[synpase.name] = synpase
            for neuron in synpase.neurons:
                if neuron.name not in self.name2neurons:
                    print("network add neuron", neuron.name)
                    if self.activation and not neuron.activation:
                        neuron.activation = self.activation
                    if self.leak and not neuron.leak:
                        neuron.activation = leak
                    self.name2neurons[neuron.name] = neuron
        print("after add synpase")
    
    def report(self):
        for neuron in self.name2neurons.values():
            print(neuron.name, neuron.states.shape)

if __name__=='__main__':
    from neuron import *
    from synpase import *
    a = Neurons(0.01, [100,100])
    b = Neurons(0.01, [100,100])
    c = SpikingNeurons(0.01, [100,100])
    n = network()
    n.add_synpase( LinearSynpase([a,b]) )
    n.add_synpase( LinearSynpase([b,c]) )
    n.add_synpase( BiLinearSynpase([a,b]) )
    n.add_synpase( LinearBPSynpase([a,b]) )
    n.add_synpase( MseSynpase([a,b]) )

