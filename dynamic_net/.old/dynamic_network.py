#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       dynamic_network
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         12/04/2022
#   Description:    ---
"""Dynamic Network Module.

Here Dynamic Network refer to Network which is a dynamic system of time.
Formula: S_t = Net_Model(S_{t-1}), t=0,1,... infine

S: I, H, O which represent input neurons, hidden neurons, output neurons.

* t is time, not sequence index like in text.
* infinite t: t should not has max_length limit, as time never stop.
* markov dependency: H_t depend on H_{t-1} without H_{t-2}.

overall description of the module or program.

"""

import os
import sys

import argparse

class AbstractDynamicNetwork(object):
    """
    Abstract Dynamic Network Base Class.

    Attributes:
        Model: a model instace_
            input_
            hidden_states_{back,pre}
            output_states_{back,pre}
            tensor: hidden_states_back, output_states_back = 
                model_apply(hidden_states_pre, output_states_pre, input, train_flag)
            predict: input = x, train_flag = 0, repeat
            train: train_flag = 1
    """

    def __init__(self,):
        self.Model = model() 
        # these build froom model
        self.inputs = model.inputs # a map from name to tensor
        self.pre_states = model.pre_states # a map from name to tensor
        self.on_states = model.on_states # a map from name to tensor, same key of pre_states
        self.train_flag = tensor # a map from name to tensor, as phase control

        self.output_names = [] # a lst of output names

    def step_once_time(self, inputs_data, states_data, train_flag_data = 0):
        fd = {}
        for name in self.inputs:
            fd[name] = inputs_data[name]
        for name in self.pre_states:
            fd[name] = states_data[name]
        fd[self.train_flag_tensor] = train_flag_data

        out = self.on_states.keys()
        out_states_data = sess.run(
            out , feed_dict=fd,
            options=run_options,
            run_metadata=run_metadata)
        return out_states_data

    def update_parameter(self, factor = 1):
        fd = {}
        for name in self.pre_states:
            fd[name] = states_data[name]
        fd[self.train_flag_tensor] = train_flag_data
        
        out = self.model.parameter_updates
        on_parameter = sess.run(
            out , feed_dict=fd,
            options=run_options,
            run_metadata=run_metadata)

        return out_states_data

    def process_double_phase(self, inputs_data):
        """ anti hebbian like 
        """
        states_data = self.init_states_data
        train_flag_data = 0
        controler = predict_controler(states_data)
        while not controler.stop():
            states_data_next = self.step_once_time(inputs_data, states_data, train_flag_data)
            states_data = states_data_next
            controler.get(states_data)
        predict_stats = states_data
        # these two chl could be combined to one update
        self.update_parameter(-1)


        states_data = self.init_states_data # change by input_data
        controler = train_controler(states_data)
        train_flag_data = 1
        while True:
            states_data_next = self.step_once_time(inputs_data, states_data, train_flag_data)
            states_data = states_data_next
            controler.get(states_data)
        train_states = states_data
        self.update_parameter()


def main():
    """one line of summary.

    overall description.

    Args:
        ...

    Returns:
        ...

    Raise:
        ...

    """
    return None

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

