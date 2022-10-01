#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       cog_arch
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/06/18
#   Description:    ---
"""Coginive Arch HighLevel Iterface.

## Interface
agent = CognitiveArch()
agent.learn(tasks)
agent.learn(env)

## System description
preception: x->understand->out_world_map
working_memory：RNN(h = in_world, i/o = out_world_map)
world_map: out_world_map + in_world_map
predict_next_world: predict on world_map
reawrd_system: eval value of state, value of  predict_state
decision_system: find best action to max fututre value.
"""

import os
import sys
from enum import Enum

import argparse

class WorkMode(Enum):
    Normal = 1
    Predict = 2
    Train = 3
    AntiTrain = 4

class Agent(object):
    """agent with preception and action maxmizing some target.
    here we discrete time as default.
    """
    def __init__(self, network= None, attract_time_step_num = 10):
        self.network = network
        self.build_network()
        self.attract_time_step_num = attract_time_step_num
        self.network.check_neurons("agent init")
    
    def work_step(self, obs, obs_names, neuron_on = True, synpase_on = True):
        """
        agent work
            env_t:{name1:tensor1}
            mode: phase control signal
        """
        # TODO here
        self.network.apply(obs, obs_names, self.attract_time_step_num, neuron_on, synpase_on)
        action = self.network.get_val("y")
        return action
    
    def init_states(self):
        """状态复原"""
        self.network.init_states()

    def build_network(self):
        """build a self.network obj  has :
            * neuron_update: forward phase, return action
            * synpase_update: learning phase
        """
        pass

    def work(self, env, obs_names, env_seed=None, neuron_on = True, synpase_on = True, max_step=1000):
        """in faced of env, return action with side effect of self-change.
        """
        self.network.check_neurons("start work")
        obs, info = env.reset(seed=env_seed, return_info=True)
        for i in range(max_step):
            action = self.work_step(obs, obs_names, neuron_on, synpase_on)
            # 确保向量action可以被env理解
            obs, reward, done, info = env.step(action)
            if done:
                break

    def get_val(self, name):
        return self.network.get_val(name)

def agent_work(agent, env, obs_names, max_step, neuron_on, synpase_on):
    for step in range(max_step):
        agent.init_states()
        agent.work(env, obs_names, neuron_on = neuron_on, synpase_on = synpase_on)
        # 获得收敛时刻的target
        # 为了防止一些自我修正的算法，使用最大值是不是更好
        target+=agent.get_val("target")
        target_num+=1
        if target_num%10000==1:
            print(target)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

