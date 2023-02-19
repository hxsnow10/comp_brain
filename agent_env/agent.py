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
import logging

class Agent(object):
    """agent with preception and action maxmizing some target.
    here we discrete time as default.
    """
    def __init__(self, network= None, network_attract_step_num = 10):
        self.network = network
        self.build_network()
        # 网络用以收敛的步伐次数
        self.network_attract_step_num = network_attract_step_num
        self.network.check_neurons("agent init done, neurons = ")
        self.network.freeze_weights = False
   
    def freeze_weights(self):
        # 工作模式的划分有以下几个子管道
        # 1. 允许前向预测
        # 2. 允许错误信号的反向传播
        # 3. 允许参数修改
        # 正常情况（训练），所有管道都打开；预测时，为了避免对预测的数据的训练，需要关闭2与3；
        # 现实，当然不存在绝对的预测时，可以把所有管道都打开。
        self.network.freeze_weights = True

    def unfreeze_weights(self):
        self.network.freeze_weights = False

    def work_step(self, obs, obs_names):
        """
        agent work
            env_t:{name1:tensor1}
            mode: phase control signal
        """
        # TODO here
        rval = self.network.apply(obs, input_names = obs_names, time_step_num = self.network_attract_step_num)
        logging.debug("step result = {}".format(rval))
        input(rval)
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

    def work(self, env, obs_names, env_seed=None, max_step=100):
        """in faced of env, return action with side effect of self-change.
        """
        self.network.check_neurons("start work, neurons = ")
        action = None
        obs, info = env.reset(seed=env_seed, return_info=True)
        for i in range(max_step):
            logging.debug("i={}, obs = ".format(i, obs)) 
            action = self.work_step(obs, obs_names)
            logging.debug("i={}, action = ".format(action)) 
            # 确保向量action可以被env理解
            obs, reward, done, info = env.step(action)
            if done:
                break
        input("get result = {}".format(action))
        return action

    def get_val(self, name):
        return self.network.get_val(name)

def agent_work(agent, env, obs_names, loop_num, loop_step_num = 100):
    for step in range(loop_num):
        agent.init_states()
        action = agent.work(env, obs_names, max_step = loop_step_num)
        # 获得收敛时刻的target
        # 为了防止一些自我修正的算法，使用最大值是不是更好
        # TODO

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

