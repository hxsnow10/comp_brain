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
agent.process(env)

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
    """agent process with env maxmizing some target.
    """
    def __init__(self, network = None, obs_names = None, action_name = None, network_attract_step_num = 10):
        self.network = network
        self.build_network()
        # 网络用以收敛的步伐次数, 即每个输入时间网络需要循环的时间次数
        self.network_attract_step_num = network_attract_step_num
        self.obs_names = obs_names
        self.action_name = action_name
        self.network.freeze_weights = False

    def set_mode(self, mode):
        """
        工作模式的划分涉及以下几个子模块
        1. 允许前向预测
        2. 允许错误信号的传播
        3. 允许参数修改
        
        RNN的训练：
        online-learning，每个时间步伐 123都打开
        offline的话，即标准的反向梯度学习，一般是12 运行完一个/一族序列后，再整体修改参数
        预测的时候：23都可以关掉
        """
        if mode == OnlineLearning:
            self.network.freeze_weights = False
        elif mode == OfflineLearning:
            self.network.freeze_weights = False
        elif mode == Predict:
            self.network.freeze_weights = True
    
    def init_states(self):
        """状态复原"""
        self.network.init_states()

    def step(self, obs, obs_names, action_names):
        """
        agent work
            env_t:{name1:tensor1}
            mode: phase control signal
        """
        rval = self.network.apply(obs, input_names = obs_names, 
                                  output_names = action_names,
                                  time_step_num = self.network_attract_step_num)
        logging.debug("step result = {}".format(rval))
        return rval
    
    def process(self, env, obs_names=None, action_names=None, env_seed=None, max_step=100):
        """in faced of env, return action with side effect of self-change.
        输入obs与action的名字是因为复杂环境与智能体的输出信号都是多维的，
        不同任务涉及到不同的输入与输出。
        """
        obs_names = obs_names or self.obs_names
        action_names = action_names or self.action_names
        self.network.check_neurons("start work, neurons = ")
        action = None
        obs, info = env.reset(seed=env_seed, return_info=True)
        for i in range(max_step):
            logging.debug("i={}, obs = ".format(i, obs))
            action = self.step(obs, obs_names, action_names)
            logging.debug("i={}, action = ".format(action))
            # 确保向量action可以被env理解
            obs, reward, done, info = env.step(action, action_names)
            if done:
                break
        logging.debug("get result = {}".format(action))
        return action

    def get_val(self, name):
        return self.network.get_val(name)

def agent_work(agent, env, obs_names, loop_num, loop_step_num = 100):
    for step in range(loop_num):
        agent.init_states()
        action = agent.process(env, obs_names, max_step = loop_step_num)
        # 获得收敛时刻的target
        # 为了防止一些自我修正的算法，使用最大值是不是更好
        # TODO

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

