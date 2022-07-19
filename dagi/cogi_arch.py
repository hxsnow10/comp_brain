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

from ..dynamic_net.base import *

class WorkMode(Enum):
    Normal = 1
    Predict = 2
    Train = 3
    AntiTrain = 4

class Agent(object):
    """agent with preception and action maxmizing some target.
    here we discrete time as default.
    """
    def __init__(self):
        self.build_network()
    
    def work_step(self, env_t, mode):
        action = None
        if mode==WorkMode.Normal:
            # 智能体正常模式，在线学习，预测与学习同步进行
            action = self.network.neuron_update()
            self.network.synpase_update()
        elif mode==WorkMode.Predict:
            action = self.network.neuron_update()
        elif mode==WorkMode.Train:
            self.network.synpase_update()
        return action

    def build_network(self):
        """build a self.network obj  has :
            * neuron_update: forward phase, return action
            * synpase_update: learning phase
        """
        pass

    def work(self, env, env_seed, mode=WorkMode.Normal, max_step=1000):
        """in faced of env, return action with side effect of self-change.
        """
        observation, info = env.reset(seed=env_seed, return_info=True)
        for i in range(max_step):
            action = agent.work_step(env_step, mode)
            # 确保向量action可以被env理解
            observation, reward, done, info = env.step(action)
            if done:
                break

class HumanBrainAgent(Agent):
    """基于对大脑智能仿真的智能体，关注内部模块
    核心概念：动力学；目标系统；决策规划；短期记忆；注意；学习；图像；自然语言；
    TODO : 考虑一些非计算的特性，比如注意力（能量的分配），大脑疲劳机制，多巴胺失调，做梦
    """

    def build_network():
        network = Network(activation = tf.relu, leak = 0.1)

        # reward
        target = Neurons(1)
        sub_targets = Neurons(1000, self_dynamics=rnn_train)
        network.link(sub_targets, target, "mlp:2", layer_sizes = [] )

        # lang-working-memroy
        lang_state = Neurons(1000000, self_dynamics=rnn_train)
        network.link(lang_state, sub_target, "mlp:3", layer_sizes = [])

        # vision-working-memory
        vision_state = Neurons(1000000, self_dynamics=rnn_train)
        network.link(vision_state, sub_target, "mlp:3", layer_sizes = [])

        # multi-model
        networki.link(vision_state, lang_state, "mlp:3", layer_sizes = [])
        networki.link(lang_state, vision_state, "mlp:3", layer_sizes = [])

        # 无历史的网络有2种形式：1）直接使用tf的算子 2）把neuron的leak设置为1
        input_vision = Neurons(1000, leak_init=1)
        model =  tf.layers.cnn()
        def CNN(states, weights):
            return model(states)
        network.link(input_vision, vision_state, CNN)
        
        input_phone = Neurons(1000, leak_init=1)
        input_char = Neurons(1000, leak_init=1)
        model =  tf.layers.rnn()
        def RNN(states, weights):
            return model(states)
        network.link(input_phone, lang_state, RNN)

        # top-down attention
        
        # attention, conscioueness, energy

        # policy-based decision
        actions = Neurons(1000,leak_init = 1)
        network.link(state, actions, "mlp:3")
        # forward could be either graph or dynamics based. equally.
        # Q(V(s,a)) Q(V(s,[a1,a2,..])) 都涉及到了时序过程，短期记忆的参与，而不是独立的结构
        # 大脑是否有某个东西来指导包括这个思考过程呢？？即短期记忆的思考过程
        # 换言之. planning是被转化为RNN参数而非显式编码的
        
        # 决策结果应该可以影响思考过程
        uncerntaty = F(actions)
        network.link(actions, state, "mlp:3")
        network.link(sub_targets, state, "mlp:3")
        
        # 如果想象中的高价值的状态，应该把它的想象行为加强置信,这时行为应该已经触发了但未表达
        network.link(targets, actions, "mlp:3")

        self.network = network

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

