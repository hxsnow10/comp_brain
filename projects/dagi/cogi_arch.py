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

class HumanBrainAgent(Agent):
    """基于对大脑智能仿真的智能体，关注内部模块
    核心概念：动力学；目标系统；决策规划；短期记忆；注意；学习；图像；自然语言；
    TODO : 考虑一些非计算的特性，比如注意力（能量的分配），大脑疲劳机制，多巴胺失调，做梦


    ### 学习
    学习，我认为一定存在某种目标。对于模块或者每个突触都是这样。
    我希望在接口层面，可以定义目标；实现层面，保证目标的学习通过局部动力学实现。
    """

    def build_network():
        synpase_type = type("BaseSynpase", (LinearSynpase, ErrorBPSynpase))
        network = Network(activation = "relu", leak = 0.1, error = True, synpase_type = synpase_type)

        # reward
        target = Neurons(1)
        sub_targets = Neurons(1000, self_dynamics=rnn_train)
        network.link(sub_targets, target, "mlp:2", layer_sizes = [] )
        # outer reward
        input_reward = Neunrons(1)
        # 应该认识到强化学习中，行为与回报不存在显式的计算图。
        # td_learning 根据预估与实际的error来学习minize(|v(s')+r-q(s,get_action(s))|)
        # TODO: implament AC error
        # Q值，多巴胺
        # 方法1：简单维护一个q(s,a)函数
        # 方法2: 保存一个s'=P(s,a)的副本呢？然后再创建一个target的副本
        # 方法3：不创建副本，而是通过时序，通过想象s'，自然生成Q值
        future_target = Neurons(1)
        network.link([state, actions],future_target, "mlp:3")
        last_future_target = Neurons(1)
        network.link(future_target, last_future_target, "mlp:1")
        # 注意到上一次的future_target与target的差形成训练信号
        msesynpase([last_future_target,target+input_reward])
        # 把这里的机制封装起来

        # lang-working-memroy
        lang_state = Neurons(1000000, self_dynamics=rnn_train)
        network.link(lang_state, sub_target, "mlp:3", layer_sizes = [])

        # vision-working-memory
        vision_state = Neurons(1000000, self_dynamics=rnn_train)
        network.link(vision_state, sub_target, "mlp:3", layer_sizes = [])

        # multi-model
        networki.link(vision_state, lang_state, "mlp:3", layer_sizes = [])
        networki.link(lang_state, vision_state, "mlp:3", layer_sizes = [])

        # 无历史的网络有2种形式：1）直接使用深度框架算子 2）把neuron的leak设置为1
        input_vision = Neurons(1000, leak_init=1)
        model =  cnn()
        def CNN(states, weights):
            return model(states)
        network.link(input_vision, vision_state, CNN)
        # learning: pc
        
        input_phone = Neurons(1000, leak_init=1)
        input_char = Neurons(1000, leak_init=1)
        model =  rnn()
        def RNN(states, weights):
            return model(states)
        network.link(input_phone, lang_state, RNN)
        # learning: pc

        # top-down attention
        
        # attention, conscioueness, energy

        # Module: Decision
        # policy-based decision
        actions = Neurons(1000,leak_init = 1)
        network.link(state, actions, "mlp:3")

        # Planning
        # Q(V(s,a)) Q(V(s,[a1,a2,..])) 都涉及到了时序过程，短期记忆的参与，而不是独立的结构
        # 大脑是否有某个东西来指导包括这个思考过程呢？？即短期记忆的思考过程
        # 换言之. planning是被转化为RNN参数而非显式编码的

        # 决策结果（包括置信度) 应该可以影响思考过程
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

