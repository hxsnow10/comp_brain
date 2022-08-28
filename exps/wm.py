#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       wm-exp
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/07/20
#   Description:    ---
"""WM EXP.

1. dataset
2. train
3. test
"""

import os
import sys

import argparse

def get_env():
    # 多任务，数据自身内容来标识任务
    # 目前这几个任务输入都是同质的符号序列
    # 复用以前的关于文本的处理工具； 但是额外增加一个转化函数。
    # TODO: 这里的函数复用了以前的组件，怎么引入最好与其他库共享

    # 加载通用的数据函数
    chars = load_words(
            vocab_path,
            vocab_skip_head,
            max_vocab_size)
    seq_p = sequence_line_processing(
            words,
            return_length=config.use_seq_length,
            seq_len=config.seq_len,
            split=config.split,
            sub_words=sub_words,
            char_len=config.char_len)
    # stm回忆的任务: seq1-seq2 任务
    # 把seq2与y_true对比
    line_p = split_line_processing([label_p, label_p])

    train_data = LineBasedDataset( train_paths, line_processing=line_p, batch_size=config.batch_size)
    test_data = LineBasedDataset( test_paths, line_processing=line_p, batch_size=config.batch_size)
    # 1. 把x向量转为化时间序列输入，持续一段时间
    # 2. 提问, 持续一段时间，收集回答
    # 3.1 根据回答反馈回报?  但是这样回报太稀疏了
    # 3.2 不如把y_true作为输入，内建回报计算模块
    stm_train_env = SeqEnv(train_data, timeidx, question=None)
    stm_test_env = SeqEnv(test_data, timeidx, question=None)

    # 情感分析监督任务: seq-cls[,prob] 任务
    label_p = label_line_processing(tags)
    line_p = split_line_processing([label_p, seq_p])
    train_data = LineBasedDataset( train_paths, line_processing=line_p, batch_size=config.batch_size)
    test_data = LineBasedDataset( test_paths, line_processing=line_p, batch_size=config.batch_size)
    question=seq_p("Q:tell the setniment of sent")
    sentiment_train_env = SeqEnv(train_data, timeidx, question=question)
    sentiment_test_env = SeqEnv(test_data, timeidx, question=question)

    # 无监督序列的任务； Predicitive-coding
    train_data = LineBasedDataset( train_paths, line_processing=line_p, batch_size=config.batch_size)
    test_data = LineBasedDataset( test_paths, line_processing=line_p, batch_size=config.batch_size)
    lang_train_env = SeqEnv(train_data, timeidx, question=None)
    lang_test_env = SeqEnv(test_data, timeidx, question=None)

    # 多轮对话,推理
    train_env = MergeEnv([stm_train_env, sentiment_train_env, lang_train_env])
    test_env = MergeEnv([stm_test_env, sentiment_test_env, lang_test_env])
    return train_env, test_env

def get_mrnn_agent():
    # 结构上是经典的RNN seq-seq的模式
    # 难点在于学习的实现，需要解决BPTT的问题。
    x = Neurons(size = 1, error=True) # char input
    target = ForwardNeurons(error=True)
    y = Neurons(size = 1, error=True) # char output
    y_true = Neurons(error=True)
    network = Network([x, target, y, y_true], activation = tf.relu)
    hidden_layer_sizes = [int(x.size/2), int(x.size/4)]
    synpase_type = BPSynpase
    rnn_synpase_type = BPSynpase
    network.add_synpase(MseTSynpase("tsynpase", [x_neurons[layer_num-1], y_true, target], target_name ="target"))
    network.add_synpase(general_mlp(x,y, synpase_type, rnn_synpase_type, error2states = 0.1))
    agnet = Agent(network)
    return agent

def test_agent(agent, test_env):
    """ 动力学网络与一般的的计算图不同的地方在于：y 是一个时间序列，而不是一个值。
    因此，loss也不是一个值，而是一个时间序列。
    在测试的时候，我们与训练时候类似的，在第二阶段，给予y，观察loss
    即测试与训练的接口一致：观察当y出现时loss的平均值的变化。
    """
    agent.work(test_sample) # without train

def main():
    # 任务
    train_env, test_env = get_env()
    # Agent: MRNN, HumanLikeAgent
    agent = get_mrnn_agent()
    for epoch in range(10):
        agent_work(agent, train_env, epoch=1, neuron_on=True, synpase_on=True)
        agent_work(agent, mtest_env, epoch=1, neuron_on=True, synpase_on=False)
    # EVAL

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

