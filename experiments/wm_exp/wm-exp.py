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

def main():
    # 多任务，数据自身内容来标识任务
    # 目前这几个任务输入都是同质的符号序列
    # stm回忆的任务
    stm_env = StmEnv(stm_data_path)
    # 无监督序列的任务； Predicitive-coding
    language_env = LanguageEnv()
    # 情感分析监督任务 
    sentiment_env = SentimentEnv()
    # 多轮对话
    train_env, test_env = MergeEnv([stm_env])
    
    # Agent
    agent = HumanBrainAgent()
    agent.work(train_env)

    # EVAL

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

