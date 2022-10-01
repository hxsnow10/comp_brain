#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       tp_model
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         17/04/2022
#   Description:    ---
"""Target Propagation.

Lee, Dong-Hyun, Saizheng Zhang, Asja Fischer, and Yoshua Bengio. 2015. ‘Difference Target Propagation’. ArXiv:1412.7525 [Cs], November. http://arxiv.org/abs/1412.7525.
main structure every layer:
1. h_{i+1} = f(h_i), which is a dense layer
2. h'_i = g(h_{i+1}), which hope to build the reverse function; save auto-encode-error
3. predict: compute all h_i
4. learning:
    * predict more with true target h^. compte all h^_i from target Propagation bcak
        h^_i = h_{i-1}-g_i(h_i)+g_i(h^_i) = AE-error_i + g_i(h^_i)
    * update f: minize f(h_i)-h^_i
    * update g: minize auto-encoder-error 

The dynamic of this model is also local:
    autoencoder with f,g.
    true target Propagation by g, then update by local bp(for example, predicitive coding).

what's the diiference of TP with predicitive coding??
PC 通过在网络中绑定in,ou,利用特殊的网络机制动态收敛出错误：
    仔细看，就是说如果本层的错误根据网络对应的下层的错误，就不稳定。
    if f(x+err_i)-f(x)!=err_{i+1}: err_i要做调整。
TP 同样绑定输入与输出，但是利用一个反向构建来推导出error。
感觉TP的思路更高明,直接逆函数算出来了。
"""

import os
import sys

import argparse

class ThisType(object):

    def __init__(self,):
        self


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

