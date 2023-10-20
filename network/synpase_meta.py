#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       synpase_meta
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/12/22
#   Description:    ---
"""基于元学习的突触

"""

import os
import sys

import argparse
class MLPMetaWDynamicSynpase(Synpase):
    """ 元突触
    meta for synpase update, 不显式使用error， meta 学习使用BP
    必须要包含 元网络构造、元学习方法
    本synpase同时包含2种突触动力学，元动力学是为了元学习，一般突触动力学是为了表达近似算法。
    """

    def set_meta_learning(self, meta_synpase_type = None, mlp_srags):
        self.meta_synpase_type = meta_synpase_type
        # 这里不仅要形成网络，还要考虑外部网络与元网络的学习信号的传递
        # 我们可以使用传统的非生物可行的BP算法，在BP里使用了error
        # 但在代码中不显式.
        # 注意到：这里的学习过程 是学习对象， 目标是更快更好地学习。
        self.w_neuron = Neurons(states_ref = self.weights, leaky=1)
        # 不设置leaky，就是把历史存在内部，设置leaky相当于引入预设
        # 注意到这里weights、delta_weights变成了neurons，形成了网络！
        self.delta_w = Neurons(w.states())
        input_n = [self.neurons, self.w_neuron]
        self.delta_error = Neurons()
        self.meta_synpase = general_mlp(self.meta_synpase_type, input_n, [delta_w], **mlp_args)
        self.sub_synpases.append(meta_synpase)
        # 这里delta_w与梯度的关系也可以是预设的，而把元学习约束到梯度传播上 TODO

    def synpase_dynamic_imp(self):
        self.meta_synpase.inference_dynamic()
        self.meta_synpase_type.synpase_dynamic_imp()
        # TODO add params
        return self.delta_w.states

class MLPMetaErrorSynpase(ErrorBPSynpase):
    """ 使用显式error的元突触
    meta for synpase update, 显式使用error, meta 使用 neuron 的近似方法
    另一种方法是引入显式error，1）F:input_n->error_implact, dT/dx = error_x 2) error_x->delta_w(error_w)
    1通过metalearning实现，2可以通过某种预设的形态实现最大反向梯度  delta_w = alpha * dx/dw * error_x
    TODO: 这里还有个问题，error_implact的error信号怎么来，我们假设F具有近似的BP算法
    error_implact改变了error继而产生delta_w改变w
    i) 如果我们知道w变化后的T的变化，至少可以打压T变差的过程，加强T变好的过程
    ii) 是否可以把一次改变变得最好(one step best)作为优化目标 Equally Min T2, 所以我们其实在问dT2/d{delta_w}
    iii) error_implact的error由delta_w反向传递获得。dT2/d{delta_w} = dT2/dw2*dw2/d{delta_w} = dT2/dw2
    
    假如我们通过元网络得到了dT2/dw2 = sum(dT2/dy * dy/dw2)，就可以用这个信号来学习元网络。
    注意到dT/dw 一定是BPTT的总和，这本身就是我们的学习目标。。  我这里似乎混乱了。。
    dT/d{error_implact} = dT/d{error_x} = 1/(alpha * dx/dw)*dT2/d{deta_w} = 1/(alpha * dx/dw) * dT/dw  
     =  1/(alpha * dx/dw) * dT/dx * dx/dw  = 1/alpha * dT/dx
     感觉得再推导一次。。。
    """
    
    def set_meta_learning(self, meta_synpase_type = None, mlp_args):
        self.meta_synpase_type = meta_synpase_type
        self.error_implact = Neurons(w.states())
        input_n = [self.neurons, self.w_neuron, Neuons(self.neurons.error)]
        self.meta_synpase = general_mlp(input_n, [error_implact], **mlp_args)
        self.sub_synpases.append(meta_synpase)
    
    def neuron_error_dynamic_imp(self):
        self.meta_synpase.inference_dynamic()
        # 关于error_implact的error怎么生成的问题
        # self.error_implact.error += 1/alpha * error_y
        return self.error_implact.states

    # TODO: implent e-prop


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

