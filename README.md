# comp_brain
基于现代tf/pytorch等可微计算框架构建的计算神经学/类脑计算的软件库。
包括：
+ 基于neuron、synpase的编程架构
+ 记忆：短期/工作记忆、长期记忆、复述
+ 学习：BackProp、Meta-Learning、CHL、TargrtProP、EquilibriumProp等；
+ 决策与规划
+ 综合各个模块，系统层面深度通用智能体DAGI

## 文档
* [DAGI通用人工智能设计](https://www.wolai.com/xiahong/2g2eh12fjzr2iPUFMzwkjr)

## 代码说明
```
.
├── LICENSE
├── README.md
├── RELEASE.md
├── TODO.md
├── agent_env                           agent与env的相关类、方法
│   ├── agent.py
│   ├── env.py
│   ├── env_from_data.py
│   ├── recall_data.txt
│   ├── recall_task.py
│   └── recall_task_gen.py
├── learning                            学习的模型
│   ├── bptt.py
│   ├── contrastive_hebbian.py
│   ├── equilibrium_propagation.py
│   ├── predictive_coding.py
│   ├── target_propagation.py
│   └── temporal_predicitive_coding.py
├── memory
│   └── embedded-memory.py
├── network                             核心基础模块：神经元、突触、网络
│   ├── network.py
│   ├── neuron.py
│   └── synpase.py
├── op                                  常用的算子封装
│   └── ops.py                  
└── projects                            实验
    ├── bp.py
    ├── bptt.py
    ├── dagi
    ├── utils.py
    └── wm.py
```

