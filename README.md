# PyTorch implementation of Meta-Constrained Policy Optimization (Meta-CPO)
This repository is an adaptation of the [CPO algorithm](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), transforming it into a Meta-learning framework. The modification involves leveraging Differentiable Convex Programming to facilitate the relaxation of gradient computations between parameters. The integration of CPO into the meta-learning framework was achieved through the application of the model-free meta-framework introduced by [MAML](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf). The primary objective of this algorithm is to undergo testing within the [Safety Gymnasium][https://github.com/PKU-Alignment/safety-gymnasium], offering an intuitive experimental platform to showcase its effectiveness in the context of Autonomous Driving tasks. For the detailed theory with regard to achieve safety guarantee, please refer to our paper[Constrained Meta-RL with DCO](https://arxiv.org/pdf/2312.10230.pdf) 

## Pre-requisites
- [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)
- [Model-Agnostic Meta-Learning](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
- [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- [Differentiable Convex Programming](https://locuslab.github.io/2019-10-28-cvxpylayers/)

### Usage
Develop your meta-learning environments and store each one in the **_envs_** list within the main.py script. Subsequently, execute the code to observe the functionality of your implemented environments.

### Code Reference
* [Khrylx/PyTorch-RL](https://github.com/Khrylx/PyTorch-RL)
* [SapanaChaudhary/PyTorch-CPO](https://github.com/SapanaChaudhary/PyTorch-CPO)


