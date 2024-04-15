# PyTorch implementation of Meta-Constrained Policy Optimization (Meta-CPO)
This repository is an adaptation of the [CPO algorithm](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), transforming it into a Meta-learning framework. The modification involves leveraging Differentiable Convex Programming to facilitate the relaxation of gradient computations between parameters. The integration of CPO into the meta-learning framework was achieved through the application of the model-free meta-framework introduced by [MAML](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf). The primary objective of this algorithm is to undergo testing within the [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium), offering an intuitive experimental platform to showcase its effectiveness in the context of Autonomous Driving tasks. For the detailed theory with regard to achieve safety guarantee, please refer to our paper [Constrained Meta-RL with DCO](https://arxiv.org/pdf/2312.10230.pdf) 

## Simulation

https://github.com/Mgineer117/Meta-CPO/assets/117319319/37199162-8561-4190-9063-14cef7c3f206

https://github.com/Mgineer117/Meta-CPO/assets/117319319/bd3a751c-ce76-45e9-a7d1-881e6c0ec172

https://github.com/Mgineer117/Meta-CPO/assets/117319319/ea699d99-255c-441d-b98d-7a0a74dba222





## Pre-requisites
- [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)
- [Model-Agnostic Meta-Learning](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
- [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- [Differentiable Convex Programming](https://locuslab.github.io/2019-10-28-cvxpylayers/)


## Usage
Our objective is to evaluate adaptive performance across various environmental parameters. To facilitate this, we've developed custom environments with randomly assigned environmental parameters. For reproducibility, we provide the "stochastic_circle_level.zip" package, which should be placed in your Safety Gym directory: /safety_gymnasium/tasks/safe_navigation/circle.... For guidance on creating your own environments, please refer to Table 1 of [our paper](https://ojs.aaai.org/index.php/AAAI/article/view/30088/31916) and the [Safety Gym](https://safety-gymnasium.readthedocs.io/en/latest/components_of_environments/tasks/task_example.html) documentation.

### Code Reference
* [Khrylx/PyTorch-RL](https://github.com/Khrylx/PyTorch-RL)
* [SapanaChaudhary/PyTorch-CPO](https://github.com/SapanaChaudhary/PyTorch-CPO)


