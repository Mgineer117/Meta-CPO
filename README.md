# PyTorch implementation of Meta-Constrained Policy Optimization (Meta-CPO)
This repository is an adaptation of the [CPO algorithm](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), transforming it into a Meta-learning framework. The modification involves leveraging Differentiable Convex Programming to facilitate the relaxation of gradient computations between parameters. The integration of CPO into the meta-learning framework was achieved through the application of the model-free meta-framework introduced by [MAML](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf). The primary objective of this algorithm is to undergo testing within the [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium), offering an intuitive experimental platform to showcase its effectiveness in the context of Autonomous Driving tasks. For the detailed theory with regard to achieve safety guarantee, please refer to our paper [Constrained Meta-RL with DCO](https://arxiv.org/pdf/2312.10230.pdf) 

## Pre-requisites
- [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf) For constrained optimization..
- [Model-Agnostic Meta-Learning](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf) For model-free meta-learning framework..
- [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) For experiments... 
- [Differentiable Convex Programming](https://locuslab.github.io/2019-10-28-cvxpylayers/) To enable efficient gradient computations between meta and local parameters


## Usage
To create a conda environment, run the following commands:

```bash
conda create --name myenv python==3.10.*
conda activate myenv
```
Then, install the required packages using pip:
```
pip install -r requirements.txt
```



To execute experiments, **ensure that you have the relevant custom meta-environments in the directory.** If not, please refer to the "Custom Environment" section below. Then, run the following command:
```
python3 main.py
```

This command will initiate the execution of the experiments.


### Custom Environment
Our objective is to evaluate adaptive performance across various environmental parameters. To facilitate this, we've developed custom environments with randomly assigned environmental parameters. For reproducibility, we provide the ```"stochastic_circle_level.zip"``` package in this file, which should be placed in your Safety Gym directory: ```/safety_gymnasium/tasks/safe_navigation/circle....``` 

The given file is for **circle** task and ```env_level_0``` is used as a fixed environment to evaluate training performance, ```env_level_1``` is to generate environments with stochastic environmental parameters, and ```env_level_2``` is to generate meta_testing environment.


For guidance on creating your own environments, please refer to Table 1 of [our paper](https://ojs.aaai.org/index.php/AAAI/article/view/30088/31916) and the [Safety Gym](https://safety-gymnasium.readthedocs.io/en/latest/components_of_environments/tasks/task_example.html) documentation.


## Simulation



https://github.com/Mgineer117/Meta-CPO/assets/117319319/bd3a751c-ce76-45e9-a7d1-881e6c0ec172 

https://github.com/Mgineer117/Meta-CPO/assets/117319319/37199162-8561-4190-9063-14cef7c3f206

https://github.com/Mgineer117/Meta-CPO/assets/117319319/ea699d99-255c-441d-b98d-7a0a74dba222

### Code Reference
* [Khrylx/PyTorch-RL](https://github.com/Khrylx/PyTorch-RL)
* [SapanaChaudhary/PyTorch-CPO](https://github.com/SapanaChaudhary/PyTorch-CPO)


