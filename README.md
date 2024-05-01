# PyTorch implementation of Meta-Constrained Policy Optimization (Meta-CPO)
This repository is an adaptation of the [CPO algorithm](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), transforming it into a Meta-learning framework. The modification involves leveraging Differentiable Convex Programming to facilitate the relaxation of gradient computations between parameters. The integration of CPO into the meta-learning framework was achieved through the application of the model-free meta-framework introduced by [MAML](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf). The primary objective of this algorithm is to undergo testing within the [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium), offering an intuitive experimental platform to showcase its effectiveness in the context of Autonomous Driving tasks. For the detailed theory with regard to achieve safety guarantee, please refer to our paper [Constrained Meta-RL with DCO](https://arxiv.org/pdf/2312.10230.pdf) 

## Citing Meta-CPO

If you find Meta-CPO useful and informative, please cite it in your publications.

```bibtex
@article{Cho_Sun_2024,
title={Constrained Meta-Reinforcement Learning for Adaptable Safety Guarantee with Differentiable Convex Programming},
volume={38},
url={https://ojs.aaai.org/index.php/AAAI/article/view/30088},
DOI={10.1609/aaai.v38i19.30088},
abstractNote={Despite remarkable achievements in artificial intelligence, the deployability of learning-enabled systems in high-stakes real-world environments still faces persistent challenges. For example, in safety-critical domains like autonomous driving, robotic manipulation, and healthcare, it is crucial not only to achieve high performance but also to comply with given constraints. Furthermore, adaptability becomes paramount in non-stationary domains, where environmental parameters are subject to change. While safety and adaptability are recognized as key qualities for the new generation of AI, current approaches have not demonstrated effective adaptable performance in constrained settings. Hence, this paper breaks new ground by studying the unique challenges of ensuring safety in nonstationary environments by solving constrained problems through the lens of the meta-learning approach (learning to learn). While unconstrained meta-learning already encounters complexities in end to end differentiation of the loss due to the bi-level nature, its constrained counterpart introduces an additional layer of difficulty, since the constraints imposed on task-level updates complicate the differentiation process. To address the issue, we first employ successive convex-constrained policy updates across multiple tasks with differentiable convex programming, which allows meta-learning in constrained scenarios by enabling end-to-end differentiation. This approach empowers the agent to rapidly adapt to new tasks under nonstationarity while ensuring compliance with safety constraints. We also provide a theoretical analysis demonstrating guaranteed monotonic improvement of our approach, justifying our algorithmic designs. Extensive simulations across diverse environments provide empirical validation with significant improvement over established benchmarks.},
number={19},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Cho, Minjae and Sun, Chuangchuang},
year={2024},
month={Mar.},
pages={20975-20983} }
}
```

--------------------------------------------------------------------------------

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


In the code, we have already implemented testing domains within the safety_gymnasium folder specifically for the ```Button``` and ```Circle``` tasks to evaluate adaptive performance (it is in ```safety_gymnasium/tasks/safe_navigation/```). If one wishes to evaluate and replicate in different environmental settings, they will need to implement their own custom environments and refer to the details in the **Custom Environment** section below. To conduct experiments, choose any desired agent and set the environment as either ```Safety[Agent]Stcircle``` or ```Safety[Agent]Stbutton``` by specifying the env_name in utils/apr_parse.py. Then, execute the following command with appropriate hyperparameter settings:

```
python3 main.py
```


## Custom Environment
To create your custom environments, we refer to [Safety Gym](https://safety-gymnasium.readthedocs.io/en/latest/components_of_environments/tasks/task_example.html) documentation, our code implementation in safety_gym folder "safety_gymnasium/tasks/safe_navigation/, and Table 1 of [our paper](https://ojs.aaai.org/index.php/AAAI/article/view/30088/31916). In our implementation ```task_level_0``` is used as a fixed environment to evaluate training performance, ```task_level_1``` is to generate environments with stochastic environmental parameters, and ```task_level_2``` is to generate meta_testing environment. Other minor changes may be required to adapt to its safety gym package.


## Simulation


https://github.com/Mgineer117/Meta-CPO/assets/117319319/bd3a751c-ce76-45e9-a7d1-881e6c0ec172 

https://github.com/Mgineer117/Meta-CPO/assets/117319319/37199162-8561-4190-9063-14cef7c3f206

https://github.com/Mgineer117/Meta-CPO/assets/117319319/ea699d99-255c-441d-b98d-7a0a74dba222

## Code Reference
* [Khrylx/PyTorch-RL](https://github.com/Khrylx/PyTorch-RL)
* [SapanaChaudhary/PyTorch-CPO](https://github.com/SapanaChaudhary/PyTorch-CPO)


