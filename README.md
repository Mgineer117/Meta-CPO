# Meta-Constrained Policy Optimization (Meta-CPO) for Safe and Fast Adaptation for Nonstationary Domains. 
This repository is an adaptation of the [CPO algorithm](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), transforming it into a Meta-learning framework. The modification involves leveraging Differentiable Convex Programming to facilitate the relaxation of gradient computations between parameters. The integration of CPO into the meta-learning framework was achieved through the application of the model-free meta-framework introduced by [MAML](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf). The primary objective of this algorithm is to undergo testing within the [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium), offering an intuitive experimental platform to showcase its effectiveness in the context of Autonomous Driving tasks. For a detailed theory about achieving safety guarantee, please refer to our paper [Constrained Meta-RL with DCO](https://arxiv.org/pdf/2312.10230.pdf) 

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
## Note
The following error may arise
```
Please consider re-formulating your problem so that it is always solvable or increasing the number of solver iterations.
```
This is attributed to the insolvable CPO problem that is natural since CPO sometimes does not have a solution where naive TRPO problem to the cost will be utilized.


## Custom Environment
To create your custom environments, we refer to [Safety Gym](https://safety-gymnasium.readthedocs.io/en/latest/components_of_environments/tasks/task_example.html) documentation, our code implementation in safety_gym folder ```safety_gymnasium/tasks/safe_navigation/```, and Table 1 of [our paper](https://ojs.aaai.org/index.php/AAAI/article/view/30088/31916). In our implementation ```task_level_0``` is used as a fixed environment to evaluate training performance, ```task_level_1``` is to generate environments with stochastic environmental parameters, and ```task_level_2``` is used to generate a meta_testing environment. Other minor changes may be required to adapt to its safety gym package.

## Citing Meta-CPO

If you find Meta-CPO useful and informative, please cite it in your publications.

```bibtex
@inproceedings{cho2024constrained,
  title={Constrained Meta-Reinforcement Learning for Adaptable Safety Guarantee with Differentiable Convex Programming},
  author={Cho, Minjae and Sun, Chuangchuang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={19},
  pages={20975--20983},
  year={2024}
}

```

## Simulation


https://github.com/Mgineer117/Meta-CPO/assets/117319319/bd3a751c-ce76-45e9-a7d1-881e6c0ec172 

https://github.com/Mgineer117/Meta-CPO/assets/117319319/37199162-8561-4190-9063-14cef7c3f206

https://github.com/Mgineer117/Meta-CPO/assets/117319319/ea699d99-255c-441d-b98d-7a0a74dba222

## Code Reference
* [Khrylx/PyTorch-RL](https://github.com/Khrylx/PyTorch-RL)
* [SapanaChaudhary/PyTorch-CPO](https://github.com/SapanaChaudhary/PyTorch-CPO)


