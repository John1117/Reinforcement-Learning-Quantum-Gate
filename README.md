# Robust Quantum Gates by Deep Reinforcement Learning

This repository contains the implementation of my master's thesis project. I leveraged the Reinforcement Learning (RL) algorithm called [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347) and [Generalized Advantage Estimation (GAE)](https://arxiv.org/pdf/1506.02438) to build quantum gates that are robust to quasi-static noise. The optimized quantum gates include the Pauli-X, Hadamard, and CNOT gates. For more informaiton, please refer to [my thesis](https://tdr.lib.ntu.edu.tw/handle/123456789/92226).

Since TensorFlow [`tf_agents.agents.PPOAgent`](https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/PPOAgent) in macOS does **not** support double precision, **float64**, I built a PPO agent by myself in this project.

## Table of Content
- [Introduction](#robust-quantum-gates-by-deep-reinforcement-learning)
- [Getting Start](#getting-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
- [References](#references)

## Getting Start

### Prerequisites
- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`
- `keras`
- `tensorflow`
- `tensorflow-probability`

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/John1117/Reinforcement-Learning-Quantum-Gate.git
    ```
    
2. Install the dependencies using:
    ```bash
    cd Reinforcement-Learning-Quantum-Gate
    pip install -r requirements.txt
    ```
### Usage
1. If you want to train a PPO agent in a quantum gate environment, let's say, Pauli-X gate environment, then please run:
    ```bash
    python reinforcement_learning/x_rl.py
    ```

2. To achieve more detailed control over the training, please tune the parameters in `reinforcement_learning/x_rl.py` file.

## References
- [Robust Quantum Gates by Deep Reinforcement Learning (my master thesis)](https://tdr.lib.ntu.edu.tw/handle/123456789/92226)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347)
- [Generalized Advantage Estimation (GAE)](https://arxiv.org/pdf/1506.02438)
