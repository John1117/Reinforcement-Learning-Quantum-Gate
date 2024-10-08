# Robust Quantum Gates by Deep Reinforcement Learning

This repository contains the implementation of my master's thesis project, which leverages the Reinforcement Learning (RL) algorithm called Proximal Policy Optimization (PPO) and Generalized Advantage Estimation (GAE) to build quantum gates that are robust to quasi-static noise. The quantum gates optimized include the Pauli-X, Hadamard, and CNOT gates. For more informaiton, please refer to my thesis [here](https://tdr.lib.ntu.edu.tw/handle/123456789/92226).

## Features

- **Quantum Gate Optimization**: Using reinforcement learning to optimize quantum gates.
- **Proximal Policy Optimization (PPO)**: A policy gradient method used for training robust quantum gates.
- **Generalized Advantage Estimation (GAE)**: Variance reduction in policy gradient estimation.
- **Noise Robustness**: Optimized gates are resistant to quasi-static noise.

## Project Structure

- `pack/agent`: Implementation of PPO.
- `pack/network`: Actor network, value network and normal projection network.
- `pack/utils`: Implementation of GAE loss and some useful functions.
- `pack/env`: Implementation of single-qubit quantum gate control RL environemnt.
- `pack/driver`: Driver that collects data in quantum control environemnt with given policy.
- `pack/buffer`: A data container called Buffer to store collected data temporarily.
- `pack/train`: The main training workflow of a PPO agent.
- `pack/test`: Noise testing for trained quantum gates.
- `pack/plot`: Plot funcitons for training curve and performance testing.
- `training_and_testing/`: Scripts for training and testing quantum gates including Pauli-X, Hadamard, and CNOT gates.
- `plotting/`: Scripts for visualizing performance and robustness.

## Prerequisites
- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`
- `keras`
- `tensorflow` (for implementing PPO and GAE)
- `tensorflow_probability`

## Reference
- [Robust Quantum Gates by Deep Reinforcement Learning (my master thesis)](https://tdr.lib.ntu.edu.tw/handle/123456789/92226)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347)
- [Generalized Advantage Estimation (GAE)](https://arxiv.org/pdf/1506.02438)
