# Breakout Reinforcement Learning Implementations

This repository contains multiple implementations of Reinforcement Learning algorithms applied to the Breakout game environment. Each subdirectory contains a different approach, including base DQN models, Stable Baselines 3 A2C, PPO, and DQN implementations.

## Directory Structure

- `BreakOut_base`: Contains the base implementation of the Deep Q-Network (DQN) for the Breakout game. See the specific README in this directory for more details.
- `BreakOut_sb3_A2C`: Houses the implementation of the A2C algorithm using Stable Baselines 3. Detailed instructions are provided in the corresponding README.
- `BreakOut_sb3_PPO`: This directory includes the Proximal Policy Optimization (PPO) approach using Stable Baselines 3. Refer to the README within for setup and usage instructions.
- `Breakout_sb3_DQN`: Features the implementation of DQN using Stable Baselines 3. The README in this folder will guide you through the necessary steps for training and testing.

## Getting Started

To get started with any of the implementations, navigate to the respective directory and follow the instructions provided in its README file.

## Installation

Ensure Python 3.x is installed along with the following dependencies, which are common across all implementations:

```bash
pip install torch gymnasium stable-baselines3[extra] wandb numpy
```
