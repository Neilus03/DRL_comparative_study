# BreakOut A2C Reinforcement Learning Project

This repository contains code for training and testing a Proximal Policy Optimization (PPO) agent on the Breakout environment using Stable Baselines 3's PPO implementation.

## Files Description

- `train_A2C.py`: This script is used for training the PPO agent. It includes the necessary imports from Stable Baselines 3 and environment setup for training.
- `test_A2C.py`: This script is for testing the trained PPO agent in the Breakout environment and logging the results to Weights and Biases (wandb).
- `config.py`: Contains all configuration variables for the project, including settings for the pretrained model, checkpoint frequency, and save paths.
- `utils.py`: Includes utility classes and functions, such as the `RewardLogger` wrapper for logging rewards to wandb.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- stable-baselines3
- gymnasium (formerly known as `gym`)
- wandb (for logging and tracking experiments) By default the flag is set to false, if you want to use it add your user.
- numpy

You can install these packages using `pip`:

```pip install stable-baselines3[extra] gymnasium wandb numpy```

## Training the Agent
To train the A2C agent, run the train_PPO.py script:

```python train_PPO.py```

Make sure to check the config.py file for configuration settings like whether to use a pretrained model and the checkpoint frequency.

## Testing the Agent
After training, you can test the agent using the test_PPO.py script:


```python test_PPO.py```

This will use the trained model and log the results to wandb for performance tracking.
