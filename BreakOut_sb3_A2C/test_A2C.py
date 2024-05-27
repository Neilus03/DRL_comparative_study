'''
Code for testing the trained A2C agent in the Breakout environment using stable baselines 3's A2C
implementation and logging the results to wandb.

Note: For understanding the code, it is recommended to read the documentation of stable baselines 3's A2C implementation and to
understand the parameters of the model, go to config.py and read the comments of each parameter.
'''

'''
Import the necessary libraries and modules
'''
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import torch
import wandb
import os
import tensorboard as tb
import config
from utils import make_env, unzip_file


'''
Set up wandb
'''
if config.log_to_wandb:
    wandb.init(project=config.name_test, entity= config.entity, sync_tensorboard=config.sync_tensorboard, name=config.name_test, notes=config.notes)


'''
Set up the environment and the model to test
'''

# Unzip the saved model if config.pretrained is set to True in config.py
if config.pretrained:
    #Unzip the file a2c_Breakout_1M.zip and store the unzipped files in the folder a2c_Breakout_unzipped
    unzip_file(config.saved_model_path, config.unzip_file_path) 
           
#We start with a single environment for Breakout with render mode set to human
env = make_env("BreakoutNoFrameskip-v4") 
#We then wrap the environment with the DummyVecEnv wrapper which converts the environment to a single vectorized environment
env = DummyVecEnv([env]) # Output shape: (1, 84, 84)
#Finally, we wrap the environment with the VecFrameStack wrapper which stacks the observations over the last 4 frames
env = VecFrameStack(env, n_stack=config.n_stack) # Output shape: (4, 84, 84)

# Create the model
model = A2C(policy = config.policy
            ,env = env
            ,verbose = config.verbose)

# Load the model if config.pretrained is set to True in config.py
if config.pretrained:
    # Load the model components, including the policy network and the value network
    model.policy.load_state_dict(torch.load(os.path.join(config.unzip_file_path, "policy.pth")))
    model.policy.optimizer.load_state_dict(torch.load(os.path.join(config.unzip_file_path, "policy.optimizer.pth")))


'''
Test the model in the environment and log the results to wandb
'''
# Run the episodes and render the gameplay
for episode in range(config.test_episodes):
    # Reset the environment and stack the initial state 4 times
    obs = env.reset()#obs = np.stack([obs] * 4, axis=0)  # Initial state stack
    done = False
    episode_reward = 0
    while not done:
        # Take an action in the environment according to the policy of the trained agent
        action, _ = model.predict(obs)
        # Take the action in the environment and store the results in the variables
        obs, reward, done, info = env.step(action) # obs shape: (1, 84, 84), reward shape: (1,), done shape: (1,), info shape: (1,) 
        # Update the total reward
        episode_reward += reward[0]
        # Render the environment to visualize the gameplay of the trained agent
        env.render()
    # Log the total reward of the episode to wandb
    if config.log_to_wandb:
        wandb.log({'test_episode_reward': episode_reward, 'test_episode': episode})


'''
Close the environment and finish the logging
'''
env.close()

if config.log_to_wandb:
    wandb.finish()
