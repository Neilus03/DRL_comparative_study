from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import gymnasium as gym
import torch
import config
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import make_env, unzip_file, CustomWandbCallback, RewardLogger
import os
from stable_baselines3.common.utils import get_latest_run_id

'''
Set up the appropriate directories for logging and saving the model
'''
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.save_path, exist_ok=True)

#Create the callback that logs the mean reward of the last 100 episodes to wandb
custom_callback = CustomWandbCallback(config.check_freq, config.save_path)


'''
Set up loging to wandb
'''

#Set wandb to log the training process
if config.log_to_wandb:
    wandb.init(project=config.project_train, entity = config.entity, name=config.name_train, notes=config.notes, sync_tensorboard=config.sync_tensorboard)
    #wandb_callback is a callback that logs the training process to wandb, this is done because wandb.watch() does not work with sb3
    wandb_callback = WandbCallback()


'''
Set up the environment
'''
# Create multiple environments and wrap them correctly
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=config.n_envs, seed=config.seed)
env = VecFrameStack(env, n_stack=config.n_stack)


'''
Set up the model
'''
#Create the model with the parameters specified in config.py, go to config.py to see the meaning of each parameter in detail
model = PPO(policy=config.policy
            ,env=env
            ,learning_rate=config.learning_rate
            ,n_steps=config.n_steps
            ,batch_size=config.batch_size
            ,n_epochs=config.n_epochs
            ,gamma=config.gamma            
            ,gae_lambda=config.gae_lambda
            ,clip_range=config.clip_range
            ,clip_range_vf=config.clip_range_vf
            ,normalize_advantage=config.normalize_advantage
            ,ent_coef=config.ent_coef
            ,vf_coef=config.vf_coef
            ,max_grad_norm=config.max_grad_norm
            ,use_sde=config.use_sde
            ,sde_sample_freq=config.sde_sample_freq
            #,rollout_buffer_class=config.rollout_buffer_class
            #,rollout_buffer_kwargs=config.rollout_buffer_kwargs
            ,target_kl=config.target_kl
            ,stats_window_size=config.stats_window_size
            ,tensorboard_log=config.log_dir
            ,policy_kwargs=config.policy_kwargs
            ,verbose=config.verbose
            ,seed=config.seed
            ,device=config.device
            ,_init_setup_model=config._init_setup_model
            )

print("model in device: ", model.device)

#Load the model if config.pretrained is set to True in config.py
if config.pretrained:
    model = PPO.load(config.saved_model_path, env=env, verbose=config.verbose, tensorboard_log=config.log_dir)
    #Unzip the file a2c_Breakout_1M.zip and store the unzipped files in the folder a2c_Breakout_unzipped
    unzip_file(config.saved_model_path, config.unzip_file_path) 
    model.policy.load_state_dict(torch.load(os.path.join(config.unzip_file_path, "policy.pth")))
    model.policy.optimizer.load_state_dict(torch.load(os.path.join(config.unzip_file_path, "policy.optimizer.pth")))



'''
Train the model and save it
'''
#model.learn will train the model for 1e6 timesteps, timestep is the number of actions taken by the agent, 
# in a game like breakout, the agent takes an action every frame, then the number of timesteps is the number of frames,
# which is the number of frames in 1 game multiplied by the number of games played.
#The average number of frames in 1 game is 1000, so 1e6 timesteps is 1000 games more or less.
#log_interval is the number of timesteps between each log, in this case, the training process will be logged every 100 timesteps.
#callback is a callback that logs the training process to wandb, this is done because wandb.watch() does not work with sb3

if config.log_to_wandb:
    model.learn(total_timesteps=config.total_timesteps, log_interval=config.log_interval, callback=[wandb_callback, custom_callback], progress_bar=True)
else:
    model.learn(total_timesteps=config.total_timesteps, log_interval=config.log_interval, callback=custom_callback, progress_bar=True)
#Save the model 
model.save(config.saved_model_path[:-4]) #remove the .zip extension from the path


''' 
Close the environment and finish the logging
'''
env.close()
if config.log_to_wandb:
    wandb.finish()