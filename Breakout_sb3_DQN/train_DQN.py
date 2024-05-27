from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch
import config
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import unzip_file, CustomWandbCallback
import os



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
    #Set wandb to log the training process
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
model = DQN(policy=config.policy,
            env=env,        
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts, 
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            replay_buffer_class=config.replay_buffer_class,
            replay_buffer_kwargs=config.replay_buffer_kwargs,
            optimize_memory_usage=config.optimize_memory_usage,
            target_update_interval=config.target_update_interval,
            exploration_fraction=config.exploration_fraction,
            exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log=config.log_dir,
            policy_kwargs=config.policy_kwargs,
            verbose=config.verbose,
            seed=config.seed,
            device=config.device,
            _init_setup_model=config._init_setup_model)
            

print("model in device: ", model.device)

#Load the model if config.pretrained is set to True in config.py
if config.pretrained:
    model = DQN.load(config.saved_model_path, env=env, verbose=config.verbose, tensorboard_log=config.log_dir)
    #Unzip the file a2c_Breakout_1M.zip and store the unzipped files in the folder DQN_Breakout_unzipped
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
    model.learn(total_timesteps=config.total_timesteps, log_interval=config.log_interval, callback=[custom_callback], progress_bar=True)

#Save the model 
model.save(config.saved_model_path[:-4]) #remove the .zip extension from the path

''' 
Close the environment and finish the logging
'''
env.close()
if config.log_to_wandb:
    wandb.finish() 
