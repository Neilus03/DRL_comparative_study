import os
from stable_baselines3.common.utils import get_latest_run_id
import torch
    

'''FILE TO STORE ALL THE CONFIGURATION VARIABLES'''

#pretrained is a boolean that indicates if a pretrained model will be loaded
pretrained = False # Set to True if you want to load a pretrained model

#check_freq is the frequency at which the callback is called, in this case, the callback is called every 2000 timesteps
check_freq = 2000

#save_path is the path where the best model will be saved
save_path = "./breakout_ppo_1M_save_path"

#log_dir is the path where the logs will be saved
log_dir = "./log_dir"


'''
Hyperparameters of the model {learning_rate, gamma, device, n_steps, gae_lambda, ent_coef, vf_coef, max_grad_norm, rms_prop_eps, use_rms_prop, use_sde, sde_sample_freq, normalize_advantage}
'''
#policy is the policy of the model, in this case, the model will use a convolutional neural network
policy = "CnnPolicy"

#learning_rate is the learning rate of the model
learning_rate =5e-4  #first trial: 5e-4   #second trial: 1e-4  #third trial: 1e-3  #fourth trial: 5e-5 #fifth trial: 5e-5 gamma = 0.90 #sixth trial: 1e-4 gamma = 0.90 #seventh trial: 5e-4 gamma = 0.90

#gamma is the discount factor
gamma = 0.99

#device is the device where the model will be trained, if cuda is available, the model will be trained in the gpu, otherwise, it will be trained in the cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

#n_steps is the number of steps taken by the model before updating the parameters
n_steps = 24

#batch_size is the number of samples used in each update
batch_size = 96

#n_epochs is the number of epochs when optimizing the surrogate loss
n_epochs = 6

#gae_lambda is the lambda parameter of the generalized advantage estimation, set to 1 to disable it
gae_lambda = 0.95

#clip_range is the clipping parameter of the surrogate loss
clip_range = 0.2

#clip_range_vf is the clipping parameter of the value function
clip_range_vf = 1

#ent_coef is the entropy coefficient, set to 0 to disable it
ent_coef = 0.01

#vf_coef is the value function coefficient, If we set it to 0.5, then the value function loss will be half the policy loss
vf_coef = 0.5

#max_grad_norm is the maximum value for the gradient clipping
max_grad_norm = 0.5

#use_sde is a boolean that indicates if the stochastic differential equation will be used
#The stochastic differential equation is a method to add noise to the actions taken by the agent to improve exploration
use_sde = False

#sde_sample_freq is the frequency at which the noise is added to the actions. If set to -1, the noise will be added every timestep
sde_sample_freq = -1

#rollout_buffer_class is the class of the rollout buffer, in this case, the model will use the RolloutBuffer class
rollout_buffer_class = None

#rollout_buffer_kwargs is a dictionary with the keyword arguments for the rollout buffer. If None, it will use the default arguments
rollout_buffer_kwargs = None

#target_kl is the target value for the KL divergence between the old and updated policy
target_kl = 0.5

#normalize_advantage is a boolean that indicates if the advantage will be normalized, by normalizing the advantage, 
# the variance of the advantage is reduced, this is done to improve the training process because the advantage is used to calculate the policy loss
normalize_advantage = False

#stats_window_size is the size of the window used to calculate the mean and standard deviation of the advantage
stats_window_size = 100

#tensorboard_log is the path where the tensorboard logs will be saved, in our case, the logs will be saved in the log_dir
tensorboard_log = log_dir

#policy_kwargs is a dictionary with the keyword arguments for the policy. If None, it will use the default arguments
policy_kwargs = None

#verbose is the verbosity level: 0 no output, 1 info, 2 debug
verbose = 2

#seed is the seed for the pseudo random number generator used by the model. It is set to None to use a random seed,
# and set to 0 to use a fixed seed for reproducibility
seed = None

#_init_setup_model is a boolean that indicates if the model will be initialized after being created, set to True to initialize the model
_init_setup_model = True

#total_timesteps is the total number of timesteps that the model will be trained. In this case, the model will be trained for 1e7 timesteps
#Take into account that the number of timesteps is not the number of episodes, in a game like breakout, the agent takes an action every frame,
# then the number of timesteps is the number of frames, which is the number of frames in 1 game multiplied by the number of games played.
#The average number of frames in 1 game is 1000, so 1e7 timesteps is 1000 games more or less.
total_timesteps = int(3e7)

#log_interval is the number of timesteps between each log, in this case, the training process will be logged every 100 timesteps.
log_interval = 100

'''
Saved model path
'''

#for the path to be shorter just put "./a2c_Breakout_1M.zip" instead of the full path
saved_model_path = "./PPO_Breakout_30M_lr_5e-4_gamma_90.zip"
unzip_file_path =  "./PPO_Breakout_30M_lr_5e-4_gamma_90_unzipped"

'''
Environment variables
'''
#n_stack is the number of frames stacked together to form the input to the model
n_stack = 4
#n_envs is the number of environments that will be run in parallel
n_envs = 4

'''
Wandb configuration
'''
#log_to_wandb is a boolean that indicates if the training process will be logged to wandb
log_to_wandb = False

# project is the name of the project in wandb
project_train = "BREAKOUT_SB3_BENCHMARK"
project_test = "breakout-PPO2-test"

#entity is the name of the team in wandb
entity = "ai42"

#name is the name of the run in wandb
name_train = "PPO_breakout_lr_5e-4_gamma_90"
name_test = "PPO2_breakout_test"
#notes is a description of the run
notes = "PPO2_breakout with parameters: {}".format(locals()) #locals() returns a dictionary with all the local variables, in this case, all the variables in this file
#sync_tensorboard is a boolean that indicates if the tensorboard logs will be synced to wandb
sync_tensorboard = True


'''
Test configuration
'''
test_episodes = 100


