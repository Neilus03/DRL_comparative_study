from stable_baselines3.common.utils import get_latest_run_id
import torch
    

'''FILE TO STORE ALL THE CONFIGURATION VARIABLES'''

#pretrained is a boolean that indicates if a pretrained model will be loaded
pretrained = False # Set to True if you want to load a pretrained model

#check_freq is the frequency at which the callback is called, in this case, the callback is called every 2000 timesteps
check_freq = 2000

#save_path is the path where the best model will be saved
save_path = "./breakout_DQN_1M_save_path"
 
#log_dir is the path where the logs will be saved
log_dir = "./log_dir"


'''
Hyperparameters of the model {policy, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model, total_timesteps, log_interval}
'''
#policy is the policy of the model, in this case, the model will use a convolutional neural network
policy = "CnnPolicy"

#learning_rate is the learning rate of the model
learning_rate =5e-4  #first trial: 5e-4   #second trial: 1e-4  #third trial: 1e-3  #fourth trial: 5e-5 #fifth trial: 5e-5 gamma = 0.90 #sixth trial: 1e-4 gamma = 0.90 #seventh trial: 5e-4 gamma = 0.90

# buffer_size is the size of the replay buffer
buffer_size=100000

#learning_starts is the number of timesteps before the first interaction with the environment
learning_starts=50000

#batch_size is the number of samples that will be taken from the replay buffer for training the model
batch_size=64

#tau is the soft update coefficient for updating the target network if set to 1 then the target network is hard updated every target_update_interval timesteps
tau= 1.0

#gamma is the discount factor 
gamma=0.99

#train_freq is the number of timesteps between each training step
train_freq=4

#gradient_steps is the number of gradient steps to take after each rollout
gradient_steps=1

#replay_buffer_class is the class of the replay buffer
replay_buffer_class=None

#replay_buffer_kwargs is the keyword arguments for the replay buffer
replay_buffer_kwargs=None

#optimize_memory_usage is a boolean that indicates if the memory usage will be optimized
optimize_memory_usage=False

#target_update_interval is the number of timesteps between each target network update
target_update_interval=10000

#exploration_fraction is the fraction of the total number of timesteps that the exploration rate will be annealed
exploration_fraction=0.1

#exploration_initial_eps is the initial value of the exploration rate
exploration_initial_eps=1.0

#exploration_final_eps is the final value of the exploration rate
exploration_final_eps=0.05

#max_grad_norm is the maximum value of the gradient norm
max_grad_norm=0.5

#stats_window_size is the size of the window for computing the stats
stats_window_size=100

#tensorboard_log is the path to the folder where the tensorboard logs will be saved
tensorboard_log=log_dir

#policy_kwargs is the keyword arguments for the policy
policy_kwargs=None

#verbose is the verbosity level: 0 no output, 1 info, 2 debug
verbose=2

#seed is the seed for the pseudo random number generator
seed=None

#device is the device on which the model will be trained
device= "cuda" if torch.cuda.is_available() else "cpu"

#_init_setup_model is a boolean that indicates if the model will be initialized
_init_setup_model=True

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
saved_model_path = "./DQN_Breakout_30M_lr_5e-4_gamma_90.zip"
unzip_file_path =  "./DQN_Breakout_30M_lr_5e-4_gamma_90_unzipped"

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
project_test = "breakout-sb3-DQN-test"

#entity is the name of the team in wandb
entity = "ai42"

#name is the name of the run in wandb
name_train = "DQN_breakout_lr_5e-4_gamma_90"
name_test = "DQN_breakout_test"
#notes is a description of the run
notes = "DQN_breakout with parameters: {}".format(locals()) #locals() returns a dictionary with all the local variables, in this case, all the variables in this file
#sync_tensorboard is a boolean that indicates if the tensorboard logs will be synced to wandb
sync_tensorboard = True


'''
Test configuration
'''
test_episodes = 100


