import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
import os
import zipfile
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv


class RewardLogger(gym.Wrapper):
    '''
    The RewardLogger wrapper is used to log the rewards of each episode to wandb
    It makes sure that the rewards of each episode are stored in a list and that the current episode reward is reset
    '''
    def __init__(self, env):
        '''
        Description:
            The __init__ function is called when the RewardLogger wrapper is created for the first time,
            it is used to initialize the RewardLogger wrapper with the environment that will be wrapped.
            
        Args:
            env (gym.Env): The environment to be wrapped.
            
        Output:
            None, initializes the RewardLogger wrapper.
        '''
        super(RewardLogger, self).__init__(env)
        # Store the rewards of each episode
        self.episode_rewards = []
        # Store the current episode reward
        self.current_episode_reward = 0

    # The step function is called every time the agent takes an action in the environment
    def step(self, action):
        '''
        Description:
            The step function is called every time the agent takes an action in the environment.
            It is used to update the current episode reward and store the episode reward when the episode is done.
            
        Args:
            action (int): The action taken by the agent.
            
        Output:
            obs (np.ndarray): The observation of the environment after taking the action the shape of the observation is (1, 84, 84).
            reward (float): The reward of the agent after taking the action.
            done (bool): A boolean that indicates if the episode is done.
            truncated (bool): A boolean that indicates if the episode was truncated.
            info (dict): A dictionary with extra information about the environment, like the lives of the agent.
        '''
        # Call the step function of the environment and store the results
        obs, reward, done, truncated, info = self.env.step(action) 
        # Update the current episode reward
        self.current_episode_reward += reward
        # If the episode is done, store the episode reward and reset the current episode reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        # Return the results as in a normal step function
        return obs, reward, done, truncated, info

    # The reset function is called every time the environment is reset (at the beginning of each episode)
    def reset(self, **kwargs):
        '''
        Description:
            The reset function is called every time the environment is reset (at the beginning of each episode).
            
        Args:
            **kwargs: Variable length argument list.
            
        Output:
            obs (np.ndarray): The observation of the environment after it is reset, the shape of the observation is (1, 84, 84).
        '''
        return self.env.reset(**kwargs)

    # The get_episode_rewards function returns the rewards of each episode
    def get_episode_rewards(self):
        '''
        Description:
            The get_episode_rewards function returns the rewards of each episode.
            
        Args:
            None
        
        Output:
            self.episode_rewards (list): A list with the rewards of the episode.
        '''
        return self.episode_rewards


class CustomWandbCallback(BaseCallback):
    '''
    The CustomWandbCallback is a callback* that logs the mean reward of the last 100 episodes to wandb.
    A callback is a function that is called at the end of each episode to perform some action,
    in this case, the action is logging the mean reward of the last 100 episodes to wandb.
    '''
    def __init__(self, check_freq, save_path, verbose=1):
        '''
        Description:
            The __init__ function is called when the CustomWandbCallback is created for the first time,
            it is used to initialize the CustomWandbCallback with the parameters that will be used.
            
        Args:
            check_freq (int): The frequency at which the callback is called.
            save_path (str): The path where the best model will be saved.
            verbose (int): The verbosity level: 0 no output, 1 info, 2 debug.
            
        Output:
            None, initializes the CustomWandbCallback.
        '''
        super(CustomWandbCallback, self).__init__(verbose)
        # Define the frequency at which the callback is called
        self.check_freq = check_freq
        # Define the path where the best model will be saved
        self.save_path = save_path
        # Define the best mean reward as -inf
        self.best_mean_reward = -np.inf


    def _on_step(self) -> bool:
        '''
        Description:
            The _on_step function is called at the end of each episode. 
            It returns True if the callback should be called again, and False otherwise.
            To do this, it checks if the number of calls to the callback is a multiple of the check_freq.
            If it is, it computes the mean reward of the last 100 episodes and logs it to wandb.
            It also saves the model if the mean reward is greater than the best mean reward.
            
        Args:
            None
            
        Output:
            True if the callback should be called again, and False otherwise.
        '''
        # Check if the number of calls to the callback is a multiple of the check_freq
        if self.n_calls % self.check_freq == 0:
            # Gather rewards from all environments, by all environments we mean all the environments in the vectorized environment, usually there is only 1 environment in the vectorized environment   
            all_rewards = []
            for env in self.training_env.envs: # self.training_env is the vectorized environment
                # logger_env is the DummyVecEnv wrapper which converts the environment to a single vectorized environment
                logger_env = env.envs[0] if isinstance(env, DummyVecEnv) else env # env.envs[0] is the AtariWrapper which wraps the environment correctly
                #Check if the logger_env is the RewardLogger wrapper
                if isinstance(logger_env, RewardLogger):
                    # If it is, get the rewards of each episode and store them in all_rewards
                    all_rewards.extend(logger_env.get_episode_rewards())# extend is used to add the elements of a list to another list

            #If there are rewards in all_rewards, compute the mean reward of the last 100 episodes and log it to wandb
            if all_rewards:
                # Compute the mean reward of the last 100 episodes
                mean_reward = np.mean(all_rewards[-self.check_freq:])
                # Log the mean reward of the last 100 episodes to wandb
                wandb.log({'mean_reward': mean_reward, 'steps': self.num_timesteps})

                # Save the best model  
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model'))
            
            
        # Return True if the callback should be called again, and False otherwise
        return True




def make_env(env_id, seed=0):
    '''
    Description:
        The make_env function is used to create the environment with the correct wrappers and rendering.
        
    Args:
        env_id (str): The id of the environment.
        seed (int): The seed of the environment.
        
    Output:
        _init (function): The function that creates and initializes the environment.
    '''
    def _init():
        # Create the environment with render mode set to human
        env = gym.make(env_id, render_mode='human')
        # Set the seed of the environment, this is done to make the results reproducible
        env.seed(seed)
        # Wrap the environment with the AtariWrapper which wraps the environment correctly
        env = AtariWrapper(env)
        # Wrap the environment with the RewardLogger wrapper which logs the rewards of each episode to wandb
        env = RewardLogger(env)
        # Return the environment
        return env
    # Return the _init function which is used to create the environment
    return _init


def unzip_file(zip_path, extract_to_folder):
    """
    Description:
        Unzips a zip file to a specified folder.
        
    Args:
        zip_path (str): The path to the zip file.
        extract_to_folder (str): The folder to extract the files to.
        
    Output:
        None, unzips the zip file to the specified folder.
    """
    # Ensure the target folder exists
    os.makedirs(extract_to_folder, exist_ok=True)
    # Extract the zip file to the target folder
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
