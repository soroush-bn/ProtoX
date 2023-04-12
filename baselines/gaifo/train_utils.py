import gym
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn

from torch.autograd import Variable
import torchvision.models as models

import torch.nn.functional as F
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time
import random
import sys

from .dagger import get_env_and_model
'''chera nist ???'''
# from gym_minigrid.wrappers import *

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch

from torch_lr_finder import LRFinder
from matplotlib import pyplot as plt
import math
import gc
from torch.utils.data.dataset import Dataset, random_split
import pickle

import wandb
from skimage.transform import resize

#from visualization import *

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        
    def __getitem__(self, index):
        return (self.observations[index].astype(np.float32), self.actions[index].astype(np.long))

    def __len__(self):
        return len(self.observations)

def crop_pong(obs):
    if obs.shape[0] == 1:
        return obs[:,10:84,:,:]
    elif obs.shape[0] == 4:
        return obs[:,10:84,:]

#@title get human normalized scores for atari
def get_human_normalized_scores(scores, env_id):
    games = ['alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'bowling', 'boxing', 'breakout', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk', 'EnduroNoFrameskip-v4', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'james_bond', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman', 'name_this_game', 'PongNoFrameskip-v4', 'private_eye', 'q_bert', 'river_raid', 'road_runner', 'robotank', 'seaquest', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor', 'zaxxon']


    random = [227.80, 5.80, 222.40, 210.00, 719.10, 12850.00, 14.20, 2360.00, 363.90, 23.10, 0.10, 1.70, 2090.90, 811.00, 10780.50, 152.10, -18.60, 0.00, -91.70, 0.00, 65.20, 257.60,  173.00, 1027.00, -11.20, 29.00, 52.00, 1598.00, 258.50, 0.00, 307.30, 2292.30, -20.70, 24.90, 163.90, 1338.50, 11.50, 2.20, 68.40, 148.00, 664.00, -23.80, 3568.00, 11.40, 533.40, 0.00, 16256.90, 563.50, 32.50]

    human = [6875.40, 1675.80, 1496.40, 8503.30, 13156.70, 29028.10, 734.40, 37800.00, 5774.70, 154.80, 4.30, 31.80, 11963.20, 9881.80, 35410.50, 3401.30, -15.50, 309.60, 5.50, 29.60, 4334.70, 2321.00, 2672.00, 25762.50, 0.90, 406.70, 3035.00, 2394.60, 22736.20, 4366.70, 15693.40, 4076.20, 9.30, 69571.30, 13455.00, 13513.30, 7845.00, 11.90, 20181.80, 1652.30, 10250.00, -8.90, 5925.00, 167.60, 9082.00, 1187.50, 17297.60, 4756.50, 9173.30]

    idx = games.index(env_id)

    return [(score - random[idx]) / abs(human[idx] - random[idx]) for score in scores]

#@title get_data
# get dataset of state-action pairs from expert
def get_data(num_interactions=int(6e4), env_id="PongNoFrameskip-v4", preprocess=True):
    env, ppo_expert = get_env_and_model(env_id)

    if env_id == 'CartPole-v1':
        img = env.render(mode='rgb_array') 
    
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    print('state shape: ', state_shape)
    print('action shape: ', action_shape)
    
    atari_games = ['PongNoFrameskip-v4',
                   'EnduroNoFrameskip-v4',
                   'breakout'
                   ]

    
    #gather data
    if isinstance(env.action_space, gym.spaces.Box):
      expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
      #expert_observations = np.empty((num_interactions, 4,84,84))
      expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))

    else:
      #expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
      expert_observations = np.empty((num_interactions, 4,84,84))
      expert_actions = np.empty((num_interactions,) + env.action_space.shape)

    episode_schedule = np.empty((num_interactions, 2))
      
    obs = env.reset()

    ep_number = 0
    
    for i in tqdm(range(num_interactions)):
        action, _ = ppo_expert.predict(obs, deterministic=True)
        #PREPROCESS AFTER EXPERT IS DONE!!!!!!
        if preprocess:
            obs = crop_pong(obs)[0]
            obs = np.expand_dims(resize(obs, (84,84,4)),0)

        expert_observations[i]= obs.transpose(0,3,1,2)
        expert_actions[i] = action

        episode_schedule[i] = np.array([ep_number, i])
        
        obs, reward, done, info = env.step(action)
        if done:
            ep_number = ep_number + 1
            obs = env.reset()

    np.savez_compressed(
        "expert_data",
        expert_actions=expert_actions,
        expert_observations=expert_observations,
        episode_schedule=episode_schedule
    )
    env.close()

    return expert_observations, expert_actions, episode_schedule

def filter_by_action(expert_observations, expert_actions):
    filtered_obs = {}
    for i in range(len(expert_observations)):
        if expert_actions[i] not in filtered_obs.keys():
            filtered_obs[expert_actions[i]] = [expert_observations[i]]
        else:
            filtered_obs[expert_actions[i]].append(expert_observations[i])
    return filtered_obs
