import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import pickle

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict


def get_env_and_model(env='EnduroNoFrameskip-v4',
                      folder='rl-trained-agents',
                      algo='ppo',
                      n_timesteps=1000,
                      num_threads=-1,
                      n_envs = 1,
                      exp_id=0,
                      verbose=1,
                      no_render=True,
                      deterministic=False,
                      load_best=False,
                      stochastic=False,
                      norm_reward=False,
                      seed=0,
                      reward_log="",
                      gym_packages=[],
                      load_checkpoint=None):  # noqa: C901

    # Going through custom gym packages to let them register in the global registory
    for env_module in gym_packages:
        importlib.import_module(env_module)

    env_id = env
    algo = algo
    folder = folder

    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={exp_id}")

    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    print('filter: ', folder)
    print('algo: ', algo)
    print('env_id: ', env_id)
    print('exp id: ', exp_id)
    print('log path: ', log_path)
        
    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        n_envs = 1

    set_random_seed(seed)

    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    print('hyperparams: ' , hyperparams)
    
    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if env_kwargs is not None:
        env_kwargs.update(env_kwargs)

    print('env kwargs: ', env_kwargs)

    log_dir = reward_log if reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=n_envs,
        stats_path=stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=not no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwupdate(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    obs = env.reset()
    print('initial obs shape: ', obs.shape)
    # Deterministic by default except for atari games
    stochastic = stochastic or is_atari and not deterministic
    deterministic = not stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []

    acts = {}

    return env, model

