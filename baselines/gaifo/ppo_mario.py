#https://raw.githubusercontent.com/vpj/rl_samples/master/ppo.py
# in hamoon file e ppo e

"""
# Proximal Policy Optimization - PPO in PyTorch

This is a minimalistic implementation of [Proximal Policy Optimization - PPO](https://arxiv.org/abs/1707.06347)
 clipped version for Atari Breakout game on OpenAI Gym.
This has less than 250 lines of code.
It runs the game environments on multiple processes to sample efficiently.
Advantages are calculated using [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

**The code for this tutorial is available at
[Github labml/rl_samples](https://github.com/lab-ml/rl_samples).**
And the web version of the tutorial is available
[on my blog](http://blog.varunajayasiri.com/ml/ppo_pytorch.html).

If someone reading this has any questions or comments
 please find me on Twitter, **[@vpj](https://twitter.com/vpj)**.
"""

import multiprocessing
import multiprocessing.connection
from typing import Dict, List

import cv2
import gym
import numpy as np
import torch
from labml import monit, tracker, logger, experiment
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
from copy import deepcopy

from baselines.viper.utils import create_test_env, get_saved_hyperparams
from .dagger import *
from mario_src.env import create_train_env
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Game:
    """
    <a id="GameEnvironment"></a>

    ## Game environment
    
    This is a wrapper for OpenAI gym game environment.
    We do a few things here:

    1. Apply the same action on four frames and get the last frame
    2. Convert observation frames to gray and scale it to (84, 84)
    3. Stack four frames of the last four actions
    4. Add episode information (total reward for the entire episode) for monitoring
    5. Restrict an episode to a single life (game has 5 lives, we reset after every single life)

    #### Observation format
    Observation is tensor of size (4, 84, 84). It is four frames
    (images of the game screen) stacked on first axis.
    i.e, each channel is a frame.
    """

    def __init__(self, seed, env_id, discriminator):

        # create environment

        folder = 'rl-trained-agents'
        algo = 'ppo'
        log_path = os.path.join(folder, algo)
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
        stats_path = os.path.join(log_path, env_id)

        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

        if 'mario' not in env_id:
            self.env = create_test_env(
                env_id,
                n_envs=1,
                stats_path=stats_path,
                seed=seed,
                log_dir='./',#log_dir,
                should_render=False,#not no_render,
                hyperparams=hyperparams,
                env_kwargs=env_kwargs,
            )
        self.env = create_train_env(env_id[-2],env_id[-1], SIMPLE_MOVEMENT,output_path=None)

        #self.env = gym.make(env_id)

        self.env.seed(seed)

        # tensor for a stack of 4 frames
        self.obs_4 = np.zeros((4, 84, 84))

        # keep track of the episode rewards
        self.rewards = []
        # and number of lives left
        #self.lives = 0
        self.discriminator = discriminator.to('cpu')
        
    def step(self, data):
        cur_obs, action = data
        
        """
        ### Step
        Executes `action` for 4 time steps and
         returns a tuple of (observation, reward, done, episode_info).

        * `observation`: stacked 4 frames (this frame and frames for last 3 actions)
        * `reward`: total reward while the action was executed
        * `done`: whether the episode finished (a life lost)
        * `episode_info`: episode information if completed
        """

        reward = 0.
        done = None

        # run for 4 steps
        for i in range(4):
            # execute the action in the OpenAI Gym environment
            obs, r, done, info = self.env.step(action)
            
            next_obs = self._process_obs(obs)
            # CHANGED HERE FOR GAIFO
            #reward += r
            
            cur_obs = torch.FloatTensor(cur_obs[-1]).unsqueeze(0).unsqueeze(0)
            next_obs = torch.FloatTensor(next_obs).unsqueeze(0).unsqueeze(0)
            
            dis_in = torch.cat([cur_obs.view(1,1,84,84),next_obs.view(1,1,84,84)],dim=1)
            

            with torch.no_grad():
                reward += self.discriminator(dis_in.flatten(1))

            '''
            # get number of lives left
            lives = self.env.unwrapped.ale.lives()
            # reset if a life is lost
            if lives < self.lives:
                done = True
                break
            '''
            if done:
                break
            
        # Transform the last observation to (84, 84)
        obs = self._process_obs(obs)

        # maintain rewards for each step
        self.rewards.append(reward)

        if done:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None
            # get the max of last two frames
            # obs = self.obs_2_max.max(axis=0)

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward.detach(), done, episode_info

    def reset(self): 
        """
        ### Reset environment
        Clean up episode info and 4 frame stack
        """

        # reset OpenAI Gym environment
        obs = self.env.reset()

        # reset caches
        obs = self._process_obs(obs)
        for i in range(4):
            self.obs_4[i] = obs
        self.rewards = []

        #self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4

    @staticmethod
    def _process_obs(obs):
        """
        #### Process game frames
        Convert game frames to gray and rescale to 84x84
        """
        obs = obs[0][-1]
        #obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        #obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def worker_process(remote: multiprocessing.connection.Connection, seed,env_id,discriminator):
    """
    ##Worker Process

    Each worker process runs this method
    """

    # create game
    game = Game(seed,env_id,discriminator)

    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    """
    Creates a new worker and runs it in a separate process.
    """

    def __init__(self, seed, env_id,disc):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed,env_id,disc))
        self.process.start()


class Model(nn.Module):
    """
    ## Model
    """

    def __init__(self):
        super().__init__()

        # The first convolution layer takes a
        # 84x84 frame and produces a 20x20 frame
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)

        # The second convolution layer takes a
        # 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        # The third convolution layer takes a
        # 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # 512 features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        # A fully connected layer to get logits for $\pi$
        self.pi_logits = nn.Linear(in_features=512, out_features=4)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

    def forward(self, obs: torch.Tensor):
        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, 7 * 7 * 64))

        h = F.relu(self.lin(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    # scale to `[0, 1]`
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class PPO_MARIO:
    def __init__(self, env_id,actor_critic,disc):
        # #### Configurations

        # $\gamma$ and $\lambda$ for advantage calculation
        self.gamma = 0.99
        self.lamda = 0.95

        # number of updates
        self.updates = 25# was 10k
        # number of epochs to train the model with sampled data
        self.epochs = 5
        # number of worker processes
        self.n_workers = 1#8
        # number of steps to run on each process for a single update
        self.worker_steps = 32#was 128
        # number of mini batches
        self.n_mini_batch = 4
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0)

        # #### Initialize

        # create workers
        self.workers = [Worker(47 + i,env_id,disc) for i in range(self.n_workers)]

        # initialize tensors for observations
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        # model for sampling
        self.model = Model().to(device)
        #self.model = actor_critic.to(device)
        
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    def sample(self) -> (Dict[str, np.ndarray], List):
        """### Sample data with current policy"""

        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 4, 84, 84), dtype=np.uint8)
        log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)

        # sample `worker_steps` from each worker
        for t in range(self.worker_steps):
            with torch.no_grad():
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.flatten().cpu().numpy()
                a = pi.sample()#.flatten()#.item()
                
                #a = a.item()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

            # run sampled actions on each worker
            for w, worker in enumerate(self.workers):
                try:
                    worker.child.send(("step", (obs[w,t], actions[w, t])))
                except Exception as e:
                    worker.child.send(("step", (obs[w,t].detach(), actions[w, t].detach())))
            for w, worker in enumerate(self.workers):
                # get results after executing the actions
                self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()

                # collect episode info, which is available if an episode finished;
                #  this includes total reward and length of the episode -
                #  look at `Game` to see how it works.
                # We also add a game frame to it for monitoring.
                if info:
                    tracker.add('reward', info['reward'])
                    tracker.add('length', info['length'])

        # calculate advantages
        advantages = self._calc_advantages(done, rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'log_pis': log_pis,
            'advantages': advantages
        }

        # samples are currently in [workers, time] table,
        #  we should flatten it
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        ### Calculate advantages

        \begin{align}
        \hat{A_t^{(1)}} &= r_t + \gamma V(s_{t+1}) - V(s)
        \\
        \hat{A_t^{(2)}} &= r_t + \gamma r_{t+1} +\gamma^2 V(s_{t+2}) - V(s)
        \\
        ...
        \\
        \hat{A_t^{(\infty)}} &= r_t + \gamma r_{t+1} +\gamma^2 r_{t+1} + ... - V(s)
        \end{align}

        $\hat{A_t^{(1)}}$ is high bias, low variance whilst
        $\hat{A_t^{(\infty)}}$ is unbiased, high variance.

        We take a weighted average of $\hat{A_t^{(k)}}$ to balance bias and variance.
        This is called Generalized Advantage Estimation.
        $$\hat{A_t} = \hat{A_t^{GAE}} = \sum_k w_k \hat{A_t^{(k)}}$$
        We set $w_k = \lambda^{k-1}$, this gives clean calculation for
        $\hat{A_t}$

        \begin{align}
        \delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t)
        \\
        \hat{A_t} &= \delta_t + \gamma \lambda \delta_{t+1} + ... +
                             (\gamma \lambda)^{T - t + 1} \delta_{T - 1}
        \\
        &= \delta_t + \gamma \lambda \hat{A_{t+1}}
        \end{align}
        """

        # advantages table
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        # $V(s_{t+1})$
        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):
            # mask if episode completed after step $t$
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            # $\delta_t$
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}}$
            last_advantage = delta + self.gamma * self.lamda * last_advantage

            # note that we are collecting in reverse order.
            # *My initial code was appending to a list and
            #   I forgot to reverse it later.
            # It took me around 4 to 5 hours to find the bug.
            # The performance of the model was improving
            #  slightly during initial runs,
            #  probably because the samples are similar.*
            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages

    def train(self, samples: Dict[str, torch.Tensor], learning_rate: float, clip_range: float):
        """
        ### Train the model based on samples
        """

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(clip_range=clip_range,
                                       samples=mini_batch)

                # compute gradients
                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor:
        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']
        
        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        pi, value = self.model(samples['obs'])

        # #### Policy

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        log_pi = pi.log_prob(samples['actions'])

        # ratio $r_t(\theta) = \frac{\pi_\theta (a_t|s_t)}{\pi_{\theta_{OLD}} (a_t|s_t)}$;
        # *this is different from rewards* $r_t$.
        ratio = torch.exp(log_pi - samples['log_pis'])

        # \begin{align}
        # \mathcal{L}^{CLIP}(\theta) =
        #  \mathbb{E}_{a_t, s_t \sim \pi_{\theta{OLD}}} \biggl[
        #    min \Bigl(r_t(\theta) \bar{A_t},
        #              clip \bigl(
        #               r_t(\theta), 1 - \epsilon, 1 + \epsilon
        #              \bigr) \bar{A_t}
        #    \Bigr)
        #  \biggr]
        # \end{align}
        #
        # The ratio is clipped to be close to 1.
        # We take the minimum so that the gradient will only pull
        # $\pi_\theta$ towards $\pi_{\theta_{OLD}}$ if the ratio is
        # not between $1 - \epsilon$ and $1 + \epsilon$.
        # This keeps the KL divergence between $\pi_\theta$
        #  and $\pi_{\theta_{OLD}}$ constrained.
        # Large deviation can cause performance collapse;
        #  where the policy performance drops and doesn't recover because
        #  we are sampling from a bad policy.
        #
        # Using the normalized advantage
        #  $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$
        #  introduces a bias to the policy gradient estimator,
        #  but it reduces variance a lot.
        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()

        # #### Entropy Bonus

        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # #### Value

        # \begin{align}
        # V^{\pi_\theta}_{CLIP}(s_t)
        #  &= clip\Bigl(V^{\pi_\theta}(s_t) - \hat{V_t}, -\epsilon, +\epsilon\Bigr)
        # \\
        # \mathcal{L}^{VF}(\theta)
        #  &= \frac{1}{2} \mathbb{E} \biggl[
        #   max\Bigl(\bigl(V^{\pi_\theta}(s_t) - R_t\bigr)^2,
        #       \bigl(V^{\pi_\theta}_{CLIP}(s_t) - R_t\bigr)^2\Bigr)
        #  \biggr]
        # \end{align}
        #
        # Clipping makes sure the value function $V_\theta$ doesn't deviate
        #  significantly from $V_{\theta_{OLD}}$.
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range,
                                                                              max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) -
        #  c_1 \mathcal{L}^{VF} (\theta) + c_2 \mathcal{L}^{EB}(\theta)$

        # we want to maximize $\mathcal{L}^{CLIP+VF+EB}(\theta)$
        # so we take the negative of it as the loss
        loss = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()

        tracker.add({'policy_reward': policy_reward,
                     'vf_loss': vf_loss,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': approx_kl_divergence,
                     'clip_fraction': clip_fraction})

        return loss

    def run_training_loop(self):
        """
        ### Run training loop
        """

        # last 100 episode information
        #tracker.set_queue('reward', 100, True)
        #tracker.set_queue('length', 100, True)

        for update in range(self.updates):#update in monit.loop(self.updates):
            progress = update / self.updates

            # decreasing `learning_rate` and `clip_range` $\epsilon$
            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)

            # sample with current policy
            samples = self.sample()

            # train the model
            self.train(samples, learning_rate, clip_range)

            # write summary info to the writer, and log to the screen
            #tracker.save()
            #if (update + 1) % 1_000 == 0:
            #    logger.log()

        self.destroy()
        
    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))
