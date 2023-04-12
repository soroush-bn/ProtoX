from mario_src.env import create_train_env
from mario_src.ppo import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

import gym
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm
import numpy as np
import random
from .dagger import *
#from trpo_atari import trpo
from .ppo_mario import *
from .train_utils import ExpertDataSet

from torch.distributions import Categorical

if torch.cuda.is_available:
    device = "cuda:0"
    print('Using GPU')
else:
    device = "cpu"
    print('using CPU')

device = 'cpu'

class Disc(nn.Module):

    def __init__(self):
        super(Disc, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(2*84*84, 1024)  # 5*5 from image dimension
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class GAIFODataSet(Dataset):
    def __init__(self, states, next_states):
        self.states = states
        self.next_states = next_states

    def __getitem__(self, index):
        return (self.states[index].astype(np.float32), self.next_states[index].astype(np.float32))

    def __len__(self):
        return len(self.states)

def test(pi, device, test_loader):
    pi.eval()

    total = 0.
    correct = 0.
    for (states, actions) in test_loader:
        states = states.to(device)
        actions = actions.to(device)

        probs, _ = pi(states)
        
        predicted = probs.sample()#.item()

        total += actions.size(0)
        correct += (predicted == actions).sum().item()

    acc = 100 * (correct / total)
    return acc

def flip_fidelity(net,
                   world=8,
                   stage=3,
                   action_type='simple',
                   saved_path='trained_models',
                   output_path=None,
                   n_flip=int(10),
                   num_interactions=int(10)):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(world, stage, actions,output_path=output_path)
    model = PPO(env.observation_space.shape[0], len(actions))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, stage),
                                         map_location=lambda storage, loc: storage))
    model.eval()

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    #expert_observations = np.empty((num_interactions,4,84,84))
    max_size = max(n_flip, num_interactions) + 1
    expert_actions = np.empty((max_size,) + env.action_space.shape)
    agent_actions = np.empty((max_size,) + env.action_space.shape)
    #episode_schedule = np.empty((num_interactions,2))
    
    ep_number = 0
    step_number = 0
    correct = 0
    total = 0
    flip_correct = 0
    flip_total = 0
    while flip_total < n_flip and step_number < num_interactions:
        print(step_number, num_interactions)
        state = torch.from_numpy(env.reset())
        while flip_total < n_flip and step_number < num_interactions:#True:
            if torch.cuda.is_available():
                state = state.cuda()
            try:
                logits, value = model(state)
            except Exception as e:
                logits = model(state)
            policy = F.softmax(logits, dim=1)
            
            # expert action
            if random.random() < .995:
                action = torch.argmax(policy).item()
            else:
                action = np.random.choice(7,size=1,p=policy.detach().cpu().numpy()[0])[0]
            
            # crop for agent
            #state = torchvision.transforms.functional.crop(state, 15,0,84-15,84)
            #state = torchvision.transforms.functional.resize(state,(84,84))       
                
            # agent action
            try:
                probs, _ = net(state.to(device))
            except Exception as e:
                probs = net(state.to(device))
            #agent_policy = F.softmax(l, dim=1)
            agent_action = probs.sample().item()#torch.argmax(agent_policy).item()
            
            if action == agent_action:
                correct += 1
            total += 1
            
            
            if step_number >= 1  and action != expert_actions[step_number-1]:
                flip_total += 1
                if agent_action == action and agent_actions[step_number-1] == expert_actions[step_number-1]:
                    flip_correct += 1
            
            state, reward, done, info = env.step(action)
            
            # save data
            #expert_observations[step_number] = state#.transpose(0,3,1,2) 
            expert_actions[step_number] = action
            agent_actions[step_number] = agent_action
            #episode_schedule[step_number] = np.array([ep_number, step_number])
            
            state = torch.from_numpy(state)
            env.render()
            
            step_number += 1
            
            if info["flag_get"]:
                print("World {} stage {} completed".format(world, stage))
                ep_number += 1
                break
                
    
    total_fidelity = (correct / total) * 100.
    flip_fidelity = (flip_correct / flip_total) * 100.
    env.close()
    print('got', flip_total, ' flip points')
    return total_fidelity, flip_fidelity
#if __name__ == "__main__":
#    opt = get_args()
#    test(opt)


def validate(pi,expert,n_eps=10,max_len=10000):
    # rollout pi
    states = []
    next_states = []
    obs = env.reset()
    steps = 0
    done = False
    corr = 0
    for j in range(n_eps):
        while not done:
            probs, _ = pi(torch.FloatTensor(obs).to(device))
            agent_action = probs.sample().item()

            expert_logits, _ = expert(torch.FloatTensor(obs).to(device))
            expert_policy = F.softmax(expert_logits, dim=1)
            expert_action = torch.argmax(expert_policy).item()

            next_obs, _, done, _ = env.step(expert_action)

            if agent_action == expert_action:
                corr += 1

            if done:
                obs = env.reset()

            if steps == max_len:
                break
            steps += 1

    return corr / steps



# make env
env_id = 'mario83'
#env, expert = get_env_and_model(env_id)
env = create_train_env(8,3, SIMPLE_MOVEMENT,output_path=None)


expert = PPO(env.observation_space.shape[0], len(SIMPLE_MOVEMENT))


if torch.cuda.is_available():
    expert.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format('trained_models', 8,3)))
    expert.cuda()
else:
    expert.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format('trained_models', 8,3),
                                         map_location=lambda storage, loc: storage))
expert.eval()
expert.to(device)

# load data
#data_path = './marioCROP.npz'
#expert_trajs = np.load(data_path)
holdout_arrs = np.load('mario84_holdout.npz')
holdout_expert_observations = holdout_arrs['expert_observations']
holdout_expert_actions = holdout_arrs['expert_actions']

holdout_expert_dataset = ExpertDataSet(holdout_expert_observations, holdout_expert_actions)

eval_loader = torch.utils.data.DataLoader(
    dataset=holdout_expert_dataset, batch_size=128, shuffle=False
)


# make discriminator
D = Disc().to(device)
D_opt = Adam(D.parameters(), lr=1e-5)
D_criterion = nn.BCELoss()

# make policy net
'''
pi = models.resnet18(pretrained=False)
pi.conv1 =
pi.fc = nn.Sequential(
    nn.Linear(512,6)
)
pi = pi.to(device)
'''
class AC(nn.Module):
    def __init__(self, num_actions):
        super(AC, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2,padding=3,bias=False)
        num_ftrs = self.resnet.fc.in_features#()
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, num_actions)
        self.fc2 = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        pi_logits = self.fc1(x)
        value = self.fc2(x).reshape(-1)

        pi = Categorical(logits=pi_logits)#.flatten())

        return pi, value
    
pi_v = AC(6)


epochs = 2
n_eps = 1000


def rollout(pi,n_eps=1000,max_len=10000,exp=False):
    # rollout pi
    states = []
    next_states = []
    obs = env.reset()
    steps = 0
    done = False
    for j in range(n_eps):
        while not done: 
            if not exp:
                probs, _ = pi(torch.FloatTensor(obs).to(device))#.permute(0,3,1,2).to(device))
                action = probs.sample().item()
                
                
            else:
                logits, _ = pi(torch.FloatTensor(obs).to(device))
                policy = F.softmax(logits, dim=1)
                action = torch.argmax(policy).item()
                
            next_obs, _, done, _ = env.step(action)


            states.append(obs[:,-1,:,:])
            next_states.append(next_obs[:,-1,:,:])
        
            if done:
                obs = env.reset()

            if steps == max_len:
                break
            steps += 1

    return states, next_states

#def train(n_epochs=3,n_eps=3):

expert_s, expert_sp = rollout(expert,exp=True)
expert_dset = GAIFODataSet(expert_s, expert_sp)
expert_loader = torch.utils.data.DataLoader(
        dataset=expert_dset, batch_size=64, shuffle=True#, **kwargs,
)
    
 
n_epochs = 1000
early_stopping_thresh = 20
v_fids = []
for i in tqdm(range(n_epochs)):
    agent_s, agent_sp = rollout(pi_v)
    agent_dset = GAIFODataSet(agent_s, agent_sp)
    agent_loader = torch.utils.data.DataLoader(
        dataset=agent_dset, batch_size=64, shuffle=True#, **kwargs,
    )
    
    # update discriminator
    for bat, (a_s, a_sp) in enumerate(agent_loader):
        x_s, x_sp = next(iter(expert_loader))
        
        real_in = torch.cat([x_s, x_sp],dim=1).to(device)
        fake_in = torch.cat([a_s, a_s],dim=1).to(device)

        
        
        real_logits = D(real_in.flatten(1)) 
        fake_logits = D(fake_in.flatten(1)) 

        #loss = -(torch.mean(torch.log(real_logits)) + torch.mean(torch.log(torch.log(1-fake_logits))))
        ones = torch.ones(real_logits.shape).to(device)
        zeros = torch.zeros(fake_logits.shape).to(device)

        
        err_real = D_criterion(real_logits, ones)
        err_real.backward()
        
        err_fake = D_criterion(fake_logits, zeros)
        err_fake.backward()

        
        err_D = (err_fake + err_real) / 2.

        D_opt.step()
        
    # train pi
    #trpo(env,pi,epochs=2, num_rollouts=2, render_frequency=None)
    #real_input = torch.concat
    ppo = PPO_MARIO(env_id,pi_v, D)
    ppo.run_training_loop()
    try:
        ppo.destroy()
    except Exception as e:
        print(e)
        pass
    
    v_fid =  validate(pi_v,expert,n_eps=5)
    print('\nEpoch ', i, ' fidelity: ',v_fid)
    if len(v_fids) > early_stopping_thresh:
        if v_fid < max(v_fids[early_stopping_thresh:]):
            print('Early Stopping')
            break
        if v_fid > max(v_fids):
            print('New best model')
            torch.save(pi_v,env_id+'_best_pi.pt')
    v_fids.append(v_fid)
    
pi_v = torch.load(env_id+'_best_pi.pt')
#print('final fidelity: ', test(pi_v,device,eval_loader))#validate(pi_v,expert))

acc, sens = flip_fidelity(pi_v.to(device),n_flip=10000,num_interactions=30000)
print('acc, sens: ', acc, sens)
