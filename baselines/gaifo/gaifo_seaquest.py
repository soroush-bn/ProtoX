import gym
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm
import numpy as np
from dagger import *
#from trpo_atari import trpo
from ppo import *
from train_utils import *

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

    
def validate(pi,expert,n_eps=30,max_len=10000):
    # rollout pi
    states = []
    next_states = []
    obs = env.reset()
    steps = 0
    done = False
    corr = 0
    for j in range(n_eps):
        while not done: 
            probs, _ = pi(torch.FloatTensor(obs).permute(0,3,1,2).to(device))
            agent_action = [probs.sample().item()]
                
            expert_action = expert.predict(obs)#torch.FloatTensor(obs).permute(0,3,1,2).to(device))
            
            next_obs, _, done, _ = env.step(expert_action)

            if agent_action[0] == expert_action[0]:
                corr += 1
            
            if done:
                obs = env.reset()

            if steps == max_len:
                break
            steps += 1

    return corr / steps


# make env
env_id = 'SeaquestNoFrameskip-v4'
env, expert = get_env_and_model(env_id)
    
# load data
data_path = './seaqest.npz'
expert_trajs = np.load(data_path)


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
    
pi_v = AC(18)


epochs = 2
n_eps = 1000


def rollout(pi,n_eps=1000,max_len=10000):
    # rollout pi
    states = []
    next_states = []
    obs = env.reset()
    steps = 0
    done = False
    for j in range(n_eps):
        while not done: 
            try:
                #logits, _ = pi(torch.FloatTensor(obs).permute(0,3,1,2).to(device))
                #_, action = torch.max(logits.data, 1)
                probs, _ = pi(torch.FloatTensor(obs).permute(0,3,1,2).to(device))
                action = [probs.sample().item()]
                
                
            except Exception as e:
                action = pi.predict(obs)#torch.FloatTensor(obs).permute(0,3,1,2).to(device))
            
            next_obs, _, done, _ = env.step(action)

            states.append(obs[:,:,:,-1])
            next_states.append(next_obs[:,:,:,-1])
        
            if done:
                obs = env.reset()

            if steps == max_len:
                break
            steps += 1

    return states, next_states

#def train(n_epochs=3,n_eps=3):

expert_s, expert_sp = rollout(expert)
expert_dset = GAIFODataSet(expert_s, expert_sp)
expert_loader = th.utils.data.DataLoader(
        dataset=expert_dset, batch_size=64, shuffle=True#, **kwargs,
)
    

n_epochs = 0#1000
early_stopping_thresh = 20
v_fids = [0]
es_ct = 20
for i in range(n_epochs):
    agent_s, agent_sp = rollout(pi_v)
    agent_dset = GAIFODataSet(agent_s, agent_sp)
    agent_loader = th.utils.data.DataLoader(
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
    ppo = PPO2(env_id,pi_v, D)
    ppo.run_training_loop()
    try:
        ppo.destroy()
    except Exception as e:
        print(e)
        pass
    
    #print('\nEpoch ', i, ' fidelity: ', test(pi_v,expert))

    v_fid = validate(pi_v, expert, n_eps=5)
    print('\nEpoch ', i, ' fidelity: ', v_fid)
    if v_fid < max(v_fids):
        es_ct -= 1
    if es_ct == 0:
        print('Early Stopping')
        break
    if v_fid > max(v_fids):
        print('New best model, reset es ct')
        es_ct = early_stopping_thresh
        torch.save(pi_v, env_id+'_best_pi.pt')
    v_fids.append(v_fid)

pi_v = torch.load(env_id + '_best_pi.pt')
#acc,sens = flip_fidelity(pi_v.to(device), n_flip=10000,num_interactions=30000)
#print('acc,sens: ', acc, sens)
#print('final fidelity: ', test(pi_v,expert))


flip_arrs = np.load('seaqest_flip.npz')
flip_observations = flip_arrs['flip_observations']
flip_actions = flip_arrs['flip_actions']

flip_dataset = ExpertDataSet(flip_observations, flip_actions)
flip_loader = th.utils.data.DataLoader(
    dataset=flip_dataset, batch_size=128, shuffle=False
)

sens = test(pi_v,device,flip_loader)
print('sensitivity: ', sens)
