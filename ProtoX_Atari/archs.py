import numpy as np
import torch as th
import torch.nn as nn
import torchvision.models as models

import torch_optimizer
from torch.autograd import Variable

import torch.nn.functional as F
import os
import time
import random
import sys
import torch
from vit_pytorch import ViT

from matplotlib import pyplot as plt
import math
import gc

import pickle
from torch.nn.utils import  prune

# import push
# import prune
# import train_and_test as tnt

import wandb
from skimage.transform import resize

#@title Mnih 2015 network arch
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#https://discuss.pytorch.org/t/positive-weights/19701/7
class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())

class Mnih2015(nn.Module):
    """CNN head similar to one used in Mnih 2015
       (Human-level control through deep reinforcement learning, Mnih 2015)"""
    def __init__(self, image_shape, num_channels, num_actions):
        super(Mnih2015, self).__init__()
        self.num_actions = num_actions
        '''
        self.conv1 = nn.Conv2d(num_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        c_out = self.conv3(self.conv2(self.conv1(torch.randn(1, num_channels, *image_shape))))
        self.conv3_size = np.prod(c_out.shape)
        #print("conv3: {}".format(self.conv3_size))

        self.fc1 = nn.Linear(self.conv3_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        '''

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.SELU(),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(nn.Linear(3136, 512), nn.SELU(), nn.Linear(512, num_actions))
    def forward(self, x):
        '''
        #x = x.squeeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(-1, self.conv3_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        '''
        return self.linear(self.cnn(x))

#@title mnih autoencoder

class MnihAE(nn.Module):
    def __init__(self):
        super(MnihAE, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.SELU(),           
        )

        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3),
            nn.SELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.SELU(),
            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4),
            nn.SELU() #remove this?
            #sigmoid????
        )

    def forward(self, x):
        enc = self.cnn(x)
        dec = self.dcnn(enc)
        return enc, dec

#@title protoResNet
#from receptive_field import compute_proto_layer_rf_info_v2

class ProtoResNet6Head(nn.Module):
    def __init__(self, prototype_shape=(60,512,3,3), num_actions=6,init_weights=True,beta=.05):
        super(ProtoResNet6Head, self).__init__()
        
        self.num_actions = num_actions
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape
        self.num_prototypes_per_action = self.num_prototypes // self.num_actions
        self.beta = beta

        print('THIS PROTOPNET IS USING SIM SCORES FOR LOGITS')
        
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7,stride=2,padding=3,bias=False)
        modules = list(resnet.children())[:-2]
        #print('modules: ', modules)
        self.convunit = nn.Sequential(*modules)
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        
        
        self.ll1 = PositiveLinear(self.num_prototypes_per_action, 1)
        self.ll2 = PositiveLinear(self.num_prototypes_per_action, 1)
        self.ll3 = PositiveLinear(self.num_prototypes_per_action, 1)
        self.ll4 = PositiveLinear(self.num_prototypes_per_action, 1)
        self.ll5 = PositiveLinear(self.num_prototypes_per_action, 1)
        self.ll6 = PositiveLinear(self.num_prototypes_per_action, 1)

        assert(self.num_prototypes % self.num_actions == 0)
        
        #one-hot matrix for prototype action label
        self.prototype_action_identity = torch.zeros(self.num_prototypes, self.num_actions)
        #num_prototypes_per_action = self.num_prototypes // self.num_actions
        for j in range(self.num_prototypes):
            self.prototype_action_identity[j, j // self.num_prototypes_per_action] = 1

        if init_weights:
            self._init_weights()
    
    def conv_features(self, x):
        return self.convunit(x)
    
    def prototype_distances(self, x):
        conv_output = self.conv_features(x)
        return torch.cdist(conv_output.flatten(1), self.prototype_vectors[None].flatten(2)).squeeze(0)

        
    def forward(self, x):
        distances = self.prototype_distances(x)
        sim_scores = torch.exp(-1*self.beta*torch.abs(distances))
        #logits = self.last_layer(sim_scores)
        
        logit1 = self.ll1(sim_scores[:, 0*self.num_prototypes_per_action:1*self.num_prototypes_per_action])
        logit2 = self.ll2(sim_scores[:, 1*self.num_prototypes_per_action:2*self.num_prototypes_per_action])
        logit3 = self.ll3(sim_scores[:, 2*self.num_prototypes_per_action:3*self.num_prototypes_per_action])
        logit4 = self.ll4(sim_scores[:, 3*self.num_prototypes_per_action:4*self.num_prototypes_per_action])
        logit5 = self.ll5(sim_scores[:, 4*self.num_prototypes_per_action:5*self.num_prototypes_per_action])
        logit6 = self.ll6(sim_scores[:, 5*self.num_prototypes_per_action:6*self.num_prototypes_per_action])
        
        logits = torch.cat([logit1,logit2,logit3,logit4,logit5,logit6],dim=1).to(device)
        
        return logits, distances
    
    def push_forward(self, x):
        conv_output = self.conv_features(x)
        dists = torch.cdist(conv_output.flatten(1), self.prototype_vectors[None].flatten(2)).squeeze(0)
                
        return conv_output, dists
    
    def select_action(self, x):
        action_prob, min_dists = self.forward(x)
        action = action_prob.multinomial(1)
        return action, min_dists
    
    def targeting_prob(self, x, labels):
        action_prob, _ = self.forward(x)
        return action_prob.gather(1, labels)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_action_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations 
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _init_weights(self):
        for m in self.convunit:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='selu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print('not setting incorrect weight strength due to positive linear!')
        #self.set_last_layer_incorrect_connection(incorrect_strength=-.5)

#@title protoNet2
#from receptive_field import compute_proto_layer_rf_info_v2

class ProtoNet2(nn.Module):
    def __init__(self, prototype_shape=(60,64,7,7), num_actions=6,init_weights=True,beta=.05):
        super(ProtoNet2, self).__init__()
        
        self.num_actions = num_actions
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape
        self.beta = beta

        print('THIS PROTOPNET IS NOT USING SIM SCORES FOR LOGITS')
        
        self.convunit = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.SELU(),
        )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_actions, bias=False)
        #self.last_layer = PositiveLinear(self.num_prototypes, 10)

        assert(self.num_prototypes % self.num_actions == 0)
        
        #one-hot matrix for prototype action label
        self.prototype_action_identity = torch.zeros(self.num_prototypes, self.num_actions)
        num_prototypes_per_action = self.num_prototypes // self.num_actions
        for j in range(self.num_prototypes):
            self.prototype_action_identity[j, j // num_prototypes_per_action] = 1

        if init_weights:
            self._init_weights()
    
    def conv_features(self, x):
        return self.convunit(x)
    
    def prototype_distances(self, x):
        conv_output = self.conv_features(x)
        return torch.cdist(conv_output.flatten(1), self.prototype_vectors[None].flatten(2)).squeeze(0)

        
    def forward(self, x):
        distances = self.prototype_distances(x)
        #sim_scores = torch.exp(-1*self.beta*torch.abs(distances))
        #logits = self.last_layer(sim_scores)
        logits = self.last_layer(distances)
        return logits, distances
        #return F.softmax(logits), min_distances
    
    def push_forward(self, x):
        conv_output = self.conv_features(x)
        dists = torch.cdist(conv_output.flatten(1), self.prototype_vectors[None].flatten(2)).squeeze(0)
                
        return conv_output, dists
    
    def select_action(self, x):
        action_prob, min_dists = self.forward(x)
        action = action_prob.multinomial(1)
        return action, min_dists
    
    def targeting_prob(self, x, labels):
        action_prob, _ = self.forward(x)
        return action_prob.gather(1, labels)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_action_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations 
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _init_weights(self):
        for m in self.convunit:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-.5)

#@title protoNetPrime
#from receptive_field import compute_proto_layer_rf_info_v2

class ProtoNetPrime(nn.Module):
    def __init__(self, prototype_shape=(60,64,7,7), 
                 num_actions=6,
                 init_weights=True,
                 protos=None):
        super(ProtoNetPrime, self).__init__()
        
        self.num_actions = num_actions
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape
        self.protos = protos
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.SELU(),
        )
        
        #self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_actions, bias=True)
        #self.last_layer = PositiveLinear(self.num_prototypes, 10)

        assert(self.num_prototypes % self.num_actions == 0)
        
        #one-hot matrix for prototype action label
        self.prototype_action_identity = torch.zeros(self.num_prototypes, self.num_actions)
        num_prototypes_per_action = self.num_prototypes // self.num_actions
        for j in range(self.num_prototypes):
            self.prototype_action_identity[j, j // num_prototypes_per_action] = 1

        if init_weights:
            self._init_weights()
    
    def conv_features(self, x):
        return self.cnn(x)
    
    def prototype_distances(self, x):
        conv_features = self.conv_features(x)
        dists = torch.cdist(conv_features.flatten(1), self.prototype_vectors[None].flatten(2)).squeeze(0)
          
        return dists

    def distance_2_similarity(self, distances):
        return -distances # linear; could use sth else
        #return 1 / (1 + torch.abs(distances))

    def forward(self, x):
        self.prototype_vectors = self.conv_features(self.protos)
        distances = self.prototype_distances(x)
        #min_distances = -F.max_pool2d(-distances, kernel_size = (distances.size()[2], distances.size()[3]))
        #min_distances = min_distances.view(-1, self.num_prototypes)
        #prototype_activations = self.distance_2_similarity(min_distances)
        #sim_scores = self.distance_2_similarity(distances)
        #logits = self.last_layer(prototype_activations)
        #return logits, min_distances
        logits = self.last_layer(distances)
        return logits, distances
    
    def push_forward(self, x):
        conv_output = self.conv_features(x)
        self.prototype_vectors = self.conv_features(self.protos)
        #distances = self._l2_convolution(conv_output)
        distances = self.prototype_distances(x)
        return conv_output, distances
    
    def select_action(self, x):
        action_prob, min_dists = self.forward(x)
        action = action_prob.multinomial(1)
        return action, min_dists
    
    def targeting_prob(self, x, labels):
        action_prob, _ = self.forward(x)
        return action_prob.gather(1, labels)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_action_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations 
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _init_weights(self):
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-.5)

#@title protoResNet
#from receptive_field import compute_proto_layer_rf_info_v2

class ProtoIsoResNet(nn.Module):
    def __init__(self,
                 prototype_shape=(60,512*3*3),
                 num_actions=6,
                 init_weights=True,
                 beta=.05,
                 tl=True,
                 frame_stack=4,
                 sim_method=0
    ):
        super(ProtoIsoResNet, self).__init__()
        
        self.num_actions = num_actions
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape
        self.beta = beta
        self.sim_method = sim_method
                
        print('THIS PROTOPNET IS USING SIM SCORES FOR LOGITS')
        
        if not tl:
            resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7,stride=2,padding=3,bias=False)
        else:
            resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7,stride=2,padding=3,bias=False)
            print('TL resnet encoder from deepcluster!')
            init_weights = False
            resnet_tl_state = torch.load('enc40.pt').state_dict()
            resnet.load_state_dict(resnet_tl_state,strict=False)
        

        modules = list(resnet.children())[:-2]
        
  
        self.convunit = nn.Sequential(*modules)

        for param in self.convunit.parameters():
            param.requires_grad = False
            
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)


        self.isometry = nn.Linear(512*3*3, prototype_shape[1], bias = False)
        if prototype_shape[1] == 512*3*3:
            self.isometry.weight.data.copy_(torch.eye(512*3*3))
        
        self.last_layer = nn.Linear(self.num_prototypes, self.num_actions, bias=False)
        #self.last_layer = PositiveLinear(self.num_prototypes, self.num_actions)

        assert(self.num_prototypes % self.num_actions == 0)
        
        #one-hot matrix for prototype action label
        self.prototype_action_identity = torch.zeros(self.num_prototypes, self.num_actions)
        num_prototypes_per_action = self.num_prototypes // self.num_actions
        for j in range(self.num_prototypes):
            self.prototype_action_identity[j, j // num_prototypes_per_action] = 1

        if init_weights:
            print('Doing weight init')
            self._init_weights()
    
    def conv_features(self, x):
        return self.convunit(x)
    
    def prototype_distances(self, x):
        conv_output = self.conv_features(x)
        conv_flat = conv_output.flatten(1)
        iso_trans = self.isometry(conv_flat)

        return torch.cdist(iso_trans, self.prototype_vectors[None]).squeeze(0)

        
    def forward(self, x):
        distances = self.prototype_distances(x)
        if self.sim_method == 0:
            sim_scores = torch.exp(-1*self.beta*torch.abs(distances))
        else:
            #eps too small
            sim_scores = torch.log((distances + 1)/(distances + 1e-10))
            
        logits = self.last_layer(sim_scores)
        #logits = self.last_layer(distances)
        return logits, distances
        #return F.softmax(logits), min_distances
    
    def push_forward(self, x):
        conv_output = self.conv_features(x)
        conv_flat = conv_output.flatten(1)
        iso_trans = self.isometry(conv_flat)
        
        dists = torch.cdist(iso_trans, self.prototype_vectors[None]).squeeze(0)
                
        return iso_trans, dists
    
    def select_action(self, x):
        action_prob, min_dists = self.forward(x)
        action = action_prob.multinomial(1)
        return action, min_dists
    
    def targeting_prob(self, x, labels):
        action_prob, _ = self.forward(x)
        return action_prob.gather(1, labels)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_action_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations 
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _init_weights(self):
        for m in self.convunit:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='selu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print('etting incorrect weight strength due to positive linear!')
        self.set_last_layer_incorrect_connection(incorrect_strength=-.5)



class ProtoIsoResNetFP(nn.Module):
    def __init__(self,
                 prototype_shape=(60,512*3*3),
                 num_actions=6,
                 init_weights=False,
                 beta=.05,
                 tl=False,
                 frame_stack=4,
                 sim_method=0,
                 fixed_protos = None
    ):
        super(ProtoIsoResNetFP, self).__init__()
        
        self.num_actions = num_actions
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape
        self.beta = beta
        self.sim_method = sim_method
        self.fixed_protos = fixed_protos
        
        print('THIS PROTOPNET IS USING SIM SCORES FOR LOGITS')
        
        if not tl:
            resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7,stride=2,padding=3,bias=False)
        else:
            resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7,stride=2,padding=3,bias=False)
            print('TL resnet encoder from deepcluster!')
            init_weights = False
            resnet_tl_state = torch.load('enc40.pt').state_dict()
            resnet.load_state_dict(resnet_tl_state,strict=False)
        

        modules = list(resnet.children())[:-2]
        
  
        self.convunit = nn.Sequential(*modules)

        for param in self.convunit.parameters():
            param.requires_grad = False
            
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=False)#false?

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)


        self.isometry = nn.Linear(512*3*3, prototype_shape[1], bias = False)
        if prototype_shape[1] == 512*3*3:
            self.isometry.weight.data.copy_(torch.eye(512*3*3))
        
        self.last_layer = nn.Linear(self.num_prototypes, self.num_actions, bias=False)
        #self.last_layer = PositiveLinear(self.num_prototypes, self.num_actions)

        #assert(self.num_prototypes % self.num_actions == 0)
        
        #one-hot matrix for prototype action label
        if init_weights:
            self.prototype_action_identity = torch.zeros(self.num_prototypes, self.num_actions)
            num_prototypes_per_action = self.num_prototypes // self.num_actions
            for j in range(self.num_prototypes):
                self.prototype_action_identity[j, j // num_prototypes_per_action] = 1

        if init_weights:
            print('Doing weight init')
            self._init_weights()
    
    def conv_features(self, x):
        return self.convunit(x)
    
    def prototype_distances(self, x):
        conv_output = self.conv_features(x)
        conv_flat = conv_output.flatten(1)
        iso_trans = self.isometry(conv_flat)

        return torch.cdist(iso_trans, self.prototype_vectors[None]).squeeze(0)

        
    def forward(self, x):
        if self.fixed_protos is not None:
            enc_prototypes = self.conv_features(self.fixed_protos)
            enc_prototypes = enc_prototypes.flatten(1)
            enc_prototypes = self.isometry(enc_prototypes)
            self.prototype_vectors.data.copy_(enc_prototypes)
        
        distances = self.prototype_distances(x)
        if self.sim_method == 0:
            sim_scores = torch.exp(-1*self.beta*torch.abs(distances))
        else:
            #eps too small
            sim_scores = torch.log((distances + 1)/(distances + 1e-10))
            
        logits = self.last_layer(sim_scores)
        #logits = self.last_layer(distances)
        return logits, distances
        #return F.softmax(logits), min_distances
    
    def push_forward(self, x):
        if self.fixed_protos is not None:
            enc_prototypes = self.conv_features(self.fixed_protos)
            enc_prototypes = enc_prototypes.flatten(1)
            enc_prototypes = self.isometry(enc_prototypes)
            self.prototype_vectors.data.copy_(enc_prototypes)
        
        conv_output = self.conv_features(x)
        conv_flat = conv_output.flatten(1)
        iso_trans = self.isometry(conv_flat)
        
        dists = torch.cdist(iso_trans, self.prototype_vectors[None]).squeeze(0)
                
        return iso_trans, dists
    
    def select_action(self, x):
        action_prob, min_dists = self.forward(x)
        action = action_prob.multinomial(1)
        return action, min_dists
    
    def targeting_prob(self, x, labels):
        action_prob, _ = self.forward(x)
        return action_prob.gather(1, labels)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_action_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations 
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _init_weights(self):
        for m in self.convunit:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='selu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print('etting incorrect weight strength due to positive linear!')
        self.set_last_layer_incorrect_connection(incorrect_strength=-.5)


 
