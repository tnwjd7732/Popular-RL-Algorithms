
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from IPython.display import display
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import threading as td
import parameters as params

#####################  hyper parameters  ####################

RANDOMSEED = 2  # random seed

EP_MAX = 10000  # total number of episodes for training
EP_LEN = 1000  # total number of steps for each episode
GAMMA = 0.99  # reward discount
A_LR = 0.0001
  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 1024  # update batchsize
A_UPDATE_STEPS = 50  # actor update steps
C_UPDATE_STEPS = 50  # critic update steps
EPS = 1e-8  # numerical residual
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.1),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization

###############################  PPO  ####################################


device_idx = 0
if params.GPU:
    device = torch.device("mps")
    #device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("PPO Device: ", device)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# important modified! hidden layer enabled
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        # self.linear4.weight.data.uniform_(-init_w, init_w)
        # self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 1D Convolutional layer
        #self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=0)
        #self.flatten1 = nn.Flatten()
        #self.dense1 = nn.Linear(params.state_dim1, 1)
        self.linear1 = nn.Linear(params.state_dim1, hidden_dim)  # Adjust input size after convolution
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        #ß self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # Implementation 1
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        # Implementation 2: not dependent on latent features, reference:
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py
        self.log_std = AddBias(torch.zeros(num_actions))

        self.num_actions = num_actions
        self.action_range = action_range

    def forward(self, state):
        remain = state[:, :2].unsqueeze(1) 
        task = state[:, 2:]
        #print("remain: ", remain, "task: ", task)
        # 1D Convolutional layer
        #x1 = F.relu(self.conv1(remain))
        #x1 = self.flatten1(x1)
        #x1 = F.relu(self.dense1(x1))

        #x = torch.cat((x1, task), dim=1)  # Ensure second part is also 2D
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        

        mean = self.action_range * torch.sigmoid(self.mean_linear(x))

        zeros = torch.zeros(mean.size(), device=state.device)
        log_std = self.log_std(zeros)

        return mean, log_std

    
    def get_action(self, state, deterministic=False):
        '''mps로 device 설정하니까 여기서 커널 죽음'''
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        #print(state)
        if deterministic:
            action = mean
            #print(action)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = normal.sample()
        action = torch.clamp(action, 0, self.action_range)
        # action이 0.95보다 클 때 1로 변환, 나머지는 action 유지
        #action = torch.where(action > 0.95, torch.tensor(1.0, device=action.device), action)

        #print("> PPO state: ", state, "action: ", action)

        return action.squeeze(0)

    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(0, 1)
        return a.numpy()
        
class PPO(object):
    def __init__(self, state_dim, action_dim, hidden_dim=128, a_lr=params.actorlr, c_lr=params.criticlr):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range=1.).to(device)
        self.actor_old = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range=1.).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c_lr)
        
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)

        
        #print(self.actor, self.critic)

    def a_train(self, s, a, adv):
        mu, log_std = self.actor(s)
        
        # log_std의 값을 적절한 범위로 제한
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        # mu와 log_std에 nan 값이 없는지 확인
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            #print("Invalid mu detected: ", mu)
            raise ValueError("mu contains nan or inf values")
        
        if torch.isnan(log_std).any() or torch.isinf(log_std).any():
            #print("Invalid log_std detected: ", log_std)
            raise ValueError("log_std contains nan or inf values")
        
        # Normal 분포 생성
        pi = Normal(mu, torch.exp(log_std))

        mu_old, log_std_old = self.actor_old(s)
        oldpi = Normal(mu_old, torch.exp(log_std_old))

        # ratio = torch.exp(pi.log_prob(a) - oldpi.log_prob(a))
        ratio = torch.exp(pi.log_prob(a)) / (torch.exp(oldpi.log_prob(a)) + EPS)

        surr = ratio * adv
        if METHOD['name'] == 'kl_pen':
            lam = METHOD['lam']
            kl = torch.distributions.kl.kl_divergence(oldpi, pi)
            kl_mean = kl.mean()
            aloss = -((surr - lam * kl).mean())
        else:  # clipping method, find this is better
            aloss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv))
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()
        self.scheduler_actor.step()

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        Update old policy parameter
        :return: None
        '''
        for p, oldp in zip(self.actor.parameters(), self.actor_old.parameters()):
            oldp.data.copy_(p)


    def c_train(self, cumulative_r, s):
        v = self.critic(s)
        advantage = cumulative_r - v
        closs = (advantage**2).mean()
        self.critic_optimizer.zero_grad()
        closs.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        #print("Actor LR:", actor_lr, "Critic LR: ", critic_lr)

    def cal_adv(self, s, cumulative_r):
        advantage = cumulative_r - self.critic(s)
        return advantage.detach()

    def update(self, s, a, r):
        s = torch.FloatTensor(s).to(device)     
        a = torch.FloatTensor(a).to(device) 
        r = torch.FloatTensor(r).to(device)   

        adv = self.cal_adv(s, r)
        adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful, not always, minus mean is dangerous

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)    
            
        self.update_old_pi()
 

    def choose_action(self, s, deterministic=False):
        action1 = self.actor.get_action(s, deterministic=False)
        return action1.detach().cpu().numpy()
    
    def get_v(self, s):
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.FloatTensor(s).to(device)  
        # return self.critic(s).detach().cpu().numpy()[0, 0]
        return self.critic(s).squeeze(0).detach().cpu().numpy()


    def save_model(self, path):
        torch.save(self.actor.state_dict(), path+'_actor')
        torch.save(self.critic.state_dict(), path+'_critic')
        torch.save(self.actor_old.state_dict(), path+'_actor_old')

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path+'_actor'))
        self.critic.load_state_dict(torch.load(path+'_critic'))
        self.actor_old.load_state_dict(torch.load(path+'_actor_old'))

        self.actor.eval()
        self.critic.eval()
        self.actor_old.eval()
        
