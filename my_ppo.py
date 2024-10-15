import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from multiprocessing import Manager
import threading as td
import parameters as params

#####################  hyper parameters  ####################

RANDOMSEED = 2  # random seed
EP_MAX = 10000  # total number of episodes for training
EP_LEN = 1000  # total number of steps for each episode
GAMMA = 0.99  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 1024  # update batchsize
A_UPDATE_STEPS = 50  # actor update steps
C_UPDATE_STEPS = 50  # critic update steps
EPS = 1e-8  # numerical residual
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.1),  # Clipped surrogate objective
][1]  # choose the method for optimization

###############################  PPO  ####################################

device_idx = 0
if params.GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1.):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.linear1 = nn.Linear(params.state_dim1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std = AddBias(torch.zeros(num_actions))
        self.num_actions = num_actions
        self.action_range = action_range

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.action_range * torch.sigmoid(self.mean_linear(x))
        zeros = torch.zeros(mean.size(), device=state.device)
        log_std = self.log_std(zeros)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device, non_blocking=True)
        mean, log_std = self.forward(state)
        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = normal.sample()
        action = torch.clamp(action, 0, self.action_range)
        return action.squeeze(0)

    def sample_action(self):
        return torch.FloatTensor(self.num_actions).uniform_(0, 1).numpy()

class PPO(object):
    def __init__(self, state_dim, action_dim, hidden_dim=128, a_lr=params.actorlr, c_lr=params.criticlr):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range=1.).to(device)
        self.actor_old = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range=1.).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c_lr)
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)

    def a_train(self, s, a, adv):
        mu, log_std = self.actor(s)
        log_std = torch.clamp(log_std, min=-20, max=2)
        pi = Normal(mu, torch.exp(log_std))
        mu_old, log_std_old = self.actor_old(s)
        oldpi = Normal(mu_old, torch.exp(log_std_old))
        ratio = torch.exp(pi.log_prob(a)) / (torch.exp(oldpi.log_prob(a)) + EPS)
        surr = ratio * adv
        if METHOD['name'] == 'kl_pen':
            lam = METHOD['lam']
            kl = torch.distributions.kl.kl_divergence(oldpi, pi).mean()
            aloss = -((surr - lam * kl).mean())
        else:  
            aloss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv))
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()
        self.scheduler_actor.step()

    def update_old_pi(self):
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

    def cal_adv(self, s, cumulative_r):
        advantage = cumulative_r - self.critic(s)
        return advantage.detach()

    def update(self, s, a, r):
        s = s.to(device, non_blocking=True) if not s.is_cuda else s
        a = a.to(device, non_blocking=True) if not a.is_cuda else a
        r = r.to(device, non_blocking=True) if not r.is_cuda else r
        adv = self.cal_adv(s, r)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(A_UPDATE_STEPS):
            self.a_train(s, a, adv)
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
        with torch.no_grad():
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
