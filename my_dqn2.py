import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random, numpy, argparse, logging, os
from collections import namedtuple
import numpy as np
import datetime, math
import gym
import parameters as params

# Hyper Parameters
MAX_EPI=10000
MAX_STEP = 10000
SAVE_INTERVAL = 20
TARGET_UPDATE_INTERVAL = 20

BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 2000

GAMMA = 0.95
EPSILON = 0.05  
EPSILON_START = 1.0
EPSILON_END = 0.0005
EPSILON_DECAY = 50000

device_idx = 0
if params.GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("DQN Device: ", device)

class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0

    def reset(self):
        self.ini_frame_idx = self.current_frame_idx

    def step(self, frame_idx):
        self.current_frame_idx = frame_idx
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * delta_frame_idx / self.eps_decay)
        if frame_idx % 999 == 0:
            params.epsilon_logging.append(self.epsilon)
        
    def get_epsilon(self):
        return self.epsilon

class QNetwork(nn.Module):
    def __init__(self, act_shape, obs_shape, hidden_size=128):
        super(QNetwork, self).__init__()

        self.conv_remain_hop = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.conv_remain_hop_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0)

        self.conv_task = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv_task_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        self.flatten = nn.Flatten()

        if params.cloud == 1:
            conv_output_length = (params.maxEdge + 1) // 2
        else: 
            conv_output_length = params.maxEdge // 2

        flattened_conv_out_size_remain_hop = 128 * (conv_output_length - 1)
        flattened_conv_out_size_task = 128 * (1)

        self.dense_remain_hop = nn.Linear(flattened_conv_out_size_remain_hop, 1)
        self.dense_task = nn.Linear(flattened_conv_out_size_task, 1)
        
        combined_input_size = 1 + 1 + 1
        self.linear1 = nn.Linear(combined_input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, act_shape)
    
    def forward(self, state):
        if params.cloud == 1:
            remain = state[:, :params.maxEdge+1].unsqueeze(1)
            hop = state[:, params.maxEdge+1:(params.maxEdge+1)*2].unsqueeze(1)
            task = state[:, (params.maxEdge+1)*2:(params.maxEdge+1)*2+3].unsqueeze(1)
            fraction = state[:, -1].unsqueeze(1)
        else: 
            remain = state[:, :params.maxEdge].unsqueeze(1)
            hop = state[:, params.maxEdge:(params.maxEdge)*2].unsqueeze(1)
            task = state[:, (params.maxEdge)*2:(params.maxEdge)*2+3].unsqueeze(1)
            fraction = state[:, -1].unsqueeze(1)

        combined_remain_hop = torch.cat((remain, hop), dim=1)

        x_remain_hop = F.relu(self.conv_remain_hop(combined_remain_hop))
        x_remain_hop = F.relu(self.conv_remain_hop_2(x_remain_hop))
        x_remain_hop = self.flatten(x_remain_hop)
        x_remain_hop = F.relu(self.dense_remain_hop(x_remain_hop))
        
        x_task = F.relu(self.conv_task(task))
        x_task = F.relu(self.conv_task_2(x_task))
        x_task = self.flatten(x_task)
        x_task = F.relu(self.dense_task(x_task))

        x = torch.cat((x_remain_hop, x_task, fraction), dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        q_values = self.output(x)
        
        return q_values

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, samples):
        wrap_tensor = lambda x: torch.tensor([x])
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*map(wrap_tensor, samples)))
        else:
            self.buffer[int(self.location)] = transition(*map(wrap_tensor, samples))
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQN(object):
    def __init__(self, env, action_dim, state_dim):
        self.action_shape = action_dim
        self.obs_shape = state_dim
        self.eval_net = QNetwork(self.action_shape, self.obs_shape).to(device)
        self.target_net = QNetwork(self.action_shape, self.obs_shape).to(device)
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=params.dqnlr)
        self.scheduler1 = optim.lr_scheduler.StepLR(self.optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)
        self.loss_func = nn.MSELoss()
        self.epsilon_scheduler = EpsilonScheduler(EPSILON_START, EPSILON_END, EPSILON_DECAY)
        self.updates = 0

    def choose_action(self, x, test):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device, non_blocking=True)
        if test == 1:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            epsilon = self.epsilon_scheduler.get_epsilon()
            if np.random.uniform() > epsilon:  
                actions_value = self.eval_net.forward(x)
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            else:  
                action = np.random.randint(0, self.action_shape)
        return action

    def learn(self, sample):
        batch_samples = transition(*zip(*sample))

        states = torch.cat(batch_samples.state).float().to(device, non_blocking=True)
        next_states = torch.cat(batch_samples.next_state).float().to(device, non_blocking=True)
        actions = torch.cat(batch_samples.action).to(device, non_blocking=True)
        rewards = torch.cat(batch_samples.reward).float().to(device, non_blocking=True)
        is_terminal = torch.cat(batch_samples.is_terminal).to(device, non_blocking=True)

        Q = self.eval_net(states)
        Q_s_a = Q.gather(1, actions)

        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

        Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
        if len(none_terminal_next_states) != 0:
            Q_s_prime_a_prime[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

        Q_s_prime_a_prime = (Q_s_prime_a_prime - Q_s_prime_a_prime.mean()) / (Q_s_prime_a_prime.std() + 1e-5)

        target = rewards + GAMMA * Q_s_prime_a_prime

        loss = F.smooth_l1_loss(target.detach(), Q_s_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler1.step()

        self.updates += 1
        if self.updates % TARGET_UPDATE_INTERVAL == 0:
            self.update_target()

        return loss.item()

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))
        self.eval_net.eval()

    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

def rollout(env, model):
    r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
    log = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    total_step = 0
    for epi in range(MAX_EPI):
        s = env.reset()
        epi_r = 0
        epi_loss = 0
        for step in range(MAX_STEP):
            total_step += 1
            a = model.choose_action(s, test=0)
            s_, r, done, info = env.step(a)
            r_buffer.add([s, s_, [a], [r], [done]])
            model.epsilon_scheduler.step(total_step)
            epi_r += r
            if total_step > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                sample = r_buffer.sample(BATCH_SIZE)
                loss = model.learn(sample)
                epi_loss += loss
            if done:
                break
            s = s_
        log.append([epi, epi_r, step])

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    model = DQN(env, env.action_space.n, env.observation_space.shape[0])
    rollout(env, model)
