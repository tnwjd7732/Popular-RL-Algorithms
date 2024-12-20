
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
import no_RL_scheme as schemes
from torch.nn import MultiheadAttention

# Hyper Parameters
MAX_EPI=10000
MAX_STEP = 10000
SAVE_INTERVAL = 20
TARGET_UPDATE_INTERVAL = 5

BATCH_SIZE = 128
REPLAY_START_SIZE = 10000
EPSILON_DECAY = 30000
GAMMA = 0.95
EPSILON = 0.05  # if not using epsilon scheduler, use a constant
if params.pre_trained == True:
    EPSILON_START = 0.5
else:   
    EPSILON_START = 0.5
EPSILON_END = 0.005
if params.cloud < 1:
    REPLAY_BUFFER_SIZE = 100000
else:
    REPLAY_BUFFER_SIZE = 100000


greedy = schemes.Greedy()
nearest = schemes.Nearest()
device_idx = 0
if params.GPU:
    #device = torch.device("mps")
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("DQN Device: ", device)

class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):
        """A scheduler for epsilon-greedy strategy.

        :param eps_start: starting value of epsilon, default 1. as purely random policy 
        :type eps_start: float
        :param eps_final: final value of epsilon
        :type eps_final: float
        :param eps_decay: number of timesteps from eps_start to eps_final
        :type eps_decay: int
        """
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self, frame_idx):
        self.current_frame_idx = frame_idx
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * delta_frame_idx / self.eps_decay)
        if frame_idx%999 == 0:
            params.epsilon_logging.append(self.epsilon)
        
    def get_epsilon(self):
        #print("epsilon:", self.epsilon)
        return self.epsilon



class QNetwork(nn.Module):
    def __init__(self, act_shape, obs_shape, hidden_size=params.hidden_dim, num_heads=4, dropout_rate=0.1):
        super(QNetwork, self).__init__()

        # 1x1 Conv for remains and hops (Server State Features)
        self.conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=1, stride=1)

        # Fully Connected Layer for Task Information
        self.task_fc = nn.Linear(4, hidden_size)

        # Attention: Key/Value from Server State, Query from Task
        self.units = 16 * (params.maxEdge + 1)
        self.query_dim = hidden_size
        
        assert self.query_dim % num_heads == 0, "Query dim must be divisible by num_heads"
        assert self.units % num_heads == 0, "Key/Value dim must be divisible by num_heads"

        # Cross Attention: Task-Server State Interaction
        self.cross_attention = MultiheadAttention(embed_dim=self.query_dim, kdim=self.units, vdim=self.units, num_heads=num_heads)

        # Fully Connected Layers for Q-value prediction
        self.fc1 = nn.Linear(self.query_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Additional FC Layer
        self.output = nn.Linear(hidden_size, act_shape)

        # Dropout and Layer Normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, state):
        # Split state into remains, hops, and task information
        remain = state[:, :params.maxEdge+1].unsqueeze(1)
        hop = state[:, params.maxEdge+1:(params.maxEdge+1)*2].unsqueeze(1)
        task_and_frac = state[:, (params.maxEdge+1)*2:]

        # Server State Features
        server_features = torch.cat((remain, hop), dim=1)
        x = F.relu(self.conv(server_features))
        x = x.view(state.size(0), -1)

        # Task Information Embedding
        task_embed = F.relu(self.task_fc(task_and_frac))

        # Multihead Cross Attention
        query = task_embed.unsqueeze(0)
        key_value = x.unsqueeze(0)
        attn_output, _ = self.cross_attention(query, key_value, key_value)

        # Fully Connected Layers with Layer Normalization and Dropout
        combined = F.relu(self.ln1(self.fc1(attn_output.squeeze(0) + task_embed)))
        combined = self.dropout(combined)
        combined = F.relu(self.ln2(self.fc2(combined)))
        combined = F.relu(self.fc3(combined))  # Additional FC Layer for Depth
        q_values = self.output(combined)

        return q_values


transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, samples):
        # Append when the buffer is not full but overwrite when the buffer is full
        wrap_tensor = lambda x: torch.tensor([x])
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*map(wrap_tensor, samples)))
        else:
            self.buffer[int(self.location)] = transition(*map(wrap_tensor, samples))

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class DQN(object):
    def __init__(self, env, action_dim, state_dim):
        self.action_shape = action_dim
        self.obs_shape = state_dim
        self.eval_net, self.target_net = QNetwork(self.action_shape, self.obs_shape).to(device), QNetwork(self.action_shape, self.obs_shape).to(device)
        self.learn_step_counter = 0                                     # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=params.dqnlr)
        self.scheduler1 = optim.lr_scheduler.StepLR(self.optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)
        self.loss_func = nn.MSELoss()
        self.epsilon_scheduler = EpsilonScheduler(EPSILON_START, EPSILON_END, EPSILON_DECAY)
        self.updates = 0

    def choose_action(self, x, test):
        # x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        # if np.random.uniform() > EPSILON:   # greedy

        if test == 1:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]     # return the argmax
        else: #training phase
            epsilon = self.epsilon_scheduler.get_epsilon()
            #print("epsilon: ", epsilon)
            if np.random.uniform() > epsilon: 
                actions_value = self.eval_net.forward(x)
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]     # return the argmax
                # print(action)
            else:   # random
                if params.cloud <= 1:
                    action = np.random.randint(0, params.action_dim2)
                else:
                    action = np.random.randint(0, params.wocloud_action_dim2)
                    '''
                    action1, action2 = greedy.choose_action(1, params.stepnum)
                    action2 = torch.tensor(action2).unsqueeze(0).numpy()[0]
                    if action2 in params.mycluster:
                        action = params.mycluster.index(action2)
                    else:
                        action1, action2 = nearest.choose_action()  # action2가 리스트에 없을 경우 0으로 설정
                        action = torch.tensor(action2).unsqueeze(0).numpy()[0]
                        action = params.mycluster.index(action)
                    '''
                    
                    #action = params.CH_glob_ID
                    #print(action)
                
        return action

    def learn(self, sample,):
        # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
        batch_samples = transition(*zip(*sample))

        # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
        states = torch.cat(batch_samples.state).float().to(device)
        next_states = torch.cat(batch_samples.next_state).float().to(device)
        actions = torch.cat(batch_samples.action).to(device)
        rewards = torch.cat(batch_samples.reward).float().to(device)
        is_terminal = torch.cat(batch_samples.is_terminal).to(device)
        # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
        # Note: policy_network output Q-values for all the actions of a state, but all we need is the A_t taken at time t
        # in state S_t.  Thus we gather along the columns and get the Q-values corresponds to S_t, A_t.
        # Q_s_a is of size (BATCH_SIZE, 1).
        Q = self.eval_net(states) 
        Q_s_a=Q.gather(1, actions)

        # Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
        # Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
        # values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
        # to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

        # Get the indices of next_states that are not terminal
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        # Select the indices of each row
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

        Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
        if len(none_terminal_next_states) != 0:
            # below is the original code, but I modify this DQN code to DDQN (Double DQN)
            Q_s_prime_a_prime[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)
            # Choose action using eval_net (Double DQN)
            # Modify from here to...
            #next_action = self.eval_net(none_terminal_next_states).detach().max(1)[1].unsqueeze(1)

            # Use target_net to calculate Q value for the chosen action
            #Q_s_prime_a_prime[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).gather(1, next_action)
            # ... here (End of modification)
        # Q_s_prime_a_prime = self.target_net(next_states).detach().max(1, keepdim=True)[0]  # this one is simpler regardless of terminal state
        Q_s_prime_a_prime = (Q_s_prime_a_prime-Q_s_prime_a_prime.mean())/ (Q_s_prime_a_prime.std() + 1e-5)  # normalization
        
        # Compute the target
        target = rewards + GAMMA * Q_s_prime_a_prime

        # Update with loss
        # loss = self.loss_func(target.detach(), Q_s_a)
        loss = F.smooth_l1_loss(target.detach(), Q_s_a)
        # Zero gradients, backprop, update the weights of policy_net
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


    def update_target(self, ):
        """
        Update the target model when necessary.
        """
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
def rollout(env, model):
    r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
    log = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    #print('\nCollecting experience...')
    total_step = 0
    for epi in range(MAX_EPI):
        s=env.reset()
        epi_r = 0
        epi_loss = 0
        for step in range(MAX_STEP):
            # env.render()
            total_step += 1
            a = model.choose_action(s)
            s_, r, done, info = env.step(a)
            # r_buffer.add(torch.tensor([s]), torch.tensor([s_]), torch.tensor([[a]]), torch.tensor([[r]], dtype=torch.float), torch.tensor([[done]]))
            r_buffer.add([s,s_,[a],[r],[done]])
            model.epsilon_scheduler.step(total_step)
            epi_r += r
            if total_step > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                sample = r_buffer.sample(BATCH_SIZE)
                loss = model.learn(sample)
                epi_loss += loss
            if done:
                break
            s = s_
        #print('Ep: ', epi, '| Ep_r: ', epi_r, '| Steps: ', step, f'| Ep_Loss: {epi_loss:.4f}', )
        log.append([epi, epi_r, step])
        # if epi % SAVE_INTERVAL == 0:
            # model.save_model()
            # np.save('log/'+timestamp, log)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    #print(env.observation_space, env.action_space)
    model = DQN(env)
    rollout(env, model)