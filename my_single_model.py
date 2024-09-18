import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
from IPython.display import clear_output
import env
import clustering
import parameters as params
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
class PPO_single(nn.Module):
    def __init__(self, state_dim, continuous_action_dim, discrete_action_dim, hidden_dim, action_range=1.0):
        super(PPO_single, self).__init__()
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dim = discrete_action_dim
        self.action_range = action_range
        
        # Actor 네트워크 (continuous + discrete)
        self.actor_continuous = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, continuous_action_dim),
            nn.Sigmoid()  # Continuous action의 출력 범위를 [0, 1]로 제한
        )

        self.actor_discrete = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, discrete_action_dim),
            nn.Softmax(dim=-1)  # Discrete action은 확률 분포로 변환
        )

        # Critic 네트워크
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # 상태의 가치를 출력
        )

        self.actor_old_continuous = copy.deepcopy(self.actor_continuous)
        self.actor_old_discrete = copy.deepcopy(self.actor_discrete)

        self.actor_optimizer = optim.Adam(list(self.actor_continuous.parameters()) + list(self.actor_discrete.parameters()), lr=params.actorlr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params.criticlr)
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma)

        


    def forward(self, state):
        mean = self.actor_continuous(state)
        log_std = torch.zeros_like(mean)  # 기본적으로 std는 1로 설정 (여기서는 로그 표준편차를 0으로 설정)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  
        mean, log_std = self.forward(state)  

        if deterministic:
            action = mean  
        else:
            std = log_std.exp()  
            normal = Normal(mean, std)  
            action = normal.sample()  

        action = torch.clamp(action, 0, self.action_range)  
        return action.squeeze(0).cpu().detach().numpy() 

    def choose_action(self, state, deterministic=False):
        continuous_action = self.get_action(state, deterministic)

        state = torch.FloatTensor(state).to(device)
        discrete_action_probs = self.actor_discrete(state)
        discrete_action = torch.multinomial(discrete_action_probs, 1).item() 

        return continuous_action, discrete_action

    def evaluate(self, state, continuous_action, discrete_action):
        state = torch.FloatTensor(state).to(device)

        continuous_action_mean = self.actor_continuous(state)
        continuous_action_var = torch.full((self.continuous_action_dim,), 0.5).to(device)  
        continuous_cov_mat = torch.diag(continuous_action_var).unsqueeze(dim=0)
        continuous_dist = MultivariateNormal(continuous_action_mean, continuous_cov_mat)
        continuous_logprob = continuous_dist.log_prob(continuous_action)

        discrete_action_probs = self.actor_discrete(state)
        discrete_dist = Categorical(discrete_action_probs)
        discrete_logprob = discrete_dist.log_prob(discrete_action)
        discrete_entropy = discrete_dist.entropy()

        state_value = self.critic(state)

        return continuous_logprob, discrete_logprob, state_value, discrete_entropy

    def update(self, states, actions_continuous, actions_discrete, rewards):
        states = torch.FloatTensor(states).to(device)
        actions_continuous = torch.FloatTensor(actions_continuous).to(device)
        actions_discrete = torch.LongTensor(actions_discrete).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        advantages = rewards - self.critic(states)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)  

        for _ in range(5):  # A_UPDATE_STEPS
            continuous_logprob, discrete_logprob, state_value, _ = self.evaluate(states, actions_continuous, actions_discrete)

            continuous_logprob_old, discrete_logprob_old, _, _ = self.evaluate(states, actions_continuous, actions_discrete)

            ratio_continuous = torch.exp(continuous_logprob - continuous_logprob_old)
            ratio_discrete = torch.exp(discrete_logprob - discrete_logprob_old)

            ratio = ratio_continuous * ratio_discrete

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

        for _ in range(5):  # C_UPDATE_STEPS
            state_value = self.critic(states)
            critic_loss = F.mse_loss(state_value, rewards)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.update_old_pi()

    def update_old_pi(self):
        self.actor_old_continuous.load_state_dict(self.actor_continuous.state_dict())
        self.actor_old_discrete.load_state_dict(self.actor_discrete.state_dict())

    def get_v(self, state):
        state = torch.FloatTensor(state).to(device)
        value = self.critic(state)
        return value.detach().cpu().numpy()

    def save_model(self, path):
        """모델의 가중치를 저장하는 함수"""
        torch.save(self.actor_continuous.state_dict(), path + '_actor_continuous')
        torch.save(self.actor_discrete.state_dict(), path + '_actor_discrete')
        torch.save(self.critic.state_dict(), path + '_critic')
        torch.save(self.actor_old_continuous.state_dict(), path + '_actor_old_continuous')
        torch.save(self.actor_old_discrete.state_dict(), path + '_actor_old_discrete')

    def load_model(self, path):
        """저장된 가중치를 불러오는 함수"""
        self.actor_continuous.load_state_dict(torch.load(path + '_actor_continuous'))
        self.actor_discrete.load_state_dict(torch.load(path + '_actor_discrete'))
        self.critic.load_state_dict(torch.load(path + '_critic'))
        self.actor_old_continuous.load_state_dict(torch.load(path + '_actor_old_continuous'))
        self.actor_old_discrete.load_state_dict(torch.load(path + '_actor_old_discrete'))

        # 모델을 평가 모드로 설정
        self.actor_continuous.eval()
        self.actor_discrete.eval()
        self.critic.eval()
        self.actor_old_continuous.eval()
        self.actor_old_discrete.eval()


# 환경 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = env.Env()
cluster = clustering.Clustering()

ppo_single = PPO_single(params.single_state_dim, params.single_action1_dim, params.single_action2_dim, hidden_dim=params.hidden_dim)

# 파서 설정
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

# 로그 데이터
rewards1 = []
losses = []
success_rate = []
action1_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 그래프 그리기 함수
def plot():
    clear_output(True)
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(27, 8))

    font_size = params.font_size # 폰트 크기 설정

    # 첫번째 서브플롯에 rewards
    ax1.plot(rewards1)
    ax1.set_xlabel('Episode', fontsize=font_size)
    ax1.set_ylabel('Reward1', fontsize=font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size-5)

    # 세 번째 서브플롯에 action1 distribution 바 그래프로 그리기
    indices = list(range(10))  # 인덱스 0~9
    ax3.bar(indices, action1_distribution[:10])  # 인덱스 0~9까지의 데이터를 바 그래프로
    ax3.set_xlabel('Action [0-9]', fontsize=font_size)
    ax3.set_ylabel('Count', fontsize=font_size)
    ax3.tick_params(axis='both', which='major', labelsize=font_size-5)

    # 네 번째 서브플롯에 wrong count를 첫 번째 요소를 제외하고 그림
    ax4.plot(params.wrong_cnt[1:])
    ax4.set_xlabel('Episode', fontsize=font_size)
    ax4.set_ylabel('Invalid action count', fontsize=font_size)
    ax4.tick_params(axis='both', which='major', labelsize=font_size-5)

    # Cloud 선택 횟수
    ax5.plot(params.cloud_cnt[1:])
    ax5.set_xlabel('Episode', fontsize=font_size)
    ax5.set_ylabel('Cloud selection', fontsize=font_size)
    ax5.tick_params(axis='both', which='major', labelsize=font_size-5)

    # Epsilon 로그
    ax6.plot(params.epsilon_logging)
    ax6.set_xlabel('Episode', fontsize=font_size)
    ax6.set_ylabel('Epsilon (DDQN)', fontsize=font_size)
    ax6.tick_params(axis='both', which='major', labelsize=font_size-5)

    # Success rate
    ax7.plot(success_rate)
    ax7.set_xlabel('Episode', fontsize=font_size)
    ax7.set_ylabel('Success rate', fontsize=font_size)
    ax7.tick_params(axis='both', which='major', labelsize=font_size-5)
    
    # Losses
    ax8.plot(losses)
    ax8.set_xlabel('Episode', fontsize=font_size)
    ax8.set_ylabel('Loss', fontsize=font_size)
    ax8.tick_params(axis='both', which='major', labelsize=font_size-5)

    plt.show()
# 학습 루프
if args.train:
    x = -1
    y = 0
    fail = 0
    for i in range(params.numEdge):
        if (i % params.grid_size == 0):
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

    buffer = {
        'state': [],
        'continuous_action': [],
        'discrete_action': [],
        'reward': [],
        'done': []
    }

    total_step = 0
    for eps in range(params.EPS):
        fail = 0
        cluster.form_cluster()  # 클러스터 형성
        cluster.visualize_clusters()  # 클러스터 시각화
        _, state = env.reset(-1, 1)  # 환경 초기화
        episode_reward = 0
        eps_r1 = 0
        t0 = time.time()

        action1_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for step in range(params.STEP * params.numVeh):
            total_step += 1

            # PPO로 continuous 및 discrete action 선택
            continuous_action, discrete_action = ppo_single.choose_action(state)
            action1_distribution[min(int(continuous_action * 10), 9)] += 1

            # 환경에 두 가지 액션 전달
            reward, r1, done = env.step_single(continuous_action, discrete_action, step, 1)

            # NaN 값 체크
            if np.isnan(reward) or np.isnan(r1):
                reward = 0
                r1 = 0

            # 버퍼에 저장
            buffer['state'].append(state)
            buffer['continuous_action'].append(continuous_action)
            buffer['discrete_action'].append(discrete_action)
            buffer['reward'].append(r1)
            buffer['done'].append(done)

            episode_reward += reward
            eps_r1 += r1

            if reward == 0:
                fail += 1

            # PPO 업데이트
            if (step + 1) % params.ppo_batch == 0:
                if done:
                    v_s_ = 0
                else:
                    v_s_ = ppo_single.get_v(state)[0]  # 마지막 상태의 가치를 예측

                discounted_r = []
                for r, d in zip(buffer['reward'][::-1], buffer['done'][::-1]):
                    v_s_ = r + params.GAMMA * v_s_ * (1 - d)
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                # 버퍼에 있는 데이터를 업데이트하기 위한 형식으로 변환
                br = np.array(discounted_r)[:, np.newaxis]
                bs = np.vstack(buffer['state'])
                ba_continuous = np.vstack(buffer['continuous_action'])
                ba_discrete = np.vstack(buffer['discrete_action'])

                # PPO 업데이트
                buffer['state'], buffer['continuous_action'], buffer['discrete_action'], buffer['reward'], buffer['done'] = [], [], [], [], []
                ppo_single.update(bs, ba_continuous, ba_discrete, br)

        # 로그 및 모델 저장
        if eps % 5 == 0 and eps > 0:
            plot()
            ppo_single.save_model(params.ppo_single_path)

        rewards1.append(eps_r1)

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        success_rate.append(success_ratio)
