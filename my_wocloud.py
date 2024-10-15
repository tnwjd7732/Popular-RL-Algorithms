import my_ppo
import my_sac
import env as environment
import torch
import parameters as params
import argparse
import time
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import my_dqn
import my_dqn2
import math
import clustering 
import sys
import no_RL_scheme as schemes

params.cloud = 0
params.actorlr *= 1
params.criticlr *= 1
params.dqnlr *= 1
params.scheduler_gamma = 0.995 # more bigger value to maintain initial learning rate setting

replay_buffer_size = 1e6
#replay_buffer = my_sac.ReplayBuffer_SAC(replay_buffer_size)
replay_buffer = my_dqn.replay_buffer(replay_buffer_size)
replay_buffer2 = my_dqn2.replay_buffer(replay_buffer_size)

env = environment.Env()
cluster = clustering.Clustering()

ppo = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim) # continous model (offloading fraction - model1)
dqn = my_dqn.DQN(env, params.wocloud_action_dim2, params.wocloud_state_dim2)
dqn2 = my_dqn2.DQN(env, params.wocloud_action_dim2, params.wocloud_state_dim2)

clst = clustering.Clustering()
nearest = schemes.Nearest()

#print(params.cloud)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()
rewards1     = []
rewards2=[]
losses = []
success_rate = []

rewards1.clear()
rewards2.clear()

losses.clear()
params.wrong_cnt.clear()
params.epsilon_logging.clear()
params.cloud_cnt.clear()
params.valid_cnt.clear()


action1_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def plot():
    clear_output(True)
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(27, 8))

    font_size = params.font_size  # 폰트 크기 설정

    # 첫번째 서브플롯에 rewards
    ax1.plot(rewards1)
    ax1.set_xlabel('Episode', fontsize=font_size)
    ax1.set_ylabel('Reward1', fontsize=font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size - 5)

    ax2.plot(rewards2)
    ax2.set_xlabel('Episode', fontsize=font_size)
    ax2.set_ylabel('Reward2', fontsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size - 5)


    # 세 번째 서브플롯에 action1 distribution 바 그래프로 그리기
    indices = list(range(10))  # 인덱스 0~9
    ax3.bar(indices, action1_distribution[:10])  # 인덱스 0~9까지의 데이터를 바 그래프로
    ax3.set_xlabel('Action [0-9]', fontsize=font_size)
    ax3.set_ylabel('Count', fontsize=font_size)
    ax3.tick_params(axis='both', which='major', labelsize=font_size - 5)

    # 네 번째 서브플롯에 wrong count를 첫 번째 요소를 제외하고 그림
    ax4.plot(params.wrong_cnt[1:])
    ax4.set_xlabel('Episode', fontsize=font_size)
    ax4.set_ylabel('Invalid action count', fontsize=font_size)
    ax4.tick_params(axis='both', which='major', labelsize=font_size - 5)

    # 네 번째 서브플롯에 wrong count를 첫 번째 요소를 제외하고 그림
    ax5.plot(params.cloud_cnt[1:])
    ax5.set_xlabel('Episode', fontsize=font_size)
    ax5.set_ylabel('Cloud selection', fontsize=font_size)
    ax5.tick_params(axis='both', which='major',labelsize=font_size - 5)

     # 네 번째 서브플롯에 wrong count를 첫 번째 요소를 제외하고 그림
    ax6.plot(params.epsilon_logging)
    ax6.set_xlabel('Episode', fontsize=font_size)
    ax6.set_ylabel('Epsilon (DDQN)', fontsize=font_size)
    ax6.tick_params(axis='both', which='major', labelsize=font_size - 5)

     # 네 번째 서브플롯에 wrong count를 첫 번째 요소를 제외하고 그림
    ax7.plot(success_rate)
    #ax7.plot(params.valid_cnt[1:]) #값의 range가 같아서 여기다 그림

    ax7.set_xlabel('Episode', fontsize=font_size)
    ax7.set_ylabel('Success rate', fontsize=font_size)
    ax7.tick_params(axis='both', which='major',labelsize=font_size - 5)
    

    # 두번째 서브플롯에 losses
    ax8.plot(losses)
    ax8.set_xlabel('Episode', fontsize=font_size)
    ax8.set_ylabel('Loss', fontsize=font_size)
    ax8.tick_params(axis='both', which='major',labelsize=font_size - 5)
    plt.show()
    
if __name__ == '__main__':
    params.cloud = 0
    x = -1
    y = 0
    fail = 0
    for i in range(params.numEdge):
        if (i % params.grid_size == 0):
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

    if args.train:
        all_ep_r = []
        buffer = {
            'state': [],
            'action': [],
            'reward': [],
            'done': []
        }
        # training loop
        total_step = 0
        for eps in range(params.EPS):
            fail = 0
            '''
            randomdist = random.randint(0, 10)
            if randomdist < 1:
                params.distribution_mode = 0
            else:
                params.distribution_mode = 1
                '''
            clst.form_cluster()
            clst.visualize_clusters()
            state1, state2_temp = env.reset(-1, 0)
            episode_reward = 0
            eps_r1 = 0
            eps_r2 = 0
            t0 = time.time()

            action1_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # sys.exit()
            for step in range(params.STEP * params.numVeh):
                total_step += 1
                #print("state1:", state1)
            
                action1 = ppo.choose_action(state1)  # ppo로 offloading fraction 만들기     
                action1_distribution[min(int(action1 * 10), 9)] += 1
                state2 = np.concatenate((state2_temp, action1))
                params.state2 = state2
                action2 = dqn2.choose_action(state2, 0)  # 0 means training phase (take epsilon greedy)
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 0)  # 두개의 action 가지고 step

                # Check for NaN values in r, r1, or r2
                if any([np.isnan(val) for val in [r, r1, r2]]):
                    r1 = 0
                    r2 = 0
                    r = 0
                    #print("nan value - did not store in buffer...")
                else:
                    buffer['state'].append(state1)
                    buffer['action'].append(action1)
                    buffer['reward'].append(r1)
                    buffer['done'].append(done)
                    replay_buffer2.add([state2, s2_, [action2], [r2], [done]])
                    dqn2.epsilon_scheduler.step(total_step)

                state1 = s1_
                state2 = s2_
                episode_reward += r
                eps_r1 += r1
                eps_r2 += r2

                if r == 0:
                    fail += 1

                # update PPO
                if (step + 1) % params.ppo_batch == 0:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = ppo.get_v(s1_)[0]
                    discounted_r = []
                    for r, d in zip(buffer['reward'][::-1], buffer['done'][::-1]):
                        v_s_ = r + params.GAMMA * v_s_ * (1 - d)
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    cleaned_discounted_r = []
                    for value in discounted_r:
                        if isinstance(value, np.ndarray):
                            cleaned_discounted_r.append(value.item())  # Convert single-element array to a scalar
                        else:
                            cleaned_discounted_r.append(value)

                    br = np.array(cleaned_discounted_r)[:, np.newaxis]
                    bs = np.vstack(buffer['state'])
                    ba = np.vstack(buffer['action'])

                    buffer['state'], buffer['action'], buffer['reward'], buffer['done'] = [], [], [], []
                    ppo.update(bs, ba, br)

                # update DQN
                if (step + 1) % params.dqn_batch == 0 and total_step > my_dqn.REPLAY_START_SIZE and len(replay_buffer2.buffer) >= params.dqn_batch:
                    sample = replay_buffer2.sample(params.dqn_batch)
                    loss = dqn2.learn(sample)

            if eps % 5 == 0 and eps > 0:  # plot and model saving interval
                plot()
                dqn2.save_model(params.woCloud_dqn_path)
                ppo.save_model(params.woCloud_ppo_path)

            #print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)
            rewards1.append(eps_r1)
            rewards2.append(eps_r2)

            success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
            success_rate.append(success_ratio)
            if total_step > my_dqn2.REPLAY_START_SIZE and len(replay_buffer2.buffer) >= params.dqn_batch:
                losses.append(loss)
                #print(loss, type(loss))

        dqn2.save_model(params.woCloud_dqn_path)
        ppo.save_model(params.woCloud_ppo_path)
