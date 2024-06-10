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
import math
import clustering 
import sys

replay_buffer_size = 1e6
#replay_buffer = my_sac.ReplayBuffer_SAC(replay_buffer_size)
replay_buffer = my_dqn.replay_buffer(replay_buffer_size)

env = environment.Env()
cluster = clustering.Clustering()

ppo = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim) # continous model (offloading fraction - model1)
dqn = my_dqn.DQN(env)
clst = clustering.Clustering()


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()
rewards     = []
losses = []
action1_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def plot():
    clear_output(True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # 첫번째 서브플롯에 rewards
    ax1.plot(rewards)
    ax1.set_title('Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # 두번째 서브플롯에 losses
    ax2.plot(losses)
    ax2.set_title('Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')

    # 세 번째 서브플롯에 action1 distribution 바 그래프로 그리기
    indices = list(range(10))  # 인덱스 0~9
    ax3.bar(indices, action1_distribution[:10])  # 인덱스 0~9까지의 데이터를 바 그래프로
    ax3.set_title('Action Distribution')
    ax3.set_xlabel('Action [0-9]')
    ax3.set_ylabel('Count')

    plt.show()

if __name__ == '__main__':
    x=-1
    y=0
    for i in range(params.numEdge):
        if (i%params.grid_size == 0):
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y+=1

    if args.train:
        all_ep_r = []
        buffer={
            'state':[],
            'action':[],
            'reward':[],
            'done':[]
        }
        # training loop
        total_step = 0
        for eps in range(params.EPS):
            clst.form_cluster()
            clst.visualize_clusters()
            state1, state2_temp =  env.reset(-1)
            episode_reward = 0
            t0 = time.time()
            
            action1_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #sys.exit()
            for step in range(params.STEP*params.numVeh):
                #print("remains: ", params.remains)
                total_step+=1
                #print("EPS: ", eps, "STEP: ", step)
                #state1: task info만 담겨있음
                #ppo에서 server info 바탕으로 attention distribution 만든거랑 task info 합쳐서 encoded_state로 리턴 (이게 곧 real state)
                action1 = ppo.choose_action(state1) # ppo로 offloading fraction 만들기                
                action1_distribution[int(action1 * 10)] +=1

                state2 = np.concatenate((state2_temp, action1))
                #state2 = np.concatenate((state1.reshape(-1), action1.reshape(-1)), axis=-1)
                params.state2 = state2
                
                # epsilon greedy
                random_number = random.uniform(0, 1)
                #action2 = sac_trainer.policy_net.get_action(state2, deterministic = True) # state2로 sac output (offloading decision) 만들기
                action2 = dqn.choose_action(state2)
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step) # 두개의 action 가지고 step
                            
                '''방금의 경험을 각각의 버퍼에 기록하는 과정'''
                buffer['state'].append(state1)
                buffer['action'].append(action1)
                buffer['reward'].append(r1) 
                buffer['done'].append(done) 
                replay_buffer.add([state2,s2_,[action2],[r2],[done]])
                dqn.epsilon_scheduler.step(total_step)

                '''상태 전이, 보상 누적'''
                state1 = s1_
                state2 = s2_
                episode_reward += r         
                '''
                # update SAC
                if len(replay_buffer) > params.sac_batch and step % params.sac_interval ==0:
                    #print("update SAC")
                    for i in range(params.update_itr):
                        _=sac_trainer.update(params.sac_batch, reward_scale=1., auto_entropy=False, target_entropy=-2)
                
                if done:
                    break
                '''
                # update PPO
                if (step + 1) % params.ppo_batch== 0:
                    #print("update PPO")
                    if done:
                        v_s_=0
                    else:
                        v_s_ = ppo.get_v(s1_)[0]
                    discounted_r = []
                    for r, d in zip(buffer['reward'][::-1], buffer['done'][::-1]):
                        v_s_ = r + params.GAMMA * v_s_ * (1-d)
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    cleaned_discounted_r = []
                    for value in discounted_r:
                        if isinstance(value, np.ndarray):
                            cleaned_discounted_r.append(value.item())  # Convert single-element array to a scalar
                        else:
                            cleaned_discounted_r.append(value)

                    # Convert the cleaned list to a NumPy array and add a new axis
                    br = np.array(cleaned_discounted_r)[:, np.newaxis]

    

                    bs = np.vstack(buffer['state'])
                    ba = np.vstack(buffer['action'])
                    #discounted_r = [float(value) for value in discounted_r]  # Convert to a list of floats if necessary
                    #br = np.array(discounted_r)[:, np.newaxis]

                   
                    buffer['state'], buffer['action'], buffer['reward'], buffer['done'] = [], [], [], []
                    ppo.update(bs, ba, br)

                if total_step > my_dqn.REPLAY_START_SIZE and len(replay_buffer.buffer) >= params.dqn_batch :
                    sample = replay_buffer.sample(params.dqn_batch)
                    loss = dqn.learn(sample)
                    #epi_loss += loss
            
            if eps % 5 == 0 and eps>0: # plot and model saving interval
                plot()
                np.save('rewards', rewards)
                #sac_trainer.save_model(params.sac_path)
                dqn.save_model()
                ppo.save_model(params.ppo_path)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)
            rewards.append(episode_reward)
            if total_step > my_dqn.REPLAY_START_SIZE and len(replay_buffer.buffer) >= params.dqn_batch :
                losses.append(loss)
                print(loss, type(loss))
        #sac_trainer.save_model(params.sac_path)
        dqn.save_model()
        ppo.save_model(params.ppo_path)

