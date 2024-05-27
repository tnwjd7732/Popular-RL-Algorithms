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

replay_buffer_size = 1e6
replay_buffer = my_sac.ReplayBuffer_SAC(replay_buffer_size)

ppo = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim) # continous model (offloading fraction - model1)
sac_trainer=my_sac.SAC_Trainer(replay_buffer, hidden_dim=params.hidden_dim, state_dim=params.state_dim2, action_dim=params.action_dim2) # discrete model (offloading action - model2)

env = environment.Env()

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()
rewards     = []

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.show()

if __name__ == '__main__':
    if args.train:
        all_ep_r = []
        buffer={
            'state':[],
            'action':[],
            'reward':[],
            'done':[]
        }
        # training loop
        epsilon = 1
        for eps in range(params.EPS):
            state1 =  env.reset(-1)
            episode_reward = 0
            t0 = time.time()
            #print("\n\n\n new episode")
            
            for step in range(params.STEP*params.numVeh):
                #print("remains: ", params.remains)

                #print("EPS: ", eps, "STEP: ", step)
                #state1: task info만 담겨있음
                #ppo에서 server info 바탕으로 attention distribution 만든거랑 task info 합쳐서 encoded_state로 리턴 (이게 곧 real state)
                encoded_state1, action1 = ppo.choose_action(state1) # ppo로 offloading fraction 만들기
                state2 = np.concatenate((encoded_state1.reshape(-1), action1.reshape(-1)), axis=-1)
                params.state2 = state2
                
                # epsilon greedy
                random_number = random.uniform(0, 1)
                #print("epsilon: ", epsilon, "random: ", random_number)
                if epsilon > random_number:
                    #print("Random action")
                    action2 = random.randint(0, params.numEdge-1)
                else:
                    #print("Greedy action")
                    action2 = sac_trainer.policy_net.get_action(state2, deterministic = True) # state2로 sac output (offloading decision) 만들기
                epsilon = epsilon * 0.9999
                s1_, s2_, r, done = env.step(action1, action2, step) # 두개의 action 가지고 step
                
                #print("action1: ", action1)
                #print("action2: ", action2)
                
                '''방금의 경험을 각각의 버퍼에 기록하는 과정'''
                buffer['state'].append(encoded_state1)
                buffer['action'].append(action1)
                buffer['reward'].append(r) 
                buffer['done'].append(done) 
                replay_buffer.push(state2, action2, r, s2_, done)

                '''상태 전이, 보상 누적'''
                s = s2_
                episode_reward += r         
                
                # update SAC
                if len(replay_buffer) > params.sac_batch and step % params.sac_interval ==0:
                    #print("update SAC")
                    for i in range(params.update_itr):
                        _=sac_trainer.update(params.sac_batch, reward_scale=1., auto_entropy=True, target_entropy=-2)

                if done:
                    break

                # update PPO
                if (step + 1) % params.ppo_batch== 0:
                    #print("update PPO")
                    if done:
                        v_s_=0
                    else:
                        v_s_ = ppo.get_v(s1_.numpy())[0]
                    discounted_r = []
                    for r, d in zip(buffer['reward'][::-1], buffer['done'][::-1]):
                        v_s_ = r + params.GAMMA * v_s_ * (1-d)
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer['state']), np.vstack(buffer['action']), np.array(discounted_r)[:, np.newaxis]
                    buffer['state'], buffer['action'], buffer['reward'], buffer['done'] = [], [], [], []
                    ppo.update(bs, ba, br)

                if done:
                    break

            if eps % 5 == 0 and eps>0: # plot and model saving interval
                plot(rewards)
                np.save('rewards', rewards)
                sac_trainer.save_model(params.sac_path)
                ppo.save_model(params.ppo_path)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)
            rewards.append(episode_reward)
        sac_trainer.save_model(params.sac_path)
        ppo.save_model(params.ppo_path)
