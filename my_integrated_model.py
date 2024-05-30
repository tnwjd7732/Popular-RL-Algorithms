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

replay_buffer_size = 1e6
#replay_buffer = my_sac.ReplayBuffer_SAC(replay_buffer_size)
replay_buffer = my_dqn.replay_buffer(replay_buffer_size)

env = environment.Env()

ppo = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim) # continous model (offloading fraction - model1)
sac_trainer=my_sac.SAC_Trainer(replay_buffer, hidden_dim=params.hidden_dim, state_dim=params.state_dim2, action_dim=params.action_dim2) # discrete model (offloading action - model2)
dqn = my_dqn.DQN(env)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()
rewards     = []
losses = []

def plot(rewards, losses):
    clear_output(True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # 첫 번째 서브플롯에 rewards를 그림
    ax1.plot(rewards)
    ax1.set_title('Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # 두 번째 서브플롯에 losses를 그림
    ax2.plot(losses)
    ax2.set_title('Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')

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
        total_step = 0
        for eps in range(params.EPS):
            state1 =  env.reset(-1)
            episode_reward = 0
            t0 = time.time()
            #print("\n\n\n new episode")
            #loss = 0
            for step in range(params.STEP*params.numVeh):
                #print("remains: ", params.remains)
                total_step+=1
                #print("EPS: ", eps, "STEP: ", step)
                #state1: task info만 담겨있음
                #ppo에서 server info 바탕으로 attention distribution 만든거랑 task info 합쳐서 encoded_state로 리턴 (이게 곧 real state)
                action1 = ppo.choose_action(state1) # ppo로 offloading fraction 만들기                
                state2 = np.concatenate((params.remains_lev, params.hop_count, params.task, action1))
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

            if eps % 2 == 0 and eps>0: # plot and model saving interval
                plot(rewards, losses)
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

