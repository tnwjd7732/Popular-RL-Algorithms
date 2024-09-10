import env as environment
import parameters as params
import time
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import random
import math
import sys
import no_RL_scheme as schemes
import clustering
import my_ppo
import my_dqn

env = environment.Env()
ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

ppo = ppo_.load_model(params.ppo_path)
dqn = dqn_.load_model(params.dqn_path)

ppo_woclst_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
dqn_woclst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

ppo_woclst = ppo_woclst_.load_model(params.woClst_ppo_path)
dqn_woclst = dqn_woclst_.load_model(params.woClst_dqn_path)

params.cloud = 0
ppo_wocloud_= my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
dqn_wocloud_ = my_dqn.DQN(env, params.wocloud_action_dim2, params.wocloud_state_dim2)

ppo_wocloud = ppo_wocloud_.load_model(params.woCloud_ppo_path)
dqn_wocloud = dqn_wocloud_.load_model(params.woCloud_dqn_path)
params.cloud = 1

nearest = schemes.Nearest()
greedy = schemes.Greedy()
clst = clustering.Clustering()

near_succ = []
gree_succ = []
gree2_succ = []
our_succ = []
woclst_succ = []
wocloud_succ = []

def plot():
    clear_output(True)
    fig, ax1 = plt.subplots(1, figsize=(8, 8))

    font_size = params.font_size  # 폰트 크기 설정
    ax1.plot(near_succ, label='Nearest', linewidth=2)
    ax1.plot(gree_succ, label='Greedy(1-hop)', linewidth=2)
    ax1.plot(gree2_succ, label="Greedy(2-hop)", linewidth=2)
    ax1.plot(our_succ, label="Our scheme", linewidth=2)
    ax1.plot(woclst_succ,label="Without clustering", linewidth=2)
    ax1.plot(wocloud_succ, label="Without cloud", linewidth=2)
    ax1.set_xlabel('Episode', fontsize=font_size)
    ax1.legend()
    ax1.set_ylabel('Success rate', fontsize=font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    plt.show()

if __name__ == '__main__':
    x = -1
    y = 0
    fail = 0
    for i in range(params.numEdge):
        if (i % params.grid_size == 0):
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

    '''our scheme'''
    for eps in range(10):
        fail = 0
        clst.form_cluster()
        #clst.visualize_clusters()
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0

        for step in range(params.STEP * params.numVeh):
            action1 = ppo_.choose_action(state1)  # ppo로 offloading fraction 만들기
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_.choose_action(state2, 1)
            s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)  # 두개의 action 가지고 step

            state1 = s1_
            state2 = s2_
            episode_reward += r

            if r == 0:
                fail += 1

        #print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        our_succ.append(success_ratio)

    '''RL wo clst scheme'''
    for eps in range(10):
        fail = 0
        clst.form_cluster_woclst()
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0

        for step in range(params.STEP * params.numVeh):
            action1 = ppo_woclst_.choose_action(state1)  # ppo로 offloading fraction 만들기
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_woclst_.choose_action(state2, 1)
            s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)  # 두개의 action 가지고 step

            state1 = s1_
            state2 = s2_
            episode_reward += r

            if r == 0:
                fail += 1

        #print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        woclst_succ.append(success_ratio)

    '''without cloud scheme'''
    params.cloud = 0
    for eps in range(10):
        fail = 0
        clst.form_cluster()
        #clst.visualize_clusters()
        state1, state2_temp = env.reset(-1, 0)
        episode_reward = 0

        for step in range(params.STEP * params.numVeh):
            action1 = ppo_wocloud_.choose_action(state1)  # ppo로 offloading fraction 만들기
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_wocloud_.choose_action(state2, 1)
            s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 0)  # 두개의 action 가지고 step

            state1 = s1_
            state2 = s2_
            episode_reward += r

            if r == 0:
                fail += 1

       #print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        wocloud_succ.append(success_ratio)

    params.cloud = 1
    '''nearest'''
    for eps in range(10):
        fail = 0
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)

        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * params.numVeh):
            action1, action2 = nearest.choose_action()
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)  # 두개의 action 가지고 step

            episode_reward += r
            if r == 0:
                fail += 1
        #print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)
        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        near_succ.append(success_ratio)

    '''greedy (1-hop)'''
    hop = 1
    for eps in range(10):
        fail = 0
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)

        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * params.numVeh):
            action1, action2 = greedy.choose_action(hop, step)
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)  # 두개의 action 가지고 step

            episode_reward += r
            if r == 0:
                fail += 1

        #print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)
        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        gree_succ.append(success_ratio)

    '''greedy (2-hop)'''
    hop = 2
    for eps in range(10):
        fail = 0
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)

        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * params.numVeh):
            action1, action2 = greedy.choose_action(hop, step)
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)  # 두개의 action 가지고 step

            episode_reward += r
            if r == 0:
                fail += 1

        #print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)
        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        gree2_succ.append(success_ratio)

    plot()
