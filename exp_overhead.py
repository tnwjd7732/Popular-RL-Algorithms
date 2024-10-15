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
import my_dqn2

def run_experiment_overhead(numVeh, repeat):
    params.numVeh = numVeh
    overhead_results = {
        'our_overhead': [],
        'woclst_overhead': [],
        'wocloud_overhead': [],
        'greedy1_overhead': [],
        'greedy2_overhead': [],
        'staticclst_overhead': []
    }
    env = environment.Env()
    ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
    dqn_ = my_dqn2.DQN(env, params.action_dim2, params.state_dim2)

    ppo_woclst_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
    dqn_woclst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

    ppo_staticClst_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
    dqn_staticClst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    
    params.cloud = 0
    ppo_wocloud_= my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
    dqn_wocloud_ = my_dqn2.DQN(env, params.wocloud_action_dim2, params.wocloud_state_dim2)

    ppo_wocloud = ppo_wocloud_.load_model(params.woCloud_ppo_path)
    dqn_wocloud = dqn_wocloud_.load_model(params.woCloud_dqn_path)
    params.cloud = 1
    
    nearest = schemes.Nearest()
    greedy = schemes.Greedy()
    clst = clustering.Clustering()

    for _ in range(repeat):
        
        x = -1

        for i in range(params.numEdge):
            if (i % params.grid_size == 0):
                x += 1
                y = 0
            params.edge_pos[i] = [0.5 + y, 0.5 + x]
            y += 1

        '''our scheme'''
        total_hop_counts = 0
        total_tasks = 0
        clst.form_cluster()
        state1, state2_temp = env.reset(-1, 1)
        params.hop_counts = []

        for step in range(params.STEP * numVeh):
            if params.remains[params.nearest] > params.resource_avg/2:
                action1, action2 = nearest.choose_action()
                s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)  # 두개의 action 가지고 step
            else:
                action1 = ppo_.choose_action(state1)  # ppo로 offloading fraction 만들기
                state2 = np.concatenate((state2_temp, action1))
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)
                if len(env.cluster) < action2:
                    action2 = 0
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)  # 두개의 action 가지고 step

            state1 = s1_
            state2 = s2_

        overhead_results['our_overhead'].append(sum(params.hop_counts))


        '''RL wo clst scheme'''
        total_hop_counts = 0
        total_tasks = 0
        clst.form_cluster_woclst()
        state1, state2_temp = env.reset(-1, 1)
        params.hop_counts = []
        print("wo clst", params.hop_counts)

        for step in range(params.STEP * numVeh):
            action1 = ppo_woclst_.choose_action(state1)
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_woclst_.choose_action(state2, 1)
            s1_, s2_, _, _, _, hopcount = env.step(action1, action2, step, 1)
            state1 = s1_
            state2 = s2_

        print("woclst", params.hop_counts)
        overhead_results['woclst_overhead'].append(sum(params.hop_counts))

        '''without cloud scheme'''
        total_hop_counts = 0
        total_tasks = 0
        params.cloud = 0
        clst.form_cluster()
        state1, state2_temp = env.reset(-1, 0)
        params.hop_counts = []

        for step in range(params.STEP * numVeh):
            action1 = ppo_wocloud_.choose_action(state1)
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_wocloud_.choose_action(state2, 1)
            s1_, s2_, _, _, _, hopcount = env.step(action1, action2, step, 0)
            state1 = s1_
            state2 = s2_


        overhead_results['wocloud_overhead'].append(sum(params.hop_counts))
        params.cloud = 1

        '''greedy (1-hop)'''
        total_hop_counts = 0
        total_tasks = 0
        state1, state2_temp = env.reset(-1, 1)
        params.hop_counts = []

        for step in range(params.STEP * numVeh):
            action1, action2 = greedy.choose_action(1, step)
            s1_, s2_, _, _, _, hopcount = env.step2(action1, action2, step)
            state1 = s1_


        overhead_results['greedy1_overhead'].append(sum(params.hop_counts))

        '''greedy (2-hop)'''
        total_hop_counts = 0
        total_tasks = 0
        state1, state2_temp = env.reset(-1, 1)
        params.hop_counts = []

        for step in range(params.STEP * numVeh):
            action1, action2 = greedy.choose_action(2, step)
            s1_, s2_, _, _, _, hopcount = env.step2(action1, action2, step)
            state1 = s1_

        overhead_results['greedy2_overhead'].append(sum(params.hop_counts))

        '''RL with static clustering'''
        total_hop_counts = 0
        total_tasks = 0
        clst.form_static_cluster()
        state1, state2_temp = env.reset(-1, 1)
        params.hop_counts = []
        for step in range(params.STEP * numVeh):
            action1 = ppo_staticClst_.choose_action(state1)
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_staticClst_.choose_action(state2, 1)
            s1_, s2_, _, _, _, hopcount = env.step(action1, action2, step, 1)
            state1 = s1_
            state2 = s2_

       
        overhead_results['staticclst_overhead'].append(sum(params.hop_counts))

    avg_results = {key: np.mean(values) for key, values in overhead_results.items()}
    return avg_results

def plot_overhead(results, veh_range):
    clear_output(True)
    plt.rcParams['font.family']= 'Times New Roman'
    plt.rcParams.update({'font.size': params.font_size-5})

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    font_size = params.font_size
    ax.plot(veh_range, results['our_overhead'], label='Our scheme', linewidth=2)
    ax.plot(veh_range, results['woclst_overhead'], label='Without clustering', linewidth=2)
    ax.plot(veh_range, results['wocloud_overhead'], label='Without cloud', linewidth=2)
    ax.plot(veh_range, results['greedy1_overhead'], label="Greedy (1-hop)", linewidth=2)
    ax.plot(veh_range, results['greedy2_overhead'], label="Greedy (2-hop)", linewidth=2)
    ax.plot(veh_range, results['staticclst_overhead'], label="Static clustering", linewidth=2)

    ax.set_xlabel('Number of Vehicles', fontsize=font_size)
    ax.legend()
    ax.set_ylabel('Average Hop Count (Communication Overhead)', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    plt.show()
if __name__ == '__main__':
    veh_range = range(50, 501, 100)  # 차량 수 범위
    repeat = params.repeat  # 반복 횟수 (실험 반복 수)
    final_overhead_results = {
        'our_overhead': [],
        'woclst_overhead': [],
        'wocloud_overhead': [],
        'greedy1_overhead': [],
        'greedy2_overhead': [],
        'staticclst_overhead': []
    }

    for numVeh in veh_range:
        avg_overhead_results = run_experiment_overhead(numVeh, repeat)
        for key in final_overhead_results.keys():
            final_overhead_results[key].append(avg_overhead_results[key])
    
    plot_overhead(final_overhead_results, veh_range)
