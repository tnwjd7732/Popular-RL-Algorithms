import env as environment
import parameters as params
params.distribution_mode = 1
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

def run_experiment(numVeh, repeat):
    print("Experiment settings")
    print("numveh: ", numVeh)
    print(" max size: ", params.max_size, " min size: ", params.min_size)
    print(" max cpu: ", params.max_cpu, " min cpu: ", params.min_cpu)

    params.distribution_mode = 1
    params.numVeh = numVeh
    results = {
        'our_succ': [],
        'our_reward': [],
        'woclst_succ': [],
        'woclst_reward': [],
        'wocloud_succ': [],
        'wocloud_reward': [],
        'near_succ': [],
        'near_reward': [],
        'cloud_near_succ': [],
        'cloud_near_reward': [],
        'gree_succ': [],
        'gree_reward': [],
        'cloud_gree_succ': [],
        'cloud_gree_reward': [],
        'cloud_gree2_succ': [],
        'cloud_gree2_reward': [],
        'gree2_succ': [],
        'gree2_reward': [],
        'staticclst_succ': [],
        'staticclst_reward': []
    }
    
    env = environment.Env()
    ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continuous model
    dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    ppo = ppo_.load_model(params.ppo_path)
    dqn = dqn_.load_model(params.dqn_path)

    ppo_woclst_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
    dqn_woclst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    ppo_woclst = ppo_woclst_.load_model(params.woClst_ppo_path)
    dqn_woclst = dqn_woclst_.load_model(params.woClst_dqn_path)

    params.cloud = 0
    ppo_wocloud_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
    dqn_wocloud_ = my_dqn.DQN(env, params.wocloud_action_dim2, params.wocloud_state_dim2)
    ppo_wocloud = ppo_wocloud_.load_model(params.woCloud_ppo_path)
    dqn_wocloud = dqn_wocloud_.load_model(params.woCloud_dqn_path)
    params.cloud = 1

    ppo_staticClst_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
    dqn_staticClst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    ppo_staticClst = ppo_staticClst_.load_model(params.staticClst_ppo_path)
    dqn_staticClst = dqn_staticClst_.load_model(params.staticClst_dqn_path)

    nearest = schemes.Nearest()
    greedy = schemes.Greedy()
    clst = clustering.Clustering()
    nearest_cloud = schemes.NearestCloud()
    greedy_cloud = schemes.GreedyCloud()

    for _ in range(repeat):
        x = -1
        for i in range(params.numEdge):
            if (i % params.grid_size == 0):
                x += 1
                y = 0
            params.edge_pos[i] = [0.5 + y, 0.5 + x]
            y += 1

        '''our scheme'''
        fail = 0
        clst.form_cluster()
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0

        for step in range(params.STEP * numVeh):
            action1 = ppo_.choose_action(state1)
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_.choose_action(state2, 1)
            if len(env.cluster) < action2:
                action2 = 0
            s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)
            state1 = s1_
            state2 = s2_
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['our_succ'].append(success_ratio)
        results['our_reward'].append(episode_reward)

        '''RL with static clst scheme'''
        fail = 0
        clst.form_static_cluster()
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1 = ppo_staticClst_.choose_action(state1)
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_staticClst_.choose_action(state2, 1)
            s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)
            state1 = s1_
            state2 = s2_
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['staticclst_succ'].append(success_ratio)
        results['staticclst_reward'].append(episode_reward)

        '''NearestCloud scheme'''
        fail = 0
        clst.form_cluster()
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1, action2 = nearest_cloud.choose_action(step)
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)
            state1 = s1_
            state2 = s2_
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['cloud_near_succ'].append(success_ratio)
        results['cloud_near_reward'].append(episode_reward)

        '''GreedyCloud 1-hop scheme'''
        fail = 0
        hop = 1
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1, action2 = greedy_cloud.choose_action(hop, step)
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)
            state1 = s1_
            state2 = s2_
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['cloud_gree_succ'].append(success_ratio)
        results['cloud_gree_reward'].append(episode_reward)

        '''GreedyCloud 2-hop scheme'''
        fail = 0
        hop = 2
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1, action2 = greedy_cloud.choose_action(hop, step)
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)
            state1 = s1_
            state2 = s2_
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['cloud_gree2_succ'].append(success_ratio)
        results['cloud_gree2_reward'].append(episode_reward)

        '''RL without clustering scheme'''
        fail = 0
        clst.form_cluster_woclst()
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1 = ppo_woclst_.choose_action(state1)
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_woclst_.choose_action(state2, 1)
            s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)
            state1 = s1_
            state2 = s2_
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['woclst_succ'].append(success_ratio)
        results['woclst_reward'].append(episode_reward)

        '''without cloud scheme'''
        params.cloud = 0
        fail = 0
        clst.form_cluster()
        state1, state2_temp = env.reset(-1, 0)
        episode_reward = 0
        for step in range(params.STEP * params.numVeh):
            if params.remains[params.nearest] > params.resource_avg/2:
                action1, action2 = nearest.choose_action()
                s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)
            else:   
                action1 = ppo_wocloud_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1))
                params.state2 = state2
                action2 = dqn_wocloud_.choose_action(state2, 1)
                if len(env.cluster) < action2:
                    action2 = 0
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 0)
            state1 = s1_
            state2 = s2_
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['wocloud_succ'].append(success_ratio)
        results['wocloud_reward'].append(episode_reward)
        params.cloud = 1

        '''nearest'''
        fail = 0
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1, action2 = nearest.choose_action()
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['near_succ'].append(success_ratio)
        results['near_reward'].append(episode_reward)

        '''greedy (1-hop)'''
        hop = 1
        fail = 0
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1, action2 = greedy.choose_action(hop, step)
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['gree_succ'].append(success_ratio)
        results['gree_reward'].append(episode_reward)

        '''greedy (2-hop)'''
        hop = 2
        fail = 0
        params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0
        for step in range(params.STEP * numVeh):
            action1, action2 = greedy.choose_action(hop, step)
            s1_, s2_, r, r1, r2, done = env.step2(action1, action2, step)
            episode_reward += r
            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['gree2_succ'].append(success_ratio)
        results['gree2_reward'].append(episode_reward)
    
    avg_results = {key: np.mean(values) for key, values in results.items()}
    return avg_results

def plot(results, veh_range):
    clear_output(True)
    plt.rcParams.update({'font.size': params.font_size - 5})
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

    font_size = params.font_size
    # Success Rate Plot
    ax1.plot(veh_range, results['near_succ'], label="Nearest", color='cyan', linestyle='--', linewidth=2, marker='o')
    ax1.plot(veh_range, results['gree_succ'], label="Greedy (1-hop)", color='orange', linestyle='--', linewidth=2, marker='s')
    ax1.plot(veh_range, results['gree2_succ'], label="Greedy (2-hop)", color='magenta', linestyle='--', linewidth=2, marker='^')
    ax1.plot(veh_range, results['our_succ'], label="Proposed", color='purple', linestyle='-', linewidth=2, marker='D')
    ax1.plot(veh_range, results['woclst_succ'], label="Without clustering", color='purple', linestyle='--', linewidth=2, marker='x')
    ax1.plot(veh_range, results['staticclst_succ'], label="Static clustering", color='purple', linestyle='-.', linewidth=2, marker='s')
    ax1.plot(veh_range, results['cloud_near_succ'], label="Nearest Cloud", color='blue', linestyle='-', linewidth=2, marker='o')
    ax1.plot(veh_range, results['cloud_gree_succ'], label="Greedy Cloud", color='green', linestyle='-', linewidth=2, marker='s')
    ax1.plot(veh_range, results['cloud_gree2_succ'], label="Greedy Cloud", color='red', linestyle='-', linewidth=2, marker='^')

    ax1.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax1.set_ylabel('Success Rate', fontsize=font_size + 10)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size-3, frameon=False)

    # Reward Plot
    ax2.plot(veh_range, results['near_reward'], label="Nearest", color='cyan', linestyle='--', linewidth=2, marker='o')
    ax2.plot(veh_range, results['gree_reward'], label="Greedy (1-hop)", color='orange', linestyle='--', linewidth=2, marker='s')
    ax2.plot(veh_range, results['gree2_reward'], label="Greedy (2-hop)", color='magenta', linestyle='--', linewidth=2, marker='^')
    ax2.plot(veh_range, results['our_reward'], label="Proposed", color='purple', linestyle='-', linewidth=2, marker='D')
    ax2.plot(veh_range, results['woclst_reward'], label="Without clustering", color='purple', linestyle='--', linewidth=2, marker='x')
    ax2.plot(veh_range, results['staticclst_reward'], label="Static clustering", color='purple', linestyle='-.', linewidth=2, marker='s')
    ax2.plot(veh_range, results['cloud_near_reward'], label="Nearest Cloud", color='blue', linestyle='-', linewidth=2, marker='o')
    ax2.plot(veh_range, results['cloud_gree_reward'], label="Greedy Cloud", color='green', linestyle='-', linewidth=2, marker='s')
    ax2.plot(veh_range, results['cloud_gree2_reward'], label="Greedy Cloud", color='red', linestyle='-', linewidth=2, marker='^')

    ax2.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax2.set_ylabel('Average Reward', fontsize=font_size + 10)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size-3, frameon=False)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    veh_range = range(250, 401, 50)
    repeat = params.repeat
    final_results = {
        'our_succ': [],
        'our_reward': [],
        'woclst_succ': [],
        'woclst_reward': [],
        'wocloud_succ': [],
        'wocloud_reward': [],
        'near_succ': [],
        'near_reward': [],
        'cloud_near_succ': [],
        'cloud_near_reward': [],
        'gree_succ': [],
        'gree_reward': [],
        'cloud_gree_succ': [],
        'cloud_gree_reward': [],
        'cloud_gree2_succ': [],
        'cloud_gree2_reward': [],        
        'gree2_succ': [],
        'gree2_reward': [],
        'staticclst_succ': [],
        'staticclst_reward': []
    }

    for numVeh in veh_range:
        avg_results = run_experiment(numVeh, repeat)
        for key in final_results.keys():
            final_results[key].append(avg_results[key])

    plot(final_results, veh_range)
