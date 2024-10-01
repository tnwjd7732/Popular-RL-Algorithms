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

def run_experiment(task_cpu, repeat):
    params.min_cpu = task_cpu
    params.max_cpu = task_cpu
    results = {
       'our_succ': [],
        'our_reward': [],
        'woclst_succ': [],
        'woclst_reward': [],
        'wocloud_succ': [],
        'wocloud_reward': [],
        'near_succ': [],
        'near_reward': [],
        'gree_succ': [],
        'gree_reward': [],
        'gree2_succ': [],
        'gree2_reward': [],
        'staticclst_succ': [],
        'staticclst_reward': []
    }

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

  
    ppo_staticClst_= my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
    dqn_staticClst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

    ppo_staticClst = ppo_staticClst_.load_model(params.staticClst_ppo_path)
    dqn_staticClst = dqn_staticClst_.load_model(params.staticClst_dqn_path)



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

        params.task_cpu = task_cpu  # Update the task CPU requirement

        '''our scheme'''
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

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        results['our_succ'].append(success_ratio)
        results['our_reward'].append(episode_reward)

        '''RL wo clst scheme'''
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

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        results['woclst_succ'].append(success_ratio)
        results['woclst_reward'].append(episode_reward)

        '''without cloud scheme'''
        params.cloud = 0
        fail = 0
        clst.form_cluster()
        # #clst.visualize_clusters()
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

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        results['wocloud_succ'].append(success_ratio)
        results['wocloud_reward'].append(episode_reward)
        params.cloud = 1

        '''RL with static clst scheme'''
        fail = 0
        clst.form_static_cluster()
        state1, state2_temp = env.reset(-1, 1)
        episode_reward = 0

        for step in range(params.STEP * params.numVeh):
            numVeh = params.numVeh

            action1 = ppo_staticClst_.choose_action(state1)  # ppo로 offloading fraction 만들기
            state2 = np.concatenate((state2_temp, action1))
            params.state2 = state2
            action2 = dqn_staticClst_.choose_action(state2, 1)
            s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)  # 두개의 action 가지고 step

            state1 = s1_
            state2 = s2_
            episode_reward += r

            if r == 0:
                fail += 1

        success_ratio = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
        results['staticclst_succ'].append(success_ratio)
        results['staticclst_reward'].append(episode_reward)


        '''nearest'''
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

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        results['near_succ'].append(success_ratio)
        results['near_reward'].append(episode_reward)

        '''greedy (1-hop)'''
        hop = 1
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

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        results['gree_succ'].append(success_ratio)
        results['gree_reward'].append(episode_reward)

        '''greedy (2-hop)'''
        hop = 2
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

        success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
        results['gree2_succ'].append(success_ratio)
        results['gree2_reward'].append(episode_reward)
    
    avg_results = {key: np.mean(values) for key, values in results.items()}
    return avg_results

def plot(results, task_cpu_range):
    clear_output(True)
    plt.rcParams.update({'font.size': params.font_size-5})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    font_size = params.font_size  # 폰트 크기 설정
    ax1.plot(task_cpu_range, results['near_succ'], label='Nearest', linewidth=2)
    ax1.plot(task_cpu_range, results['gree_succ'], label='Greedy(1-hop)', linewidth=2)
    ax1.plot(task_cpu_range, results['gree2_succ'], label="Greedy(2-hop)", linewidth=2)
    ax1.plot(task_cpu_range, results['our_succ'], label="Our scheme", linewidth=2)
    ax1.plot(task_cpu_range, results['woclst_succ'], label="Without clustering", linewidth=2)
    ax1.plot(task_cpu_range, results['wocloud_succ'], label="Without cloud", linewidth=2)
    ax1.plot(task_cpu_range, results['staticclst_succ'], label="Static clustering", linewidth=2)

    ax1.set_xlabel('Task CPU', fontsize=font_size)
    ax1.legend()
    ax1.set_ylabel('Success rate', fontsize=font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)

    ax2.plot(task_cpu_range, results['near_reward'], label='Nearest', linewidth=2)
    ax2.plot(task_cpu_range, results['gree_reward'], label='Greedy(1-hop)', linewidth=2)
    ax2.plot(task_cpu_range, results['gree2_reward'], label="Greedy(2-hop)", linewidth=2)
    ax2.plot(task_cpu_range, results['our_reward'], label="Our scheme", linewidth=2)
    ax2.plot(task_cpu_range, results['woclst_reward'], label="Without clustering", linewidth=2)
    ax2.plot(task_cpu_range, results['wocloud_reward'], label="Without cloud", linewidth=2)
    ax2.plot(task_cpu_range, results['staticclst_reward'], label="Static clustering", linewidth=2)

    ax2.set_xlabel('Task CPU', fontsize=font_size)
    ax2.legend()
    ax2.set_ylabel('Average Reward', fontsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    plt.show()

if __name__ == '__main__':
    task_cpu_range = np.arange(0.5, 4, 0.5)
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
        'gree_succ': [],
        'gree_reward': [],
        'gree2_succ': [],
        'gree2_reward': [],
        'staticclst_succ': [],
        'staticclst_reward': []
    }

    for task_cpu in task_cpu_range:
        avg_results = run_experiment(task_cpu, repeat)
        for key in final_results.keys():
            final_results[key].append(avg_results[key])
    
    plot(final_results, task_cpu_range)
