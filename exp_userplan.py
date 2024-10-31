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

env = environment.Env()

ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)  # continous model (offloading fraction - model1)
dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

ppo = ppo_.load_model(params.ppo_path)
dqn = dqn_.load_model(params.dqn_path)
clst = clustering.Clustering()

# run_experiment_performance_cost 함수
def run_experiment_performance_cost(numVeh, repeat, userplan):
    x=-1
    for i in range(params.numEdge):
        if (i % params.grid_size == 0):
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

    params.numVeh = numVeh
    results = {
    
        'performance_score_fixed': [],  # 고정 스킴 성능 점수 저장
        'cost_score_fixed': [],  # 고정 스킴 비용 점수 저장
        'performance_score_premium': [],
        'cost_score_premium': [],
        'performance_score_basic': [],
        'cost_score_basic': []
    }

    # W_size, W_CPU 값 설정 (ppt 식에 따라)
    # 고정 스킴일 경우 처리
    if userplan != 1:  # Fixed allocation 스킴
        params.userplan = userplan
        performance_scores_fixed = []
        cost_scores_fixed = []

        for _ in range(repeat):
            clst.form_cluster()
            state1, state2_temp = env.reset(-1, 1)
            episode_reward = 0
            cost = 0

            for step in range(params.STEP * numVeh):
                action1 = ppo_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1))
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)

                # 성능 점수와 비용 계산
                t_tasktime = params.task[2]
                t_total = params.realtime
                if t_tasktime > t_total:
                    performance_score = (t_tasktime - t_total) / t_tasktime if t_tasktime != 0 else 0
                    print(performance_score)
                else:
                    performance_score = 0
                
                
                # Cost Score 계산 (W_size + W_CPU로 나누기)
                c_max = (params.max_size * params.unitprice_size * 1.65 + params.max_cpu * params.unitprice_cpu*1.65)
                c_min = (params.min_size * params.unitprice_size*0.85 + params.min_cpu * params.unitprice_cpu*0.85)
                c_paid = params.costpaid
                task_total = params.task[0] + params.task[1] # W_size + W_CPU로 나누는 부분 추가
                if t_tasktime > t_total:
                    cost_score = ((c_max - c_paid) / (c_max - c_min)) / task_total if (c_max - c_min) != 0 else 0
                else:
                    cost_score=0

                episode_reward += performance_score
                cost += cost_score
                state1 = s1_
                state2 = s2_

            performance_scores_fixed.append(episode_reward/(params.STEP*numVeh))
            cost_scores_fixed.append(cost/(params.STEP*numVeh))

        # 평균 성능 및 비용 점수 저장
        results['performance_score_fixed'] = np.mean(performance_scores_fixed)
        results['cost_score_fixed'] = np.mean(cost_scores_fixed)

    else:
        params.userplan = userplan
        # 우리 스킴 처리 (Basic, Premium 구분)
        performance_scores_premium = []
        performance_scores_basic = []
        cost_scores_premium = []
        cost_scores_basic = []

        for _ in range(repeat):
            clst.form_cluster()
            state1, state2_temp = env.reset(-1, 1)
            episode_reward_premium = 0
            episode_reward_basic = 0
            cost_premium = 0
            cost_basic = 0
            pre = 0
            basic = 0

            for step in range(params.STEP * numVeh):
                veh_id = step % numVeh
                action1 = ppo_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1))
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)

                t_tasktime = params.task[2]
                t_total = params.realtime
                if t_tasktime > t_total:
                    performance_score = (t_tasktime - t_total) / t_tasktime if t_tasktime != 0 else 0
                else:
                    performance_score = 0
                
                # Cost Score 계산 (W_size + W_CPU로 나누기)
                c_max = (params.max_size * params.unitprice_size * 1.65 + params.max_cpu * params.unitprice_cpu*1.65)
                c_min = (params.min_size * params.unitprice_size*0.85 + params.min_cpu * params.unitprice_cpu*0.85)
                c_paid = params.costpaid
                task_total = params.task[0] + params.task[1] # W_size + W_CPU로 나누는 부분 추가
                if t_tasktime > t_total:
                    cost_score = ((c_max - c_paid) / (c_max - c_min)) / task_total if (c_max - c_min) != 0 else 0
                else:
                    cost_score = 0

                if env.plan_info[veh_id] == 1:  # Premium user
                    episode_reward_premium += performance_score
                    cost_premium += cost_score
                    pre+=1
                else:  # Basic user
                    episode_reward_basic += performance_score
                    cost_basic += cost_score
                    basic+=1

                state1 = s1_
                state2 = s2_

            performance_scores_premium.append(episode_reward_premium/(pre))
            performance_scores_basic.append(episode_reward_basic/(basic))
            cost_scores_premium.append(cost_premium/(pre))
            cost_scores_basic.append(cost_basic/(basic))

        # 평균 성능 및 비용 점수 저장
        results['performance_score_premium'] = np.mean(performance_scores_premium)
        results['cost_score_premium'] = np.mean(cost_scores_premium)
        results['performance_score_basic'] = np.mean(performance_scores_basic)
        results['cost_score_basic'] = np.mean(cost_scores_basic)

    return results


# 플롯 함수 수정 (가로로 나란히 배치, 범례를 그래프 내부로 이동)
# 플롯 함수 수정 (범례를 그래프 밖으로 이동, 겹치지 않게 조정)
def plot_performance_cost_comparison(results, veh_range, fixed_allocations):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': params.font_size - 5})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # 가로로 나란히 배치

    font_size = params.font_size

    # Performance score plot
    ax1.plot(veh_range, results['performance_score_basic'], label="Proposed (Basic)", 
             color='purple', linestyle='--', marker='o', linewidth=2)  # 우리 스킴 Basic: 점선 + 동그라미
    ax1.plot(veh_range, results['performance_score_premium'], label="Proposed (Premium)", 
             color='purple', linestyle='-', marker='o', linewidth=2)  # 우리 스킴 Premium: 실선 + 동그라미
    
    # 각 fixed allocation에 대해 별도로 라인을 그린다 (고정 스킴: 실선 + 각기 다른 색)
    fixed_colors = ['green', 'blue', 'orange', 'red', 'cyan', 'magenta']
    markers = ['s', 'v', '^', '<', '>', 'D']  # 각기 다른 심볼
    for i, fixed_allocation in enumerate(fixed_allocations):
        ax1.plot(veh_range, results[f'performance_score_fixed_{fixed_allocation}'], 
                 label=f"Fixed {fixed_allocation}x", 
                 color=fixed_colors[i], linestyle='-', marker=markers[i], linewidth=2)

    ax1.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax1.set_ylabel('Performance Score', fontsize=font_size + 10)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=font_size - 3, frameon=False)  # 범례를 그래프 밖으로 배치
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # Cost score plot
    ax2.plot(veh_range, results['cost_score_basic'], label="Proposed (Basic)", 
             color='purple', linestyle='--', marker='o', linewidth=2)  # 우리 스킴 Basic: 점선 + 동그라미
    ax2.plot(veh_range, results['cost_score_premium'], label="Proposed (Premium)", 
             color='purple', linestyle='-', marker='o', linewidth=2)  # 우리 스킴 Premium: 실선 + 동그라미
    
    # 각 fixed allocation에 대해 별도로 라인을 그린다 (고정 스킴: 실선 + 각기 다른 색)
    for i, fixed_allocation in enumerate(fixed_allocations):
        ax2.plot(veh_range, results[f'cost_score_fixed_{fixed_allocation}'], 
                 label=f"Fixed {fixed_allocation}x", 
                 color=fixed_colors[i], linestyle='-', marker=markers[i], linewidth=2)

    ax2.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax2.set_ylabel('Cost Score', fontsize=font_size + 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=font_size - 3, frameon=False)  # 범례를 그래프 밖으로 배치
    ax2.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# Main 함수
if __name__ == '__main__':
    veh_range = range(200, 401, 50)  # 차량 수 범위
    repeat = params.repeat

    final_results = {
        'performance_score_basic': [],
        'performance_score_premium': [],
        'cost_score_basic': [],
        'cost_score_premium': []
    }

    fixed_allocations = [1.2, 1.6, 2.0]

    for fixed_allocation in fixed_allocations:
        # 각 fixed allocation에 대한 결과 저장을 위한 키 추가
        final_results[f'performance_score_fixed_{fixed_allocation}'] = []
        final_results[f'cost_score_fixed_{fixed_allocation}'] = []

    for numVeh in veh_range:
        # 우리 스킴 실행 (Basic, Premium)
        avg_results_our = run_experiment_performance_cost(numVeh, repeat, userplan=1)
        final_results['performance_score_basic'].append(avg_results_our['performance_score_basic'])
        final_results['cost_score_basic'].append(avg_results_our['cost_score_basic'])
        final_results['performance_score_premium'].append(avg_results_our['performance_score_premium'])
        final_results['cost_score_premium'].append(avg_results_our['cost_score_premium'])

        # Fixed allocation 실행 (각 fixed allocation에 대해 별도로 저장)
        for fixed_allocation in fixed_allocations:
            avg_results_fixed = run_experiment_performance_cost(numVeh, repeat, userplan=fixed_allocation)

            # 각 fixed allocation에 대해 결과를 저장
            final_results[f'performance_score_fixed_{fixed_allocation}'].append(avg_results_fixed['performance_score_fixed'])
            final_results[f'cost_score_fixed_{fixed_allocation}'].append(avg_results_fixed['cost_score_fixed'])

    # 성능 및 비용 점수 비교 시각화
    plot_performance_cost_comparison(final_results, veh_range, fixed_allocations)
