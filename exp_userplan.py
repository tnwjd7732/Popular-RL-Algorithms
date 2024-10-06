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


clst = clustering.Clustering()
def run_experiment(numVeh, repeat, credit_based=True, fixed_allocation=None):
    params.numVeh = numVeh
    results = {
        'premium_succ': [],
        'premium_reward': [],
        'basic_succ': [],
        'basic_reward': [],
        'nocredit_succ_1.22': [],  # 1.22 자원 할당 스킴에 대한 성공률
        'nocredit_reward_1.22': [],  # 1.22 자원 할당 스킴에 대한 리워드
        'nocredit_succ_1.62': [],  # 1.62 자원 할당 스킴에 대한 성공률
        'nocredit_reward_1.62': [],  # 1.62 자원 할당 스킴에 대한 리워드
        'premium_credit_avg': [],  # 프리미엄 유저 크레딧 평균
        'premium_credit_min': [],  # 프리미엄 유저 크레딧 최소값
        'premium_credit_max': [],  # 프리미엄 유저 크레딧 최대값
        'basic_credit_avg': [],  # 베이직 유저 크레딧 평균
        'basic_credit_min': [],  # 베이직 유저 크레딧 최소값
        'basic_credit_max': []  # 베이직 유저 크레딧 최대값
    }

    if credit_based:
        params.userplan = 1  # 크레딧 기반 스킴
    else:
        params.userplan = 0  # 고정 자원 할당 스킴

    env = environment.Env()  # 환경 초기화
    ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
    dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

    ppo = ppo_.load_model(params.ppo_path)
    dqn = dqn_.load_model(params.dqn_path)

    x=-1
    for i in range(params.numEdge):
        if (i % params.grid_size == 0):
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

    for _ in range(repeat):
        fail_premium = 0
        premium_cnt = 0
        basic_cnt = 0
        fail_basic = 0
        fail_nocredit = 0
        premium_reward = 0
        basic_reward = 0
        nocredit_reward = 0
        clst.form_cluster()
        state1, state2_temp = env.reset(-1, 1)

        for step in range(params.STEP * numVeh):
            veh_id = step % numVeh
            if params.userplan == 1:  # 크레딧 기반 스킴 (프리미엄 or 베이직 유저)
                action1 = ppo_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1))
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)
                if env.plan_info[veh_id] == 1:  # 프리미엄 유저
                    premium_reward += r
                    premium_cnt +=1
                    if r == 0:
                        fail_premium += 1
                else:  # 베이직 유저
                    basic_reward += r
                    basic_cnt += 1
                    if r == 0:
                        fail_basic += 1
            else:  # 고정 자원 할당 스킴 (1.22 또는 1.62에 따라)
                action1 = ppo_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1))
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)  # fixed_allocation 값을 사용
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step, 1)
                nocredit_reward += r
                if r == 0:
                    fail_nocredit += 1

            state1 = s1_
            state2 = s2_

        # 성공률 계산
        if credit_based:
            success_ratio_premium = (premium_cnt - fail_premium) / premium_cnt #(params.STEP * numVeh)
            success_ratio_basic = (basic_cnt - fail_basic) / basic_cnt #(params.STEP * numVeh)
            #print("repa:", repeat, "veh:", numVeh, "my", fail_premium,fail_basic, 1-((fail_premium+fail_basic)/(premium_cnt+basic_cnt)), basic_cnt+premium_cnt)
        else:
            success_ratio_nocredit = (params.STEP * numVeh - fail_nocredit) / (params.STEP * numVeh)
        

        # 결과 저장 (프리미엄과 베이직의 성공률과 리워드도 저장)
        if credit_based:
            results['premium_succ'].append(success_ratio_premium)
            #print(success_ratio_premium)
            #print(success_ratio_basic)

            results['premium_reward'].append(premium_reward)
            results['basic_succ'].append(success_ratio_basic)
            results['basic_reward'].append(basic_reward)

        else:
            if fixed_allocation == 1.22:
                results['nocredit_succ_1.22'].append(success_ratio_nocredit)
                results['nocredit_reward_1.22'].append(nocredit_reward)
            elif fixed_allocation == 1.62:
                results['nocredit_succ_1.62'].append(success_ratio_nocredit)
                results['nocredit_reward_1.62'].append(nocredit_reward)

        # 크레딧 기반 스킴이라면 프리미엄과 베이직 크레딧 평균/최소/최대도 저장
        if credit_based:
            premium_credits = [env.credit_info[i] for i in range(numVeh) if env.plan_info[i] == 1]
            basic_credits = [env.credit_info[i] for i in range(numVeh) if env.plan_info[i] == 0]

            premium_credit_avg = np.mean(premium_credits) if premium_credits else 0
            premium_credit_min = np.min(premium_credits) if premium_credits else 0
            premium_credit_max = np.max(premium_credits) if premium_credits else 0

            basic_credit_avg = np.mean(basic_credits) if basic_credits else 0
            basic_credit_min = np.min(basic_credits) if basic_credits else 0
            basic_credit_max = np.max(basic_credits) if basic_credits else 0

            results['premium_credit_avg'].append(premium_credit_avg)
            results['premium_credit_min'].append(premium_credit_min)
            results['premium_credit_max'].append(premium_credit_max)
            results['basic_credit_avg'].append(basic_credit_avg)
            results['basic_credit_min'].append(basic_credit_min)
            results['basic_credit_max'].append(basic_credit_max)
    # 0이 아닌 값들로만 평균을 계산
    avg_results = {key: np.mean([v for v in values if v != 0]) if values and any(values) else 0 for key, values in results.items()}

    #avg_results = {key: np.mean(values) for key, values in results.items()}
    return avg_results



def plot_comparison(results, veh_range):
    clear_output(True)
    plt.rcParams.update({'font.size': params.font_size-5})

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(29, 8))  # 크레딧 정보 그래프 추가를 위해 3개의 subplot

    font_size = params.font_size

    # Success rate comparison between premium, basic, and no-credit users
    ax1.plot(veh_range, results['premium_succ'], label="Premium User Success", linewidth=2)
    ax1.plot(veh_range, results['basic_succ'], label="Basic User Success", linewidth=2)
    ax1.plot(veh_range, results['nocredit_succ_1.22'], label="No-Credit Scheme Success (1.22x)", linewidth=2)
    ax1.plot(veh_range, results['nocredit_succ_1.62'], label="No-Credit Scheme Success (1.62x)", linewidth=2)

    ax1.set_xlabel('Number of Vehicles', fontsize=font_size)
    ax1.legend()
    ax1.set_ylabel('Success rate', fontsize=font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    #ax1.set_ylim(0.7, 0.9)  # y축 범위를 0.7에서 0.9로


    # 크레딧 기반 리워드 합산 (프리미엄 + 베이직)
    premium_reward = np.array(results['premium_reward'])
    basic_reward = np.array(results['basic_reward'])
    nocredit_reward_1_22 = np.array(results['nocredit_reward_1.22'])
    nocredit_reward_1_62 = np.array(results['nocredit_reward_1.62'])

    # Stacked bar chart: 크레딧 기반의 프리미엄과 베이직 기여도
    bar_width = 20  # 막대의 너비 조절
    ax2.bar(veh_range, premium_reward, label="Premium User Reward", color="blue", width=bar_width)
    ax2.bar(veh_range, basic_reward, bottom=premium_reward, label="Basic User Reward", color="orange", width=bar_width)

    # 크레딧 없는 방안의 리워드
    ax2.plot(veh_range, nocredit_reward_1_22, label="No-Credit Scheme Reward (1.22x)", color="green", linewidth=2)
    ax2.plot(veh_range, nocredit_reward_1_62, label="No-Credit Scheme Reward (1.62x)", color="red", linewidth=2)

    ax2.set_xlabel('Number of Vehicles', fontsize=font_size)
    ax2.legend()
    ax2.set_ylabel('Total Reward', fontsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    # 크레딧 정보 비교: 평균, 최소값, 최대값 포함
    premium_credit_avg = np.array(results['premium_credit_avg'])
    premium_credit_min = np.array(results['premium_credit_min'])
    premium_credit_max = np.array(results['premium_credit_max'])

    basic_credit_avg = np.array(results['basic_credit_avg'])
    basic_credit_min = np.array(results['basic_credit_min'])
    basic_credit_max = np.array(results['basic_credit_max'])

    # 프리미엄 유저 크레딧 평균, 최소, 최대 표시
    ax3.errorbar(veh_range, premium_credit_avg, yerr=[premium_credit_avg - premium_credit_min, premium_credit_max - premium_credit_avg],
                 label="Premium User Avg. Credit", fmt='-o', color="blue", capsize=5, linewidth=2)

    # 베이직 유저 크레딧 평균, 최소, 최대 표시
    ax3.errorbar(veh_range, basic_credit_avg, yerr=[basic_credit_avg - basic_credit_min, basic_credit_max - basic_credit_avg],
                 label="Basic User Avg. Credit", fmt='-o', color="orange", capsize=5, linewidth=2)

    ax3.set_xlabel('Number of Vehicles', fontsize=font_size)
    ax3.legend()
    ax3.set_ylabel('Avg. Credit', fontsize=font_size)
    ax3.tick_params(axis='both', which='major', labelsize=font_size)

    plt.show()

if __name__ == '__main__':
    veh_range = range(200, 501, 50)  # 차량 수 범위
    repeat = params.repeat

    final_results = {
        'premium_succ': [],
        'premium_reward': [],
        'basic_succ': [],
        'basic_reward': [],
        'nocredit_succ_1.22': [],
        'nocredit_reward_1.22': [],
        'nocredit_succ_1.62': [],
        'nocredit_reward_1.62': [],
        'premium_credit_avg': [],  # 프리미엄 유저 크레딧 평균
        'premium_credit_min': [],  # 프리미엄 유저 크레딧 최소값
        'premium_credit_max': [],  # 프리미엄 유저 크레딧 최대값
        'basic_credit_avg': [],  # 베이직 유저 크레딧 평균
        'basic_credit_min': [],  # 베이직 유저 크레딧 최소값
        'basic_credit_max': []   # 베이직 유저 크레딧 최대값
    }

    # 각 차량 수에 대해 크레딧 기반 스킴과 고정 자원 할당 스킴을 비교
    for numVeh in veh_range:
        # 크레딧 기반 스킴 (프리미엄, 베이직)
        params.credit = 0
        params.userplan = 1
        avg_results_credit = run_experiment(numVeh, repeat, credit_based=True)
        final_results['premium_succ'].append(avg_results_credit['premium_succ'])
        final_results['premium_reward'].append(avg_results_credit['premium_reward'])
        final_results['basic_succ'].append(avg_results_credit['basic_succ'])
        final_results['basic_reward'].append(avg_results_credit['basic_reward'])
        final_results['premium_credit_avg'].append(avg_results_credit['premium_credit_avg'])
        final_results['premium_credit_min'].append(avg_results_credit['premium_credit_min'])
        final_results['premium_credit_max'].append(avg_results_credit['premium_credit_max'])
        final_results['basic_credit_avg'].append(avg_results_credit['basic_credit_avg'])
        final_results['basic_credit_min'].append(avg_results_credit['basic_credit_min'])
        final_results['basic_credit_max'].append(avg_results_credit['basic_credit_max'])

        # 고정 자원 할당 스킴 실행 (1.22 자원 할당)
        params.credit = 1
        params.userplan = 0
        avg_results_nocredit_1_22 = run_experiment(numVeh, repeat, credit_based=False, fixed_allocation=1.22)
        final_results['nocredit_succ_1.22'].append(avg_results_nocredit_1_22['nocredit_succ_1.22'])
        final_results['nocredit_reward_1.22'].append(avg_results_nocredit_1_22['nocredit_reward_1.22'])

        # 고정 자원 할당 스킴 실행 (1.62 자원 할당)
        params.credit = 2
        params.userplan = 0
        avg_results_nocredit_1_62 = run_experiment(numVeh, repeat, credit_based=False, fixed_allocation=1.62)
        final_results['nocredit_succ_1.62'].append(avg_results_nocredit_1_62['nocredit_succ_1.62'])
        final_results['nocredit_reward_1.62'].append(avg_results_nocredit_1_62['nocredit_reward_1.62'])

    # 성공률, 리워드, 크레딧 평균 및 min, max 비교 시각화
    plot_comparison(final_results, veh_range)

