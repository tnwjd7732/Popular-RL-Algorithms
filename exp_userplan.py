import env as environment
import parameters as params
import numpy as np
import matplotlib.pyplot as plt
import clustering
import my_ppo
import my_dqn
import no_RL_scheme as schemes

env = environment.Env()
clst = clustering.Clustering()

ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo = ppo_.load_model(params.ppo_path)
dqn = dqn_.load_model(params.dqn_path)
nearest = schemes.Nearest()

# 성능 및 비용 점수 계산 함수
def run_experiment_performance_cost(numVeh, repeat, userplan):
    results = {
        'performance_score_fixed': [],
        'cost_score_fixed': [],
        'success_score_fixed': [],
        'performance_score_premium': [],
        'cost_score_premium': [],
        'success_score_premium': [],
        'performance_score_basic': [],
        'cost_score_basic': [],
        'success_score_basic': []
    }
    
    params.numVeh = numVeh
    x = -1
    for i in range(params.numEdge):
        if (i % params.grid_size == 0):
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

    # 고정 스킴 또는 Credit-based 방안의 성공률 계산
    if userplan != 1:
        params.userplan = userplan
        performance_scores_fixed = []
        cost_scores_fixed = []
        success_rates_fixed = []

        for _ in range(repeat):
            clst.form_cluster()
            state1, state2_temp = env.reset(-1)
            episode_reward = 0
            cost = 0
            success_count = 0

            for step in range(params.STEP * numVeh):
                action1 = ppo_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1)) if step == 0 else np.copy(state2)
                state2[-1] = action1
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)
                if len(env.cluster) < action2:
                    if params.task[2] > 0.5:
                        action2 = 0
                    else:
                        _, action2 = nearest.choose_action(step)
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step)

                # 성능 점수 및 비용 점수 계산
                t_tasktime = params.task[2]
                t_total = params.realtime
                performance_score = max(0, (t_tasktime - t_total) / t_tasktime) if t_tasktime != 0 else 0
                c_max = params.max_size * params.unitprice_size * 1.65 + params.max_cpu * params.unitprice_cpu * 1.65
                c_min = params.min_size * params.unitprice_size * 0.85 + params.min_cpu * params.unitprice_cpu * 0.85
                cost_score = ((c_max - params.costpaid) / (c_max - c_min)) / (params.task[0] + params.task[1]) if (c_max - c_min) != 0 else 0
                episode_reward += performance_score
                # 기존 성공률 계산 방식에서 조건을 약간 수정하여 고정 스킴의 특성 반영
                if t_tasktime > t_total:
                    cost += cost_score
                    success_count += 1

                state1, state2 = s1_, s2_

            performance_scores_fixed.append(episode_reward / (params.STEP * numVeh)) # check all user
            cost_scores_fixed.append(cost / success_count) #only count the success user set
            success_rates_fixed.append(success_count / (params.STEP * numVeh))

        results['performance_score_fixed'] = np.mean(performance_scores_fixed)
        results['cost_score_fixed'] = np.mean(cost_scores_fixed)
        results['success_score_fixed'] = np.mean(success_rates_fixed)

    else:
        params.userplan = userplan
        performance_scores_premium = []
        performance_scores_basic = []
        cost_scores_premium = []
        cost_scores_basic = []
        success_rates_premium = []
        success_rates_basic = []

        for _ in range(repeat):
            clst.form_cluster()
            state1, state2_temp = env.reset(-1)
            episode_reward_premium = 0
            episode_reward_basic = 0
            cost_premium = 0
            cost_basic = 0
            pre = 0
            basic = 0
            pre_succ = 0
            basic_succ = 0

            for step in range(params.STEP * numVeh):
                veh_id = step % numVeh
                action1 = ppo_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1)) if step == 0 else np.copy(state2)
                state2[-1] = action1
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)
                if len(env.cluster) < action2:
                    if params.task[2] > 0.5:
                        action2 = 0
                    else:
                        _, action2 = nearest.choose_action(step)
                s1_, s2_, r, r1, r2, done = env.step(action1, action2, step)

                t_tasktime = params.task[2]
                t_total = params.realtime
                performance_score = max(0, (t_tasktime - t_total) / t_tasktime) if t_tasktime != 0 else 0
                c_max = params.max_size * params.unitprice_size * 1.65 + params.max_cpu * params.unitprice_cpu * 1.65
                c_min = params.min_size * params.unitprice_size * 0.85 + params.min_cpu * params.unitprice_cpu * 0.85
                cost_score = ((c_max - params.costpaid) / (c_max - c_min)) / (params.task[0] + params.task[1]) if (c_max - c_min) != 0 else 0

                if env.plan_info[veh_id] == 1:
                    episode_reward_premium += performance_score
                    if t_tasktime > t_total:
                        cost_premium += cost_score
                        pre_succ += 1
                    pre += 1
                else:
                    episode_reward_basic += performance_score
                    if t_tasktime > t_total:
                        cost_basic += cost_score
                        basic_succ += 1
                    basic += 1
                state1, state2 = s1_, s2_

            performance_scores_premium.append(episode_reward_premium / pre if pre else 0)
            performance_scores_basic.append(episode_reward_basic / basic if basic else 0)
            cost_scores_premium.append(cost_premium / pre_succ if pre_succ else 0)
            cost_scores_basic.append(cost_basic / basic_succ if basic_succ else 0)
            success_rates_premium.append(pre_succ / pre if pre else 0)
            success_rates_basic.append(basic_succ / basic if basic else 0)

        results['performance_score_premium'] = np.mean(performance_scores_premium)
        results['cost_score_premium'] = np.mean(cost_scores_premium)
        results['success_score_premium'] = np.mean(success_rates_premium)
        results['performance_score_basic'] = np.mean(performance_scores_basic)
        results['cost_score_basic'] = np.mean(cost_scores_basic)
        results['success_score_basic'] = np.mean(success_rates_basic)

    return results

def plot_performance_cost_comparison(results, veh_range, fixed_allocations):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))  # Success Rate 그래프 추가

    font_size = params.font_size

    # 색상과 스타일 설정
    fixed_colors = ['darkgreen', 'forestgreen', 'lightgreen']
    premium_color = 'purple'
    basic_color = 'purple'
    markers = 'o'
    
    # Performance score plot
    ax1.plot(veh_range, results['performance_score_basic'], label=" (Basic)", 
             color=basic_color, linestyle='--', marker=markers, linewidth=2)
    ax1.plot(veh_range, results['performance_score_premium'], label="Credit-based (Premium)", 
             color=premium_color, linestyle='-', marker=markers, linewidth=2)
    for i, fixed_allocation in enumerate(fixed_allocations):
        ax1.plot(veh_range, results[f'performance_score_fixed_{fixed_allocation}'], 
                 label=f"Fixed {fixed_allocation}x", 
                 color=fixed_colors[i], linestyle='-', marker=markers, linewidth=2)
    ax1.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax1.set_ylabel('Performance Score', fontsize=font_size + 10)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=font_size - 3, frameon=False)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=font_size+10)

    # Cost score plot
    ax2.plot(veh_range, results['cost_score_basic'], label="Credit-based (Basic)", 
             color=basic_color, linestyle='--', marker=markers, linewidth=2)
    ax2.plot(veh_range, results['cost_score_premium'], label="Credit-based (Premium)", 
             color=premium_color, linestyle='-', marker=markers, linewidth=2)
    for i, fixed_allocation in enumerate(fixed_allocations):
        ax2.plot(veh_range, results[f'cost_score_fixed_{fixed_allocation}'], 
                 label=f"Fixed {fixed_allocation}x", 
                 color=fixed_colors[i], linestyle='-', marker=markers, linewidth=2)
    ax2.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax2.set_ylabel('Cost Score', fontsize=font_size + 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=font_size - 3, frameon=False)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=font_size+10)

    # Success rate plot
    ax3.plot(veh_range, results['success_score_basic'], label="Credit-based (Basic)", 
             color=basic_color, linestyle='--', marker=markers, linewidth=2)
    ax3.plot(veh_range, results['success_score_premium'], label="Credit-based (Premium)", 
             color=premium_color, linestyle='-', marker=markers, linewidth=2)
    for i, fixed_allocation in enumerate(fixed_allocations):
        ax3.plot(veh_range, results[f'success_score_fixed_{fixed_allocation}'], 
                 label=f"Fixed {fixed_allocation}x", 
                 color=fixed_colors[i], linestyle='-', marker=markers, linewidth=2)
    ax3.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax3.set_ylabel('Success Rate', fontsize=font_size + 10)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=font_size - 3, frameon=False)
    ax3.grid(True, linestyle='--', linewidth=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=font_size+10)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    veh_range = range(200, 401, 50)
    repeat = params.repeat
    final_results = {
        'performance_score_basic': [],
        'performance_score_premium': [],
        'cost_score_basic': [],
        'cost_score_premium': [],
        'success_score_basic': [],
        'success_score_premium': []
    }

    fixed_allocations = [1.2, 1.6, 2.0]

    for fixed_allocation in fixed_allocations:
        final_results[f'performance_score_fixed_{fixed_allocation}'] = []
        final_results[f'cost_score_fixed_{fixed_allocation}'] = []
        final_results[f'success_score_fixed_{fixed_allocation}'] = []

    for numVeh in veh_range:
        avg_results_our = run_experiment_performance_cost(numVeh, repeat, userplan=1)
        final_results['performance_score_basic'].append(avg_results_our['performance_score_basic'])
        final_results['cost_score_basic'].append(avg_results_our['cost_score_basic'])
        final_results['success_score_basic'].append(avg_results_our['success_score_basic'])
        final_results['performance_score_premium'].append(avg_results_our['performance_score_premium'])
        final_results['cost_score_premium'].append(avg_results_our['cost_score_premium'])
        final_results['success_score_premium'].append(avg_results_our['success_score_premium'])

        for fixed_allocation in fixed_allocations:
            avg_results_fixed = run_experiment_performance_cost(numVeh, repeat, userplan=fixed_allocation)
            final_results[f'performance_score_fixed_{fixed_allocation}'].append(avg_results_fixed['performance_score_fixed'])
            final_results[f'cost_score_fixed_{fixed_allocation}'].append(avg_results_fixed['cost_score_fixed'])
            final_results[f'success_score_fixed_{fixed_allocation}'].append(avg_results_fixed['success_score_fixed'])

    plot_performance_cost_comparison(final_results, veh_range, fixed_allocations)
