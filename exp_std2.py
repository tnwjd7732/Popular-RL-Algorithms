import env as environment
import parameters as params
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import clustering
import my_ppo
import no_RL_scheme as schemes
import my_dqn

env = environment.Env()
clst = clustering.Clustering()
nearest, greedy, nearest_cloud, greedy_cloud = schemes.Nearest(), schemes.Greedy(), schemes.NearestCloud(), schemes.GreedyCloud()


# RL 모델 초기화 및 로드
ppo_proposed = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_proposed = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo_ = ppo_proposed.load_model(params.ppo_path)
dqn_ = dqn_proposed.load_model(params.dqn_path)

ppo_static = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_static = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppostatic_ = ppo_static.load_model(params.staticClst_ppo_path)
dqnstatic_ = dqn_static.load_model(params.staticClst_dqn_path)

ppo_woclst = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_woclst = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppowoclst_ = ppo_woclst.load_model(params.woClst_ppo_path)
dqnwoclst_ = dqn_woclst.load_model(params.woClst_dqn_path)

# 각 엣지 서버 위치 초기화
def initialize_edge_positions():
    x = -1
    for i in range(params.numEdge):
        if i % params.grid_size == 0:
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

def run_experiment(veh_range, repeat):
    results = {
        'premium_alloc': [], 'basic_alloc': [],
        'premium_alloc_std': [], 'basic_alloc_std': [],
        'premium_succ_rate': [], 'basic_succ_rate': [],
        'premium_succ_rate_std': [], 'basic_succ_rate_std': [],
        'performance_score_basic': [], 'performance_score_premium': [],
        'cost_score_basic': [], 'cost_score_premium': [],
        'success_score_basic': [], 'success_score_premium': [],
        'premium_users': [], 'basic_users': []
    }

    for numVeh in veh_range:
        alloc_premium = []
        alloc_basic = []
        succ_premium = []
        succ_basic = []
        performance_premium = []
        performance_basic = []
        cost_premium = []
        cost_basic = []
        user_count_premium = 0
        user_count_basic = 0

        for _ in range(repeat):
            params.numVeh = numVeh
            clst.form_cluster()
            state1, state2_temp = env.reset(-1)
            
            premium_alloc_temp = []
            basic_alloc_temp = []
            success_premium = 0
            success_basic = 0
            perf_premium = 0
            perf_basic = 0
            cost_prem_temp = 0
            cost_basic_temp = 0
            
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
                        
                s1_, s2_, _, r1, r2, done = env.step(action1, action2, step)
                
                alloc = env.credit_info[veh_id]
                if env.plan_info[veh_id] == 1:  # Premium
                    premium_alloc_temp.append(alloc)
                    if params.success:
                        success_premium += 1
                        perf_premium += r1
                        cost_prem_temp += r2
                else:  # Basic
                    basic_alloc_temp.append(alloc)
                    if params.success:
                        success_basic += 1
                        perf_basic += r1
                        cost_basic_temp += r2
                        
                state1, state2 = s1_, s2_

            if premium_alloc_temp:
                alloc_premium.append(np.mean(premium_alloc_temp))
                succ_premium.append(success_premium / len(premium_alloc_temp))
                performance_premium.append(perf_premium / (success_premium or 1))
                cost_premium.append(cost_prem_temp / (success_premium or 1))
                user_count_premium = len(premium_alloc_temp)

            if basic_alloc_temp:
                alloc_basic.append(np.mean(basic_alloc_temp))
                succ_basic.append(success_basic / len(basic_alloc_temp))
                performance_basic.append(perf_basic / (success_basic or 1))
                cost_basic.append(cost_basic_temp / (success_basic or 1))
                user_count_basic = len(basic_alloc_temp)

        results['premium_alloc'].append(np.mean(alloc_premium))
        results['basic_alloc'].append(np.mean(alloc_basic))
        results['premium_alloc_std'].append(np.std(alloc_premium))
        results['basic_alloc_std'].append(np.std(alloc_basic))
        results['premium_succ_rate'].append(np.mean(succ_premium))
        results['basic_succ_rate'].append(np.mean(succ_basic))
        results['premium_succ_rate_std'].append(np.std(succ_premium))
        results['basic_succ_rate_std'].append(np.std(succ_basic))
        results['performance_score_premium'].append(np.mean(performance_premium))
        results['performance_score_basic'].append(np.mean(performance_basic))
        results['cost_score_premium'].append(np.mean(cost_premium))
        results['cost_score_basic'].append(np.mean(cost_basic))
        results['success_score_premium'].append(np.mean(succ_premium))
        results['success_score_basic'].append(np.mean(succ_basic))
        results['premium_users'].append(user_count_premium)
        results['basic_users'].append(user_count_basic)

    return results


def plot_results(results, std_range):
    clear_output(True)

    plt.figure(figsize=(16, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    font_size = params.font_size

    # Success Rate Plot (ax1)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(std_range, results['proposed_succ'], label='Proposed', color='purple', linestyle='-', marker='D')
    ax1.plot(std_range, results['staticclst_succ'], label='Static Clustering', color='green', linestyle='--', marker='s')
    ax1.plot(std_range, results['woclst_succ'], label='Without Clustering', color='blue', linestyle='-.', marker='o')
    ax1.set_xlabel('Resource Standard Deviation', fontsize=params.font_size + 10)
    ax1.set_ylabel('Success Rate', fontsize=params.font_size + 10)
    ax1.legend(fontsize=params.font_size)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # Cluster Std Dev Plot (ax2)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(std_range, results['proposed_std'], label='Proposed', color='purple', linestyle='-', marker='D')
    ax2.plot(std_range, results['staticclst_std'], label='Static Clustering', color='green', linestyle='--', marker='s')
    ax2.plot(std_range, results['woclst_std'], label='Without Clustering', color='blue', linestyle='-.', marker='o')
    ax2.set_xlabel('Resource Standard Deviation', fontsize=params.font_size + 10)
    ax2.set_ylabel('Average Cluster Standard Deviation', fontsize=params.font_size + 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def main():
    std_range = range(1, 11, 2)  # 자원 표준편차 범위
    repeat = params.repeat

    final_results = {key: [] for key in [
        'proposed_succ', 'proposed_std', 
        'staticclst_succ', 'staticclst_std', 
        'woclst_succ', 'woclst_std'
    ]}

    for resource_std in std_range:
        avg_results = run_experiment(resource_std, repeat)
        for key in final_results.keys():
            final_results[key].append(avg_results[key])

    plot_results(final_results, std_range)

if __name__ == '__main__':
    initialize_edge_positions()
    main()
