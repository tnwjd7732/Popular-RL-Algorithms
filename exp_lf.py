import env as environment
import parameters as params
import numpy as np
import matplotlib.pyplot as plt
import clustering
import my_ppo
import my_dqn
import no_RL_scheme as schemes

# 환경 및 모델 초기화
env = environment.Env()
clst = clustering.Clustering()

ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo = ppo_.load_model(params.ppo_path)
dqn = dqn_.load_model(params.dqn_path)
nearest = schemes.Nearest()

# 실험 함수
def run_experiment(veh_range, repeat, fixed_allocations):
    results = {
        'premium_alloc': [], 'basic_alloc': [],
        'premium_alloc_std': [], 'basic_alloc_std': [],
        'premium_succ_rate': [], 'basic_succ_rate': [],
        'premium_succ_rate_std': [], 'basic_succ_rate_std': [],
        'fixed_allocations': {alloc: {'alloc': [], 'alloc_std': [], 'succ_rate': [], 'succ_rate_std': []} for alloc in fixed_allocations}
    }

    for numVeh in veh_range:
        alloc_premium = []
        alloc_basic = []
        succ_premium = []
        succ_basic = []

        for _ in range(repeat):
            params.numVeh = numVeh
            clst.form_cluster()
            state1, state2_temp = env.reset(-1)

            premium_alloc_temp = []
            basic_alloc_temp = []
            success_premium = 0
            success_basic = 0

            for step in range(params.STEP * numVeh):
                veh_id = step % numVeh
                action1 = ppo_.choose_action(state1)
                state2 = np.concatenate((state2_temp, action1)) if step == 0 else np.copy(state2)
                state2[-1] = action1
                params.state2 = state2
                action2 = dqn_.choose_action(state2, 1)

                if len(env.cluster) < action2:
                    action2 = 0

                s1_, s2_, _, _, _, done = env.step(action1, action2, step)

                alloc = env.credit_info[veh_id]
                if env.plan_info[veh_id] == 1:  # Premium
                    
                    if params.success:
                        success_premium += 1
                        premium_alloc_temp.append(alloc)
                    else:
                        premium_alloc_temp.append(0)
                else:  # Basic
                    
                    if params.success:
                        success_basic += 1
                        basic_alloc_temp.append(alloc)
                    else:
                        basic_alloc_temp.append(0)

                state1, state2 = s1_, s2_

            if premium_alloc_temp:
                alloc_premium.append(np.mean(premium_alloc_temp))
                succ_premium.append(success_premium / len(premium_alloc_temp))
            if basic_alloc_temp:
                alloc_basic.append(np.mean(basic_alloc_temp))
                succ_basic.append(success_basic / len(basic_alloc_temp))

        results['premium_alloc'].append(np.mean(alloc_premium))
        results['basic_alloc'].append(np.mean(alloc_basic))
        results['premium_alloc_std'].append(np.std(alloc_premium))
        results['basic_alloc_std'].append(np.std(alloc_basic))
        results['premium_succ_rate'].append(np.mean(succ_premium))
        results['basic_succ_rate'].append(np.mean(succ_basic))
        results['premium_succ_rate_std'].append(np.std(succ_premium))
        results['basic_succ_rate_std'].append(np.std(succ_basic))

        # 고정 할당 스킴 실험
        for alloc in fixed_allocations:
            fixed_alloc_temp = []
            success_fixed = []

            for _ in range(repeat):
                params.numVeh = numVeh
                params.userplan = alloc  # 고정 할당
                clst.form_cluster()
                env.reset(-1)

                alloc_temp_fixed = []
                success_fixed_temp = 0

                for step in range(params.STEP * numVeh):
                    veh_id = step % numVeh
                    action1 = ppo_.choose_action(state1)
                    state2 = np.concatenate((state2_temp, action1)) if step == 0 else np.copy(state2)
                    state2[-1] = action1
                    params.state2 = state2
                    action2 = dqn_.choose_action(state2, 1)

                    if len(env.cluster) < action2:
                        action2 = 0

                    s1_, s2_, _, _, _, done = env.step(action1, action2, step)

                    alloc = env.credit_info[veh_id]
                    if params.success:
                        success_fixed_temp += 1
                        alloc_temp_fixed.append(alloc)
                    else:
                        alloc_temp_fixed.append(0)
                    state1, state2 = s1_, s2_

                fixed_alloc_temp.append(np.mean(alloc_temp_fixed))
                success_fixed.append(success_fixed_temp / (params.STEP * numVeh))

            results['fixed_allocations'][alloc]['alloc'].append(np.mean(fixed_alloc_temp))
            results['fixed_allocations'][alloc]['alloc_std'].append(np.std(fixed_alloc_temp))
            results['fixed_allocations'][alloc]['succ_rate'].append(np.mean(success_fixed))
            results['fixed_allocations'][alloc]['succ_rate_std'].append(np.std(success_fixed))

    return results

# 결과 시각화 함수
def plot_results(results, veh_range, fixed_allocations):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

    font_size = params.font_size
    ax1.errorbar(veh_range, results['premium_alloc'], yerr=results['premium_alloc_std'], label='Premium Allocation', color='purple', linestyle='-', marker='o', capsize=5)
    ax1.errorbar(veh_range, results['basic_alloc'], yerr=results['basic_alloc_std'], label='Basic Allocation', color='blue', linestyle='--', marker='s', capsize=5)
    for alloc in fixed_allocations:
        ax1.errorbar(veh_range, results['fixed_allocations'][alloc]['alloc'], yerr=results['fixed_allocations'][alloc]['alloc_std'], label=f'Fixed {alloc}x Allocation', linestyle=':', marker='^', capsize=5)
    ax1.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax1.set_ylabel('Average Resource Allocation', fontsize=font_size + 10)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=font_size+10)

    ax2.errorbar(veh_range, results['premium_succ_rate'], yerr=results['premium_succ_rate_std'], label='Premium Success Rate', color='purple', linestyle='-', marker='o', capsize=5)
    ax2.errorbar(veh_range, results['basic_succ_rate'], yerr=results['basic_succ_rate_std'], label='Basic Success Rate', color='blue', linestyle='--', marker='s', capsize=5)
    for alloc in fixed_allocations:
        ax2.errorbar(veh_range, results['fixed_allocations'][alloc]['succ_rate'], yerr=results['fixed_allocations'][alloc]['succ_rate_std'], label=f'Fixed {alloc}x Success Rate', linestyle=':', marker='^', capsize=5)
    ax2.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax2.set_ylabel('Success Rate', fontsize=font_size + 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=font_size+10)


    plt.tight_layout()
    plt.show()

# 메인 함수
if __name__ == '__main__':
    veh_range = range(500, 701, 50)
    repeat = params.repeat
    fixed_allocations = [1.2, 1.6, 2.0]

    final_results = run_experiment(veh_range, repeat, fixed_allocations)
    plot_results(final_results, veh_range, fixed_allocations)
