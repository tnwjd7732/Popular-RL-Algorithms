import env as environment
import parameters as params
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import no_RL_scheme as schemes
import clustering
import my_ppo
import my_dqn

def run_experiment(numVeh, repeat):
    params.numVeh = numVeh
    results = {key: [] for key in [
        'our_succ', 'our_reward', 'wocloud_succ', 'wocloud_reward', 
        'near_succ', 'near_reward', 'gree_succ', 'gree_reward', 
        'gree2_succ', 'gree2_reward', 'cloud_near_succ', 'cloud_near_reward', 
        'cloud_gree_succ', 'cloud_gree_reward', 'cloud_gree2_succ', 'cloud_gree2_reward'
    ]}

    env = environment.Env()
    # RL 모델 초기화
    params.cloud = 1
    ppo_model = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
    dqn_model = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    ppo_ = ppo_model.load_model(params.ppo_path)
    dqn_ = dqn_model.load_model(params.dqn_path)
    
    params.cloud = 0
    
    # Without cloud RL 모델 초기화
    ppo_wocloud, dqn_wocloud = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim), my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    ppo_wocloud_ = ppo_wocloud.load_model(params.woCloud_ppo_path)
    dqn_wocloud_ = dqn_wocloud.load_model(params.woCloud_dqn_path)
    params.cloud = 1

    # Non-RL 스킴 객체 생성
    nearest, greedy, nearest_cloud, greedy_cloud = schemes.Nearest(), schemes.Greedy(), schemes.NearestCloud(), schemes.GreedyCloud()
    clst = clustering.Clustering()

    # 스킴 딕셔너리 정의
    schemes_dict = {
        'our': ('our_succ', 'our_reward', True, ppo_model, dqn_model, clst.form_cluster, None, 1),
        'wocloud': ('wocloud_succ', 'wocloud_reward', True, ppo_wocloud, dqn_wocloud, clst.form_cluster, None, 0),
        'nearest': ('near_succ', 'near_reward', False, nearest, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), None, 1),
        'nearest_cloud': ('cloud_near_succ', 'cloud_near_reward', False, nearest_cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), None, 1),
        'greedy_1': ('gree_succ', 'gree_reward', False, greedy, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 1, 1),
        'greedy_2': ('gree2_succ', 'gree2_reward', False, greedy, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 2, 1),
        'greedy_cloud_1': ('cloud_gree_succ', 'cloud_gree_reward', False, greedy_cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 1, 1),
        'greedy_cloud_2': ('cloud_gree2_succ', 'cloud_gree_reward', False, greedy_cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 2, 1),
    }

    for _ in range(repeat):
        set_positions(params.numEdge, params.grid_size)

        for key, (succ_key, reward_key, is_rl, model, secondary_model, setup, hop, cloud_setting) in schemes_dict.items():
            fail = 0
            params.cloud = cloud_setting  # 각 스킴에 맞게 클라우드 설정 적용
            episode_reward = 0

            # 자원 초기화: RL 스킴은 클러스터 형성, 비-RL 스킴은 random_list로 자원 초기화
            if is_rl:
                clst.form_cluster()  # RL 스킴의 경우 클러스터 초기화
            else:
                params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)  # 비-RL 스킴의 경우 random_list로 자원 초기화

            state1, state2_temp = env.reset(-1)

            for step in range(params.STEP * numVeh):
                if is_rl:  # RL 스킴
                    action1 = model.choose_action(state1)
                    state2 = np.concatenate((state2_temp, action1)) if step == 0 else np.copy(state2)
                    state2[-1] = action1
                    params.state2 = state2
                    action2 = secondary_model.choose_action(state2, 1)
                    if len(env.cluster) < action2:
                        if params.task[2] > 0.5:
                            action2 = 0
                        else:
                            _, action2 = nearest.choose_action(step)
                    s1_, s2_, r, _, _, _ = env.step(action1, action2, step)
                else:  # Non-RL 스킴
                    action1, action2 = (model.choose_action(hop, step) if hop else model.choose_action(step))
                    s1_, s2_, r, *_ = env.step2(action1, action2, step)

                state1, state2 = s1_, s2_
                episode_reward += r
                if r == 0:
                    fail += 1
            success_rate = (params.STEP * numVeh - fail) / (params.STEP * numVeh)
            
            results[succ_key].append(success_rate)
            results[reward_key].append(episode_reward)

    return {key: np.mean(values) for key, values in results.items()}



def set_positions(numEdge, grid_size):
    x = -1
    for i in range(numEdge):
        if i % grid_size == 0:
            x += 1
            y = 0
        params.edge_pos[i] = [0.5 + y, 0.5 + x]
        y += 1

def plot(results, veh_range):
    clear_output(True)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': params.font_size - 5})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    font_size = params.font_size

    # LF 스킴
    ax1.plot(veh_range, results['near_succ'], label='LF', color='blue', linestyle='--', linewidth=2, marker='o')
    ax1.plot(veh_range, results['cloud_near_succ'], label='LF Cloud', color='blue', linestyle='-', linewidth=2, marker='o')

    # Greedy(1-hop)
    ax1.plot(veh_range, results['gree_succ'], label='Greedy(1-hop)', color='green', linestyle='--', linewidth=2, marker='s')
    ax1.plot(veh_range, results['cloud_gree_succ'], label='Greedy Cloud(1-hop)', color='green', linestyle='-', linewidth=2, marker='s')

    # Greedy(2-hop)
    ax1.plot(veh_range, results['gree2_succ'], label="Greedy(2-hop)", color='red', linestyle='--', linewidth=2, marker='^')
    ax1.plot(veh_range, results['cloud_gree2_succ'], label="Greedy Cloud(2-hop)", color='red', linestyle='-', linewidth=2, marker='^')

    # Proposed 스킴
    ax1.plot(veh_range, results['our_succ'], label="Proposed", color='purple', linestyle='-', linewidth=2, marker='D')
    ax1.plot(veh_range, results['wocloud_succ'], label="Proposed w/o Cloud", color='purple', linestyle='--', linewidth=2, marker='D')

    ax1.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax1.set_ylabel('Success Ratio', fontsize=font_size + 10)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=font_size+10)

    ax2.plot(veh_range, results['near_reward'], label='LF', color='blue', linestyle='--', linewidth=2, marker='o') 
    ax2.plot(veh_range, results['cloud_near_reward'], label='LF Cloud', color='blue', linestyle='-', linewidth=2, marker='o')

    ax2.plot(veh_range, results['gree_reward'], label='Greedy(1-hop)', color='green', linestyle='--', linewidth=2, marker='s')
    ax2.plot(veh_range, results['cloud_gree_reward'], label='Greedy Cloud(1-hop)', color='green', linestyle='-', linewidth=2, marker='s')

    ax2.plot(veh_range, results['gree2_reward'], label="Greedy(2-hop)", color='red', linestyle='--', linewidth=2, marker='^')
    ax2.plot(veh_range, results['cloud_gree2_reward'], label="Greedy Cloud(2-hop)", color='red', linestyle='-', linewidth=2, marker='^')

    ax2.plot(veh_range, results['our_reward'], label="Proposed", color='purple', linestyle='-', linewidth=2, marker='D')
    ax2.plot(veh_range, results['wocloud_reward'], label="Proposed w/o Cloud", color='purple', linestyle='--', linewidth=2, marker='D')

    ax2.set_xlabel('Number of Vehicles', fontsize=font_size + 10)
    ax2.set_ylabel('Average Reward', fontsize=font_size + 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax2.tick_params(axis='both', which='major', labelsize=font_size+10)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

import parameters as params
from exp_numVeh import run_experiment, plot  # run_experiment와 plot 함수 가져오기

def main():
    veh_range = range(200, 401, 100)  # 차량 수 범위 설정
    repeat = params.repeat  # 반복 횟수 설정

    # 결과를 저장할 딕셔너리 초기화
    final_results = {key: [] for key in [
        'our_succ', 'our_reward', 'wocloud_succ', 'wocloud_reward',
        'near_succ', 'near_reward', 'gree_succ', 'gree_reward',
        'gree2_succ', 'gree2_reward', 'cloud_near_succ', 'cloud_near_reward',
        'cloud_gree_succ', 'cloud_gree_reward', 'cloud_gree2_succ', 'cloud_gree2_reward'
    ]}

    # 각 차량 수에 대해 실험 실행
    for numVeh in veh_range:
        avg_results = run_experiment(numVeh, repeat)  # 각 차량 수에 대해 실험 수행
        for key in final_results.keys():
            final_results[key].append(avg_results[key])  # 결과 저장

    # 결과 그래프 출력
    plot(final_results, veh_range)

if __name__ == '__main__':
    main()
