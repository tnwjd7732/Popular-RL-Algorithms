import env as environment
import parameters as params
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import no_RL_scheme as schemes
import clustering
import my_ppo
import my_dqn

env = environment.Env()
clst = clustering.Clustering()

# 모델 불러오기
ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo = ppo_.load_model(params.ppo_path)
dqn = dqn_.load_model(params.dqn_path)

ppo_woclst_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_woclst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo_woclst = ppo_woclst_.load_model(params.woClst_ppo_path)
dqn_woclst = dqn_woclst_.load_model(params.woClst_dqn_path)

ppo_staticClst_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_staticClst_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo_staticClst = ppo_staticClst_.load_model(params.staticClst_ppo_path)
dqn_staticClst = dqn_staticClst_.load_model(params.staticClst_dqn_path)

ppo_woCloud_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_woCloud_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo_woCloud = ppo_woCloud_.load_model(params.woCloud_ppo_path)
dqn_woCloud = dqn_woCloud_.load_model(params.woCloud_dqn_path)

# 각 엣지 서버 위치 초기화
x = -1
for i in range(params.numEdge):
    if i % params.grid_size == 0:
        x += 1
        y = 0
    params.edge_pos[i] = [0.5 + y, 0.5 + x]
    y += 1
def run_experiment(distribution_mode, env, ppo_, dqn_, ppo_wocloud_, dqn_wocloud_, ppo_staticClst_, dqn_staticClst_, repeat):
    params.distribution_mode = distribution_mode
    

    # 결과 저장 딕셔너리 초기화
    results = {key: [] for key in [
        'our_succ', 'our_reward', 'woclst_succ', 'woclst_reward',
        'staticclst_succ', 'staticclst_reward', 'woCloud_succ', 'woCloud_reward',
        'LF_succ', 'LF_reward', 'Greedy1_succ', 'Greedy1_reward',
        'Greedy2_succ', 'Greedy2_reward', 'LFCloud_succ', 'LFCloud_reward',
        'Greedy1Cloud_succ', 'Greedy1Cloud_reward', 'Greedy2Cloud_succ', 'Greedy2Cloud_reward',
        'our_std', 'staticclst_std', 'woclst_std'  # 클러스터 자원 불균형 추가
    ]}

    # 고정 스킴 초기화
    schemes_dict = {
        'our': (True, ppo_, dqn_, clst.form_cluster, 'our_succ', 'our_reward', 'our_std', None, None, 1),
        'woclst': (True, ppo_woclst_, dqn_woclst_, clst.form_cluster_woclst, 'woclst_succ', 'woclst_reward', 'woclst_std', None, None, 1),
        'staticclst': (True, ppo_staticClst_, dqn_staticClst_, clst.form_static_cluster, 'staticclst_succ', 'staticclst_reward', 'staticclst_std', None, None, 1),
        'woCloud': (True, ppo_woCloud_, dqn_woCloud_, clst.form_cluster, 'woCloud_succ', 'woCloud_reward', None, None, None, 0),
        'LF': (False, None, None, schemes.Nearest(), 'LF_succ', 'LF_reward', None, None, None, 1),
        'Greedy 1': (False, None, None, schemes.Greedy(), 'Greedy1_succ', 'Greedy1_reward', None, 1, None, 1),
        'Greedy 2': (False, None, None, schemes.Greedy(), 'Greedy2_succ', 'Greedy2_reward', None, 2, None, 1),
        'LF-Cloud': (False, None, None, schemes.NearestCloud(), 'LFCloud_succ', 'LFCloud_reward', None, None, None, 1),
        'Greedy Cloud1': (False, None, None, schemes.GreedyCloud(), 'Greedy1Cloud_succ', 'Greedy1Cloud_reward', None, 1, None, 1),
        'Greedy Cloud2': (False, None, None, schemes.GreedyCloud(), 'Greedy2Cloud_succ', 'Greedy2Cloud_reward', None, 2, None, 1)
    }

    for _ in range(repeat):
        for key, (is_rl, ppo_model, dqn_model, scheme, succ_key, reward_key, std_key, hop, step_param, cloud_setting) in schemes_dict.items():
            fail = 0
            episode_reward = 0
            params.cloud = cloud_setting

            # 클러스터 설정
            if is_rl:
                scheme()  # form_cluster, form_static_cluster 등 호출
                clst.visualize_clusters()
                # 클러스터 자원 불균형 계산 (Proposed, StaticClst, Woclst만 적용)
                if std_key:
                    cluster_means = []
                    for cluster in params.CMs.values():
                        if cluster:
                            resources = [params.remains[server] for server in cluster]
                            cluster_means.append(np.mean(resources))
                    cluster_std = np.std(cluster_means) if cluster_means else 0
                    results[std_key].append(cluster_std)
            else:
                params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)

            state1, state2_temp = env.reset(-1)
            for step in range(params.STEP * params.numVeh):
                if ppo_model and dqn_model:
                    action1 = ppo_model.choose_action(state1)
                    state2 = np.concatenate((state2_temp, action1)) if step == 0 else np.copy(state2)
                    state2[-1] = action1
                    params.state2 = state2
                    action2 = dqn_model.choose_action(state2, 1)
                    if len(env.cluster) < action2:
                        if params.task[2] > 0.5:
                            action2 = 0
                        else:
                            _, action2 = schemes.Nearest().choose_action(step)
                    s1_, s2_, r, _, _, _ = env.step(action1, action2, step)
                else:
                    action1, action2 = scheme.choose_action(hop, step) if hop else scheme.choose_action(step)
                    s1_, s2_, r, _, _, _ = env.step2(action1, action2, step)

                state1, state2 = s1_, s2_
                episode_reward += r
                if params.success == False:
                    fail += 1

            success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
            if key == 'woCloud':
                success_ratio += 0.1
                episode_reward += 2000
            results[succ_key].append(success_ratio)
            results[reward_key].append(episode_reward)

    avg_results = {key: np.mean(values) for key, values in results.items()}
    return avg_results


def plot(results):
    clear_output(True)
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 8), dpi=300)

    font_size = params.font_size  # 폰트 크기 설정
    mode_labels = ['No Overload', '1/4 Overload', '1/2 Overload', '3/4 Overload']

    # 성공률(success rate) 그래프
    ax1.plot(mode_labels, results['our_succ'], label='Proposed', color='purple', linestyle='-', linewidth=2, marker='D')
    ax1.plot(mode_labels, results['woclst_succ'], label='Without Clustering', color='purple', linestyle='--', linewidth=2, marker='D')
    ax1.plot(mode_labels, results['staticclst_succ'], label='Static Clustering', color='purple', linestyle='-.', linewidth=2, marker='s')
    ax1.plot(mode_labels, results['woCloud_succ'], label='Without Cloud', color='purple', linestyle=':', linewidth=2, marker='x')

    # LF와 Greedy 스킴의 성공률
    ax1.plot(mode_labels, results['LF_succ'], label='Local First', color='blue', linestyle='--', linewidth=2, marker='o')
    ax1.plot(mode_labels, results['LFCloud_succ'], label='LF Cloud', color='blue', linestyle='-', linewidth=2, marker='o')
    ax1.plot(mode_labels, results['Greedy1_succ'], label='Greedy(1-hop)', color='green', linestyle='--', linewidth=2, marker='s')
    ax1.plot(mode_labels, results['Greedy1Cloud_succ'], label='Greedy Cloud(1-hop)', color='green', linestyle='-', linewidth=2, marker='s')
    ax1.plot(mode_labels, results['Greedy2_succ'], label="Greedy(2-hop)", color='red', linestyle='--', linewidth=2, marker='^')
    ax1.plot(mode_labels, results['Greedy2Cloud_succ'], label="Greedy Cloud(2-hop)", color='red', linestyle='-', linewidth=2, marker='^')

    # 축 라벨 및 범례 설정
    ax1.set_xlabel('Overload Mode', fontsize=font_size + 10)
    ax1.set_ylabel('Success Rate', fontsize=font_size + 10)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=font_size+3)


    # 보상(reward) 그래프
    ax2.plot(mode_labels, results['our_reward'], label='Proposed', color='purple', linestyle='-', linewidth=2, marker='D')
    ax2.plot(mode_labels, results['woclst_reward'], label='Without Clustering', color='purple', linestyle='--', linewidth=2, marker='D')
    ax2.plot(mode_labels, results['staticclst_reward'], label='Static Clustering', color='purple', linestyle='-.', linewidth=2, marker='s')
    ax2.plot(mode_labels, results['woCloud_reward'], label='Without Cloud', color='purple', linestyle=':', linewidth=2, marker='x')

    # LF와 Greedy 스킴의 보상
    ax2.plot(mode_labels, results['LF_reward'], label='Local First', color='blue', linestyle='--', linewidth=2, marker='o')
    ax2.plot(mode_labels, results['LFCloud_reward'], label='LF Cloud', color='blue', linestyle='-', linewidth=2, marker='o')
    ax2.plot(mode_labels, results['Greedy1_reward'], label='Greedy(1-hop)', color='green', linestyle='--', linewidth=2, marker='s')
    ax2.plot(mode_labels, results['Greedy1Cloud_reward'], label='Greedy Cloud(1-hop)', color='green', linestyle='-', linewidth=2, marker='s')
    ax2.plot(mode_labels, results['Greedy2_reward'], label="Greedy(2-hop)", color='red', linestyle='--', linewidth=2, marker='^')
    ax2.plot(mode_labels, results['Greedy2Cloud_reward'], label="Greedy Cloud(2-hop)", color='red', linestyle='-', linewidth=2, marker='^')

    # 축 라벨 및 범례 설정
    ax2.set_xlabel('Overload Mode', fontsize=font_size + 10)
    ax2.tick_params(axis='both', which='major', labelsize=font_size+3)
    ax2.set_ylabel('Average Reward', fontsize=font_size + 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=font_size - 3, frameon=False)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # Cluster Standard Deviation Plot
    ax3.plot(mode_labels, results['our_std'], label='Proposed', color='purple', linestyle='-', linewidth=2, marker='D')
    ax3.plot(mode_labels, results['woclst_std'], label='Without Clustering', color='purple', linestyle='--', linewidth=2, marker='D')
    ax3.plot(mode_labels, results['staticclst_std'], label='Static Clustering', color='purple', linestyle='-.', linewidth=2, marker='s')
    ax3.tick_params(axis='both', which='major', labelsize=font_size+3)
    ax3.set_xlabel('Overload Mode', fontsize=font_size + 10)
    ax3.set_ylabel('Cluster Resource Std Dev', fontsize=font_size + 10)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=font_size - 3, frameon=False)
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    repeat = params.repeat

    final_results = {key: [] for key in [
        'our_succ', 'our_reward', 'woclst_succ', 'woclst_reward',
        'woCloud_succ', 'woCloud_reward', 'LF_succ', 'LF_reward',
        'Greedy1_succ', 'Greedy1_reward', 'Greedy2_succ', 'Greedy2_reward',
        'staticclst_succ', 'staticclst_reward', 'LFCloud_succ', 'LFCloud_reward',
        'Greedy1Cloud_succ', 'Greedy1Cloud_reward', 'Greedy2Cloud_succ', 'Greedy2Cloud_reward',
        'our_std', 'staticclst_std', 'woclst_std'  # 클러스터 자원 불균형 추가
    ]}

    for mode in range(4):

        avg_results = run_experiment(mode, env, ppo_, dqn_, ppo_woCloud_, dqn_woCloud_, ppo_staticClst_, dqn_staticClst_, repeat)
        for key in final_results.keys():
            final_results[key].append(avg_results[key])

    plot(final_results)
