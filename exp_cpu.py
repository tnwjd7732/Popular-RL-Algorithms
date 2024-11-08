import env as environment
import parameters as params
import numpy as np
import matplotlib.pyplot as plt
import no_RL_scheme as schemes
import clustering
import my_ppo
import my_dqn

def run_experiment(task_cpu, repeat):
    params.min_cpu = task_cpu
    params.max_cpu = task_cpu
    results = {
        'our_succ': [], 'our_reward': [],
        'wocloud_succ': [], 'wocloud_reward': [],
        'LF_succ': [], 'LF_reward': [],
        'Greedy1_succ': [], 'Greedy1_reward': [],
        'Greedy2_succ': [], 'Greedy2_reward': [],
        'LFCloud_succ': [], 'LFCloud_reward': [],
        'Greedy1Cloud_succ': [], 'Greedy1Cloud_reward': [],
        'Greedy2Cloud_succ': [], 'Greedy2Cloud_reward': []
    }

    env = environment.Env()
    clst = clustering.Clustering()

    # 모델 불러오기
    ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
    dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    ppo = ppo_.load_model(params.ppo_path)
    dqn = dqn_.load_model(params.dqn_path)

    params.cloud = 0
    ppo_wocloud_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
    dqn_wocloud_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
    ppo_wocloud = ppo_wocloud_.load_model(params.woCloud_ppo_path)
    dqn_wocloud = dqn_wocloud_.load_model(params.woCloud_dqn_path)
    params.cloud = 1

    # 비교 스킴 초기화
    nearest = schemes.Nearest()
    greedy1 = schemes.Greedy()
    greedy2 = schemes.Greedy()
    LF = schemes.Nearest()
    LFCloud = schemes.NearestCloud()
    Greedy1Cloud = schemes.GreedyCloud()
    Greedy2Cloud = schemes.GreedyCloud()

    schemes_dict = {
        'our': ('our_succ', 'our_reward', True, ppo_model, dqn_model, clst.form_cluster, None, 1),
        'wocloud': ('wocloud_succ', 'wocloud_reward', True, ppo_wocloud, dqn_wocloud, clst.form_cluster, None, 0),
        'nearest': ('near_succ', 'near_reward', False, nearest, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), None, 1),
        'nearest_cloud': ('cloud_near_succ', 'cloud_near_reward', False, LFCloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), None, 1),
        'greedy_1': ('gree_succ', 'gree_reward', False, greedy1, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 1, 1),
        'greedy_2': ('gree2_succ', 'gree2_reward', False, greedy2, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 2, 1),
        'greedy_cloud_1': ('cloud_gree_succ', 'cloud_gree_reward', False, Greedy1Cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 1, 1),
        'greedy_cloud_2': ('cloud_gree2_succ', 'cloud_gree_reward', False, Greedy2Cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 2, 1),
    }

    for _ in range(repeat):
        x = -1
        for i in range(params.numEdge):
            if i % params.grid_size == 0:
                x += 1
                y = 0
            params.edge_pos[i] = [0.5 + y, 0.5 + x]
            y += 1

        for key, (succ_key, reward_key, is_rl, ppo_model, dqn_model, setup, hop, cloud_setting) in schemes_dict.items():
            fail = 0
            params.cloud = cloud_setting  # 각 스킴에 맞게 클라우드 설정 적용
            episode_reward = 0

            # 자원 초기화: RL 스킴은 클러스터 형성, 비-RL 스킴은 random_list로 자원 초기화
            if is_rl:
                clst.form_cluster()  # RL 스킴의 경우 클러스터 초기화
            else:
                params.remains = clst.random_list(params.numEdge, params.resource_avg, params.resource_std)  # 비-RL 스킴의 경우 random_list로 자원 초기화

            state1, state2_temp = env.reset(-1)
            for step in range(params.STEP * params.numVeh):
                if ppo_model and dqn_model:
                    action1 = ppo_model.choose_action(state1)
                    state2 = np.concatenate((state2_temp, action1)) if step == 0 else np.copy(state2)
                    state2[-1] = action1
                    params.state2 = state2
                    action2 = dqn_model.choose_action(state2, 1)
                    s1_, s2_, r, _, _, _ = env.step(action1, action2, step)
                else:
                    if hop is not None:
                        action1, action2 = ppo_model.choose_action(hop, step)
                    else:
                        action1, action2 = ppo_model.choose_action()
                    s1_, s2_, r, _, _, _ = env.step2(action1, action2, step)
                
                state1, state2 = s1_, s2_
                episode_reward += r
                if r == 0:
                    fail += 1
            
            success_ratio = (params.STEP * params.numVeh - fail) / (params.STEP * params.numVeh)
            results[succ_key].append(success_ratio)
            results[reward_key].append(episode_reward)

    avg_results = {key: np.mean(values) for key, values in results.items()}
    return avg_results

def plot(results, task_cpu_range):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    font_size = params.font_size

    # 성공률 플롯
    ax1.plot(task_cpu_range, results['our_succ'], label="Proposed", linewidth=2)
    ax1.plot(task_cpu_range, results['wocloud_succ'], label="Without Cloud", linewidth=2)
    ax1.plot(task_cpu_range, results['LF_succ'], label="Local First", linewidth=2)
    ax1.plot(task_cpu_range, results['LFCloud_succ'], label="Local First Cloud", linewidth=2)
    ax1.plot(task_cpu_range, results['Greedy1_succ'], label="Greedy(1-hop)", linewidth=2)
    ax1.plot(task_cpu_range, results['Greedy2_succ'], label="Greedy(2-hop)", linewidth=2)
    ax1.plot(task_cpu_range, results['Greedy1Cloud_succ'], label="Greedy Cloud(1-hop)", linewidth=2)
    ax1.plot(task_cpu_range, results['Greedy2Cloud_succ'], label="Greedy Cloud(2-hop)", linewidth=2)
    ax1.set_xlabel('Task CPU', fontsize=font_size + 10)
    ax1.set_ylabel('Success Rate', fontsize=font_size + 10)
    ax1.legend()
    ax1.grid(True)

    # 리워드 플롯
    ax2.plot(task_cpu_range, results['our_reward'], label="Proposed", linewidth=2)
    ax2.plot(task_cpu_range, results['wocloud_reward'], label="Without Cloud", linewidth=2)
    ax2.plot(task_cpu_range, results['LF_reward'], label="Local First", linewidth=2)
    ax2.plot(task_cpu_range, results['LFCloud_reward'], label="Local First Cloud", linewidth=2)
    ax2.plot(task_cpu_range, results['Greedy1_reward'], label="Greedy(1-hop)", linewidth=2)
    ax2.plot(task_cpu_range, results['Greedy2_reward'], label="Greedy(2-hop)", linewidth=2)
    ax2.plot(task_cpu_range, results['Greedy1Cloud_reward'], label="Greedy Cloud(1-hop)", linewidth=2)
    ax2.plot(task_cpu_range, results['Greedy2Cloud_reward'], label="Greedy Cloud(2-hop)", linewidth=2)
    ax2.set_xlabel('Task CPU', fontsize=font_size + 10)
    ax2.set_ylabel('Average Reward', fontsize=font_size + 10)
    ax2.legend()
    ax2.grid(True)

    plt.show()

if __name__ == '__main__':
    task_cpu_range = np.arange(1, 4, 0.25)
    repeat = params.repeat
    final_results = {key: [] for key in [
        'our_succ', 'our_reward', 'wocloud_succ', 'wocloud_reward', 'LF_succ', 'LF_reward',
        'Greedy1_succ', 'Greedy1_reward', 'Greedy2_succ', 'Greedy2_reward',
        'LFCloud_succ', 'LFCloud_reward', 'Greedy1Cloud_succ', 'Greedy1Cloud_reward',
        'Greedy2Cloud_succ', 'Greedy2Cloud_reward'
    ]}

    for task_cpu in task_cpu_range:
        avg_results = run_experiment(task_cpu, repeat)
        for key in final_results.keys():
            final_results[key].append(avg_results[key])

    plot(final_results, task_cpu_range)