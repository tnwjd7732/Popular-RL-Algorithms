import env as environment
import parameters as params
import numpy as np
import matplotlib.pyplot as plt
import no_RL_scheme as schemes
import clustering
import my_ppo
import my_dqn

env = environment.Env()
clst = clustering.Clustering()
nearest = schemes.Nearest()
greedy1 = schemes.Greedy()
greedy2 = schemes.Greedy()
nearest_cloud = schemes.NearestCloud()
greedy1_cloud = schemes.GreedyCloud()
greedy2_cloud = schemes.GreedyCloud()

ppo_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo_woCloud = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_woCloud = my_dqn.DQN(env, params.action_dim2, params.state_dim2)

ppo_woCloud_ = my_ppo.PPO(params.state_dim1, params.action_dim1, hidden_dim=params.hidden_dim)
dqn_woCloud_ = my_dqn.DQN(env, params.action_dim2, params.state_dim2)
ppo_woCloud = ppo_woCloud_.load_model(params.woCloud_ppo_path)
dqn_woCloud = dqn_woCloud_.load_model(params.woCloud_dqn_path)

def run_experiment(task_size, repeat, env, ppo_, dqn_, ppo_wocloud_, dqn_wocloud_):
    params.min_size = task_size * 8  # Task size in bits
    params.max_size = task_size * 8
    results = {
        'our_succ': [], 'our_reward': [],
        'wocloud_succ': [], 'wocloud_reward': [],
        'nearest_succ': [], 'nearest_reward': [],
        'greedy1_succ': [], 'greedy1_reward': [],
        'greedy2_succ': [], 'greedy2_reward': [],
        'nearestCloud_succ': [], 'nearestCloud_reward': [],
        'greedy1Cloud_succ': [], 'greedy1Cloud_reward': [],
        'greedy2Cloud_succ': [], 'greedy2Cloud_reward': []
    }

    # Initialize schemes
    

    # 스킴 딕셔너리 정의
    schemes_dict = {
        'our': ('our_succ', 'our_reward', True, ppo_, dqn_, clst.form_cluster, None, 1),
        'wocloud': ('wocloud_succ', 'wocloud_reward', True, ppo_woCloud_, ppo_woCloud_, clst.form_cluster, None, 0),
        'nearest': ('near_succ', 'near_reward', False, nearest, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), None, 1),
        'nearest_cloud': ('cloud_near_succ', 'cloud_near_reward', False, nearest_cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), None, 1),
        'greedy_1': ('gree_succ', 'gree_reward', False, greedy1, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 1, 1),
        'greedy_2': ('gree2_succ', 'gree2_reward', False, greedy2, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 2, 1),
        'greedy_cloud_1': ('cloud_gree_succ', 'cloud_gree_reward', False, greedy1_cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 1, 1),
        'greedy_cloud_2': ('cloud_gree2_succ', 'cloud_gree_reward', False, greedy2_cloud, None, lambda: clst.random_list(params.numEdge, params.resource_avg, params.resource_std), 2, 1),
    }

    for _ in range(repeat):
        x = -1
        for i in range(params.numEdge):
            if i % params.grid_size == 0:
                x += 1
                y = 0
            params.edge_pos[i] = [0.5 + y, 0.5 + x]
            y += 1

        for key, (is_rl, succ_key, reward_key, ppo_model, dqn_model, setup, hop, cloud_setting) in schemes_dict.items():
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

def plot(results, task_size_range):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    font_size = params.font_size

    # Success rate plot
    ax1.plot(task_size_range, results['our_succ'], label="Proposed", linewidth=2)
    ax1.plot(task_size_range, results['wocloud_succ'], label="Without Cloud", linewidth=2)
    ax1.plot(task_size_range, results['nearest_succ'], label="Nearest", linewidth=2)
    ax1.plot(task_size_range, results['greedy1_succ'], label="Greedy(1-hop)", linewidth=2)
    ax1.plot(task_size_range, results['greedy2_succ'], label="Greedy(2-hop)", linewidth=2)
    ax1.plot(task_size_range, results['nearestCloud_succ'], label="Nearest Cloud", linewidth=2)
    ax1.plot(task_size_range, results['greedy1Cloud_succ'], label="Greedy Cloud(1-hop)", linewidth=2)
    ax1.plot(task_size_range, results['greedy2Cloud_succ'], label="Greedy Cloud(2-hop)", linewidth=2)
    ax1.set_xlabel('Task Size (MB)', fontsize=font_size + 10)
    ax1.set_ylabel('Success Rate', fontsize=font_size + 10)
    ax1.legend()
    ax1.grid(True)

    # Reward plot
    ax2.plot(task_size_range, results['our_reward'], label="Proposed", linewidth=2)
    ax2.plot(task_size_range, results['wocloud_reward'], label="Without Cloud", linewidth=2)
    ax2.plot(task_size_range, results['nearest_reward'], label="Nearest", linewidth=2)
    ax2.plot(task_size_range, results['greedy1_reward'], label="Greedy(1-hop)", linewidth=2)
    ax2.plot(task_size_range, results['greedy2_reward'], label="Greedy(2-hop)", linewidth=2)
    ax2.plot(task_size_range, results['nearestCloud_reward'], label="Nearest Cloud", linewidth=2)
    ax2.plot(task_size_range, results['greedy1Cloud_reward'], label="Greedy Cloud(1-hop)", linewidth=2)
    ax2.plot(task_size_range, results['greedy2Cloud_reward'], label="Greedy Cloud(2-hop)", linewidth=2)
    ax2.set_xlabel('Task Size (MB)', fontsize=font_size + 10)
    ax2.set_ylabel('Average Reward', fontsize=font_size + 10)
    ax2.legend()
    ax2.grid(True)

    plt.show()

if __name__ == '__main__':
    task_size_range = np.arange(0.5, 3.1, 0.25)
    repeat = params.repeat

    final_results = {
        'our_succ': [], 'our_reward': [],
        'wocloud_succ': [], 'wocloud_reward': [],
        'nearest_succ': [], 'nearest_reward': [],
        'greedy1_succ': [], 'greedy1_reward': [],
        'greedy2_succ': [], 'greedy2_reward': [],
        'nearestCloud_succ': [], 'nearestCloud_reward': [],
        'greedy1Cloud_succ': [], 'greedy1Cloud_reward': [],
        'greedy2Cloud_succ': [], 'greedy2Cloud_reward': []
    }

    for task_size in task_size_range:
        avg_results = run_experiment(task_size, repeat, env, ppo_, dqn_, ppo_woCloud_, dqn_woCloud_)
        for key in final_results.keys():
            final_results[key].append(avg_results[key])

    plot(final_results, task_size_range)
