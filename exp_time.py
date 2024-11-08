import env as environment
import parameters as params
import numpy as np
import matplotlib.pyplot as plt
import no_RL_scheme as schemes
import clustering
import my_ppo
import my_dqn

def run_experiment(task_time, repeat):
    params.min_time = task_time
    params.max_time = task_time

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
        'our': (ppo_, dqn_, clst.form_cluster, 'our_succ', 'our_reward', None, 1),
        'wocloud': (ppo_wocloud_, dqn_wocloud_, clst.form_cluster, 'wocloud_succ', 'wocloud_reward', None, 0),
        'LF': (None, None, LF, 'LF_succ', 'LF_reward', None, 1),
        'Greedy1': (None, None, greedy1, 'Greedy1_succ', 'Greedy1_reward', 1, 1),
        'Greedy2': (None, None, greedy2, 'Greedy2_succ', 'Greedy2_reward', 2, 1),
        'LFCloud': (None, None, LFCloud, 'LFCloud_succ', 'LFCloud_reward', None, 1),
        'Greedy1Cloud': (None, None, Greedy1Cloud, 'Greedy1Cloud_succ', 'Greedy1Cloud_reward', 1, 1),
        'Greedy2Cloud': (None, None, Greedy2Cloud, 'Greedy2Cloud_succ', 'Greedy2Cloud_reward', 2, 1)
    }

    for _ in range(repeat):
        # 각 엣지 서버 위치 초기화
        x = -1
        for i in range(params.numEdge):
            if i % params.grid_size == 0:
                x += 1
                y = 0
            params.edge_pos[i] = [0.5 + y, 0.5 + x]
            y += 1

        for key, (ppo_model, dqn_model, scheme, succ_key, reward_key, hop, cloud_setting) in schemes_dict.items():
            fail = 0
            episode_reward = 0
            params.cloud = cloud_setting  # 각 스킴에 맞게 클라우드 사용 설정
            
            if hasattr(scheme, '__call__'):
                scheme()  # 클러스터 형성 함수 호출
            
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
                        action1, action2 = scheme.choose_action(hop, step)
                    else:
                        action1, action2 = scheme.choose_action()
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

def plot(results, time_range):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    font_size = params.font_size

    # 성공률 플롯
    ax1.plot(time_range, results['our_succ'], label="Proposed", linewidth=2)
    ax1.plot(time_range, results['wocloud_succ'], label="Without Cloud", linewidth=2)
    ax1.plot(time_range, results['LF_succ'], label="Local First", linewidth=2)
    ax1.plot(time_range, results['LFCloud_succ'], label="Local First Cloud", linewidth=2)
    ax1.plot(time_range, results['Greedy1_succ'], label="Greedy(1-hop)", linewidth=2)
    ax1.plot(time_range, results['Greedy2_succ'], label="Greedy(2-hop)", linewidth=2)
    ax1.plot(time_range, results['Greedy1Cloud_succ'], label="Greedy Cloud(1-hop)", linewidth=2)
    ax1.plot(time_range, results['Greedy2Cloud_succ'], label="Greedy Cloud(2-hop)", linewidth=2)
    ax1.set_xlabel('Task Time', fontsize=font_size + 10)
    ax1.set_ylabel('Success Rate', fontsize=font_size + 10)
    ax1.legend()
    ax1.grid(True)

    # 리워드 플롯
    ax2.plot(time_range, results['our_reward'], label="Proposed", linewidth=2)
    ax2.plot(time_range, results['wocloud_reward'], label="Without Cloud", linewidth=2)
    ax2.plot(time_range, results['LF_reward'], label="Local First", linewidth=2)
    ax2.plot(time_range, results['LFCloud_reward'], label="Local First Cloud", linewidth=2)
    ax2.plot(time_range, results['Greedy1_reward'], label="Greedy(1-hop)", linewidth=2)
    ax2.plot(time_range, results['Greedy2_reward'], label="Greedy(2-hop)", linewidth=2)
    ax2.plot(time_range, results['Greedy1Cloud_reward'], label="Greedy Cloud(1-hop)", linewidth=2)
    ax2.plot(time_range, results['Greedy2Cloud_reward'], label="Greedy Cloud(2-hop)", linewidth=2)
    ax2.set_xlabel('Task Time', fontsize=font_size + 10)
    ax2.set_ylabel('Average Reward', fontsize=font_size + 10)
    ax2.legend()
    ax2.grid(True)

    plt.show()

if __name__ == '__main__':
    time_range = np.arange(1, 3.1, 0.25)
    repeat = params.repeat
    final_results = {key: [] for key in [
        'our_succ', 'our_reward', 'wocloud_succ', 'wocloud_reward', 'LF_succ', 'LF_reward',
        'Greedy1_succ', 'Greedy1_reward', 'Greedy2_succ', 'Greedy2_reward',
        'LFCloud_succ', 'LFCloud_reward', 'Greedy1Cloud_succ', 'Greedy1Cloud_reward',
        'Greedy2Cloud_succ', 'Greedy2Cloud_reward'
    ]}

    for task_time in time_range:
        avg_results = run_experiment(task_time, repeat)
        for key in final_results.keys():
            final_results[key].append(avg_results[key])

    plot(final_results, time_range)
