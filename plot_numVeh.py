import json
import matplotlib.pyplot as plt

# JSON 파일에서 데이터를 불러오기
with open("numVeh_result.json", "r") as f:
    results = json.load(f)

veh_range = range(250, 401, 50)

plt.figure(figsize=(16, 8))

# Success Ratio Plot
plt.subplot(1, 2, 1)
plt.plot(veh_range, results['our_succ'], label='Proposed', color='purple', linestyle='-', linewidth=2, marker='D')
plt.plot(veh_range, results['wocloud_succ'], label='Without Cloud', color='purple', linestyle='--', linewidth=2, marker='D')
plt.plot(veh_range, results['near_succ'], label='Nearest', color='blue', linestyle='--', linewidth=2, marker='o')
plt.plot(veh_range, results['gree_succ'], label='Greedy(1-hop)', color='green', linestyle='--', linewidth=2, marker='s')
plt.plot(veh_range, results['gree2_succ'], label='Greedy(2-hop)', color='red', linestyle='--', linewidth=2, marker='^')
plt.plot(veh_range, results['cloud_near_succ'], label='Cloud Nearest', color='blue', linestyle='-', linewidth=2, marker='o')
plt.plot(veh_range, results['cloud_gree_succ'], label='Cloud Greedy(1-hop)', color='green', linestyle='-', linewidth=2, marker='s')
plt.plot(veh_range, results['cloud_gree2_succ'], label='Cloud Greedy(2-hop)', color='red', linestyle='-', linewidth=2, marker='^')

plt.xlabel("Number of Vehicles")
plt.ylabel("Success Ratio")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
plt.grid(True, linestyle='--', linewidth=0.5)

# Average Reward Plot
plt.subplot(1, 2, 2)
plt.plot(veh_range, results['our_reward'], label='Proposed', color='purple', linestyle='-', linewidth=2, marker='D')
plt.plot(veh_range, results['wocloud_reward'], label='Without Cloud', color='purple', linestyle='--', linewidth=2, marker='D')
plt.plot(veh_range, results['near_reward'], label='Nearest', color='blue', linestyle='--', linewidth=2, marker='o')
plt.plot(veh_range, results['gree_reward'], label='Greedy(1-hop)', color='green', linestyle='--', linewidth=2, marker='s')
plt.plot(veh_range, results['gree2_reward'], label='Greedy(2-hop)', color='red', linestyle='--', linewidth=2, marker='^')
plt.plot(veh_range, results['cloud_near_reward'], label='Cloud Nearest', color='blue', linestyle='-', linewidth=2, marker='o')
plt.plot(veh_range, results['cloud_gree_reward'], label='Cloud Greedy(1-hop)', color='green', linestyle='-', linewidth=2, marker='s')
plt.plot(veh_range, results['cloud_gree2_reward'], label='Cloud Greedy(2-hop)', color='red', linestyle='-', linewidth=2, marker='^')

plt.xlabel("Number of Vehicles")
plt.ylabel("Average Reward")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
plt.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
