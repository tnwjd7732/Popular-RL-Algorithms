import parameters as params
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
class Clustering():
    def __init__(self):
        self.glob_avg = 0
        self.CH = []
        self.local_remains = np.zeros(params.numEdge)  # global remains 복사 용도
        self.grid_size = int(math.sqrt(params.numEdge))
        self.cluster_members = {ch_id: [] for ch_id in range(params.numEdge)}
        self.cluster_averages = np.zeros(params.numEdge)  # 클러스터 평균 자원량 저장용

    def form_cluster(self):
        self.CH = []
        self.cluster_members = {ch_id: [] for ch_id in range(params.numEdge)}
        params.remains = self.random_list(params.numEdge, params.resource_avg, params.resource_std)
        self.glob_avg = self.calc_globavg()
        #print("Global average: ", self.glob_avg)
        self.local_remains = np.copy(params.remains)

        CH_resources, CH_ids = self.top_k_elements(params.lamb)
        self.CH = CH_ids.tolist()

        unassigned_servers = set(range(params.numEdge)) - set(self.CH)
        while unassigned_servers:
            for ch_id in self.CH:
                if not unassigned_servers:
                    break
                
                all_members = [ch_id] + self.cluster_members[ch_id]
                neighbors = set()
                for member in all_members:
                    neighbors.update(self.get_neighbors(member))
                
                eligible_neighbors = [n for n in neighbors if n in unassigned_servers]
                
                if eligible_neighbors:
                    selected_neighbor = self.select_best_neighbor(all_members, eligible_neighbors)
                    self.cluster_members[ch_id].append(selected_neighbor)
                    unassigned_servers.remove(selected_neighbor)
            
            self.update_cluster_averages()
            
            # 우선 순위를 glob_avg에서 가장 먼 CH로 정렬
            self.CH.sort(key=lambda ch_id: abs(self.glob_avg - self.calc_cluster_avg(ch_id)), reverse=True)
        
        print(params.remains)
        print("Cluster Heads:", self.CH)
        for ch_id in self.CH:
            print(f"Cluster Head {ch_id}: Members {self.cluster_members[ch_id]}")
        
        # 각각의 리스트를 오름차순으로 정렬
        params.CHs = sorted(self.CH)
        params.CMs = {ch_id: sorted(members) for ch_id, members in self.cluster_members.items()}


    def get_neighbors(self, index):
        row, col = divmod(index, self.grid_size)
        neighbors = []
        
        # 상하좌우의 이웃을 추가
        if row > 0:
            neighbors.append((row - 1) * self.grid_size + col)  # 위
        if row < self.grid_size - 1:
            neighbors.append((row + 1) * self.grid_size + col)  # 아래
        if col > 0:
            neighbors.append(row * self.grid_size + col - 1)  # 왼쪽
        if col < self.grid_size - 1:
            neighbors.append(row * self.grid_size + col + 1)  # 오른쪽
        
        return neighbors

    def select_best_neighbor(self, members, neighbors):
        best_neighbor = None
        min_diff = float('inf')
        
        for neighbor in neighbors:
            total_resources = sum(self.local_remains[member] for member in members) + self.local_remains[neighbor]
            potential_avg = total_resources / (len(members) + 1)
            diff = abs(self.glob_avg - potential_avg)
            
            if diff < min_diff:
                min_diff = diff
                best_neighbor = neighbor
        
        return best_neighbor

    def update_cluster_averages(self):
        for ch_id in self.CH:
            if self.cluster_members[ch_id]:
                self.cluster_averages[ch_id] = self.calc_cluster_avg(ch_id)

    def calc_cluster_avg(self, ch_id):
        total_resources = self.local_remains[ch_id] + sum(self.local_remains[member] for member in self.cluster_members[ch_id])
        return total_resources / (len(self.cluster_members[ch_id]) + 1)

    def is_adjacent(self, index1, index2):
        row1, col1 = divmod(index1, self.grid_size)
        row2, col2 = divmod(index2, self.grid_size)
        return abs(row1 - row2) <=1 and abs(col1 - col2) <= 1

    def top_k_elements(self, k):
        sorted_indices = np.argsort(self.local_remains)
        sorted_indices = sorted_indices[::-1]  # 내림차순으로 정렬
        sorted_values = self.local_remains[sorted_indices]

        selected_indices = []
        selected_values = []

        i = 0
        while len(selected_indices) < k and i < len(sorted_indices):
            index = sorted_indices[i]
            if not any(self.is_adjacent(index, selected_index) for selected_index in selected_indices):
                selected_indices.append(index)
                selected_values.append(self.local_remains[index])
            i += 1

        return np.array(selected_values), np.array(selected_indices)

    def calc_globavg(self):
        return np.mean(params.remains)
    
    def visualize_clusters(self):
        colors = cm.rainbow(np.linspace(0, 1, len(self.CH)))
        color_map = {ch_id: color for ch_id, color in zip(self.CH, colors)}

        plt.figure(figsize=(8, 8))
        for ch_id in self.CH:
            ch_pos = divmod(ch_id, self.grid_size)
            ch_avg = self.cluster_averages[ch_id]
            plt.scatter(ch_pos[1] + 0.5, ch_pos[0] + 0.5, color=color_map[ch_id], edgecolor='black', s=200, marker='s')
            plt.text(ch_pos[1] + 0.5, ch_pos[0] + 0.7, f"{params.remains[ch_id]:.1f}\n(Avg: {ch_avg:.1f})",
                     fontsize=9, ha='center', va='center', color='black')
            
            for member in self.cluster_members[ch_id]:
                member_pos = divmod(member, self.grid_size)
                plt.scatter(member_pos[1] + 0.5, member_pos[0] + 0.5, color=color_map[ch_id], s=100)
                plt.text(member_pos[1] + 0.5, member_pos[0] + 0.7, f"{params.remains[member]:.1f}",
                         fontsize=9, ha='center', va='center', color='black')
                
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.gca().invert_yaxis()
        plt.xticks(np.arange(0, self.grid_size + 1, 1))
        plt.yticks(np.arange(0, self.grid_size + 1, 1))
        plt.grid(True)
        plt.title(f'Cluster Visualization (Global Avg: {self.glob_avg:.1f})')
        plt.show()
        #sys.exit()

    def random_list(self, size, target_mean, target_std):
        random_values = []
        while len(random_values) < size:
            value = np.random.normal(loc=target_mean, scale=target_std)
            if value >= 0:
                random_values.append(value)
        return np.array(random_values)
