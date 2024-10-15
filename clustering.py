import parameters as params
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import random
class Clustering():

    def __init__(self):
        self.glob_avg = 0
        self.CH = []
        self.local_remains = np.zeros(params.numEdge)  # global remains 복사 용도
        self.grid_size = int(math.sqrt(params.numEdge))
        self.cluster_members = {ch_id: [] for ch_id in range(params.numEdge)}
        self.cluster_averages = np.zeros(params.numEdge)  # 클러스터 평균 자원량 저장용
    
    def form_cluster_woclst(self):
        self.CH = list(range(params.numEdge))  # 모든 서버를 클러스터 헤드로 설정
        self.cluster_members = {ch_id: [] for ch_id in self.CH}
        params.remains = self.random_list(params.numEdge, params.resource_avg, params.resource_std)

        for ch_id in self.CH:
            neighbors = self.get_neighbors(ch_id)
            self.cluster_members[ch_id] = neighbors  # 1-hop 이웃을 클러스터 멤버로 추가

        # 각각의 리스트를 오름차순으로 정렬
        params.CHs = sorted(self.CH)
        params.CMs = {ch_id: sorted(members) for ch_id, members in self.cluster_members.items()}
    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_farthest_servers(self, grid, num_clusters):
        """그리드에서 가장 먼 서버들 중 lamb개의 서버를 찾는 함수"""
        max_distance = 0
        farthest_pairs = []
        
        # 그리드에서 가장 거리가 먼 두 서버를 찾음
        for i in range(len(grid)):
            for j in range(i+1, len(grid)):
                distance = self.calculate_distance(grid[i], grid[j])
                if distance > max_distance:
                    max_distance = distance
                    farthest_pairs = [i, j]
        
        # 가장 먼 서버 두 개를 먼저 CH로 선택
        CHs = [farthest_pairs[0], farthest_pairs[1]]
        
        # 나머지 lamb-2 개의 CH를 선택 (거리가 먼 서버 위주로)
        while len(CHs) < num_clusters:
            max_distance = 0
            next_ch = None
            for i in range(len(grid)):
                if i not in CHs:
                    # 기존 CH들과의 최소 거리를 계산
                    min_distance_to_ch = min([self.calculate_distance(grid[i], grid[ch]) for ch in CHs])
                    if min_distance_to_ch > max_distance:
                        max_distance = min_distance_to_ch
                        next_ch = i
            CHs.append(next_ch)
        
        return CHs

    def form_static_cluster(self):
        params.remains = self.random_list(params.numEdge, params.resource_avg, params.resource_std)

        # params.numEdge는 서버의 수를 나타냄
        num_servers = params.numEdge
        num_clusters = params.lamb

        # 그리드의 한 변 길이
        grid_size = int(math.sqrt(num_servers))

        if grid_size * grid_size != num_servers:
            raise ValueError("numEdge 값이 완전한 정사각형이 아닙니다.")
        
        # 그리드를 구성 (서버의 좌표를 저장)
        grid = [(i // grid_size, i % grid_size) for i in range(num_servers)]

        # 클러스터 멤버 초기화
        self.cluster_members = {}

        # lamb에 따른 클러스터링 방식
        if num_clusters == 2:
            # 가로로 나누는 클러스터링
            mid = grid_size // 2
            upper_cluster = [i for i in range(num_servers) if grid[i][0] < mid]  # 위쪽 클러스터
            lower_cluster = [i for i in range(num_servers) if grid[i][0] >= mid]  # 아래쪽 클러스터
            
            # 클러스터 헤드 설정 (각 클러스터의 첫 번째 서버)
            CH = [upper_cluster[0], lower_cluster[0]]
            self.CH = CH

            # 클러스터 멤버 구성
            self.cluster_members[CH[0]] = upper_cluster
            self.cluster_members[CH[1]] = lower_cluster

        elif num_clusters == 4:
            # +형태로 나누는 클러스터링
            mid_x = grid_size // 2
            mid_y = grid_size // 2

            cluster_1 = [i for i in range(num_servers) if grid[i][0] < mid_x and grid[i][1] < mid_y]  # 왼쪽 위
            cluster_2 = [i for i in range(num_servers) if grid[i][0] < mid_x and grid[i][1] >= mid_y]  # 오른쪽 위
            cluster_3 = [i for i in range(num_servers) if grid[i][0] >= mid_x and grid[i][1] < mid_y]  # 왼쪽 아래
            cluster_4 = [i for i in range(num_servers) if grid[i][0] >= mid_x and grid[i][1] >= mid_y]  # 오른쪽 아래
            
            # 클러스터 헤드 설정 (각 클러스터의 첫 번째 서버)
            CH = [cluster_1[0], cluster_2[0], cluster_3[0], cluster_4[0]]
            self.CH = CH

            # 클러스터 멤버 구성
            self.cluster_members[CH[0]] = cluster_1
            self.cluster_members[CH[1]] = cluster_2
            self.cluster_members[CH[2]] = cluster_3
            self.cluster_members[CH[3]] = cluster_4

        else:
            raise ValueError("지원되지 않는 lamb 값입니다. lamb는 2 또는 4이어야 합니다.")
        
        # 각각의 리스트를 오름차순으로 정렬
        params.CHs = sorted(self.CH)
        params.CMs = {ch_id: sorted(members) for ch_id, members in self.cluster_members.items()}

    def form_cluster(self):
        # 초기화
        self.CH = []
        self.cluster_members = {ch_id: [] for ch_id in range(params.numEdge)}
        params.remains = self.random_list(params.numEdge, params.resource_avg, params.resource_std)
        self.glob_avg = self.calc_globavg()
        self.local_remains = np.copy(params.remains)

        CH_resources, CH_ids = self.top_k_elements(params.lamb)
        self.CH = CH_ids.tolist()

        unassigned_servers = set(range(params.numEdge)) - set(self.CH)
        loop_count = 0  # 무한 루프 방지를 위한 카운터

        while unassigned_servers:
            loop_count += 1
            for ch_id in self.CH:
                if not unassigned_servers:
                    break
                if len(self.cluster_members[ch_id]) >= (params.maxEdge - 1):
                    continue  # 이미 maxEdge에 도달한 경우 건너뜀

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
            self.CH.sort(key=lambda ch_id: self.calc_cluster_avg(ch_id), reverse=True)  # 클러스터 내 평균 자원이 많은 순서대로 정렬
            if loop_count > 1000:  # 무한 루프 방지를 위한 임의의 큰 숫자
                break

        # 1-hop 연결을 유지하며 미배정 서버 처리
        if unassigned_servers:
            for server in list(unassigned_servers):
                # 1-hop 이내에 있는 클러스터 헤드 찾기
                neighbor_chs = [ch_id for ch_id in self.CH if server in self.get_neighbors(ch_id)]
                if neighbor_chs:
                    best_ch_id = None
                    best_diff = float('inf')
                    for ch_id in neighbor_chs:
                        if len(self.cluster_members[ch_id]) < (params.maxEdge - 1):
                            diff = abs(self.calc_cluster_avg(ch_id) - params.remains[server])
                            if diff < best_diff:
                                best_diff = diff
                                best_ch_id = ch_id

                    if best_ch_id is not None:
                        self.cluster_members[best_ch_id].append(server)
                        unassigned_servers.remove(server)
                    else:
                        # 적절한 클러스터를 찾지 못한 경우 새로운 클러스터 헤드를 생성
                        new_ch_id = server
                        self.CH.append(new_ch_id)
                        self.cluster_members[new_ch_id] = []
                else:
                    # 1-hop 이내에 클러스터 헤드가 없는 경우 새로운 클러스터 헤드를 생성
                    new_ch_id = server
                    self.CH.append(new_ch_id)
                    self.cluster_members[new_ch_id] = []

        params.CHs = sorted(self.CH)
        params.CMs = {ch_id: sorted(members) for ch_id, members in self.cluster_members.items()}


    def get_neighbors(self, nodeId):
            """ Get n-hop neighbors of the given nodeId """
            node_x, node_y = params.edge_pos[nodeId]
            neighbors = []
            for i in range(params.numEdge):
                if i == nodeId:
                    continue
                x, y = params.edge_pos[i]
                distance = np.sqrt((node_x - x)**2 + (node_y - y)**2)
                if distance <= 1 * (params.radius*2):
                    neighbors.append(i)
            #print(neighbors)
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
        return abs(row1 - row2) + abs(col1 - col2) <= 1

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
        #print("cluster visualize")
        
        colors = cm.rainbow(np.linspace(0, 1, len(self.CH)))
        color_map = {ch_id: color for ch_id, color in zip(self.CH, colors)}

        plt.figure(figsize=(4, 4))
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
        #print("EOV")
        

    def random_list(self, size, target_mean, target_std):
        if params.distribution_mode == 0:
            random_values = []
            target_this_turn = random.randint(0, target_std) # 0~target_std 중 매번 새롭게 선택 
            while len(random_values) < size:
                value = np.random.normal(loc=target_mean, scale=target_this_turn)
                if value >= 0:
                    random_values.append(value)
            return np.array(random_values)
        
        elif params.distribution_mode == 1: #region별로 분포가 다른 경우
            random_values = []
            while len(random_values) < int(size/3):
                value = np.random.normal(loc=target_mean*3, scale=target_std)
                if value >= 0:
                    random_values.append(value)
            while len(random_values) < size:
                value = np.random.normal(loc=target_mean*0.1, scale=1)
                if value >= 0:
                    random_values.append(value)
            return np.array(random_values)
