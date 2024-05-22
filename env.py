import numpy as np
import parameters as param

class Env():
    def __init__(self):
        super(Env, self).__init__()
        
        # Define action and observation space
        self.numEdge = param.numEdge
        self.numVeh = param.numVeh
        
        self.taskInfo = np.zeros(3)
        self.mobilityInfo = np.zeros(4)
        self.releaseInfo = np.zeros(3, self.numVeh * param.maxStep) # line 0: remaining time, line 1: allocated resource for task i, line 2: computing server of task i

    def random_list(self, size, target_mean, target_std):
        random_values = []
        while len(random_values) < size:
            value = np.random.normal(loc=target_mean, scale=target_std)
            if value >= 0:
                random_values.append(value)
        return np.array(random_values)
    
    def find_closest_rsu(self, vehicle_position):
        min_distance = float("inf")
        closest_rsu_index = -1
        for i in range (self.num_edge_servers):
            distance = self.calculate_distance(vehicle_position[0], vehicle_position[1], param.edge_pos[i][0], param.edge_pos[i][1])
            if min_distance > distance:
                min_distance = distance
                closest_rsu_index = i
        return closest_rsu_index
    
    def reset(self, step):
        ''' 초기 자원 할당, 자원 release 구현 '''
        if step == -1: # episode 시작되고 첫 reset call
            param.remains = self.random_list(self.numEdge, param.resource_avg, param.resource_std)
        else:
            for i in range (step+1):
                if self.releaseInfo[0][i] > 0: # 아직 리소스 사용 중, 실패하더라도 -1로 주지말고 max time 동안 자원 차지하고 있도록 해보자
                    self.releaseInfo[0][i] = max(0, self.releaseInfo[0][i]-1/self.numVeh)
                    if self.releaseInfo[0][i] == 0:
                        param.remains[int(self.releaseInfo[2][i])] += self.releaseInfo[1][i]

        ''' 작업 초기화 '''
        self.taskInfo[0] = np.random.uniform(param.min_size, param.max_size)
        self.taskInfo[1] = np.random.uniform(param.min_cpu, param.max_cpu)
        self.taskInfo[2] = np.random.uniform(param.min_time, param.max_time)
        
        ''' 차량 모빌리티 모델 초기화 (24.05.22 기준으로 정해진 구역 내 완전히 random -> 추후에 more realistic한 mobility model로 업데이트 필요) '''
        self.mobilityInfo[0] = np.random.uniform(0, 4) #x-axis
        self.mobilityInfo[1] = np.random.uniform(0, 4) #x-axis#y-axis
        self.mobilityInfo[2] = np.random.uniform(0.0083, 0.02) #velocity
        self.mobilityInfo[3] = np.random.choice([0, 1, 2, 3]) #angle

        ''' 차량 가장 가까운 edge server 찾기 '''
        self.nearest = self.find_closest_rsu(self.mobilityInfo[:2])

        return self.taskInfo

    def step(self, action1, action2):
        #action1: offloading fraction (0~1) from ppo (continous)
        #action2: offloading decision (0~N) from sac (discrete)

        # 자원 할당 여기서! (credit based resource allocation like KARMA)
        # 문제 - 카르마처럼 하려면 user set이 있어야 하는데.. 문제는 현재 스킴은 한번에 task 1개만 결정됨 
        # 이거에 대한 논리적 구조를 설계해야 함
       
        
        return new_state, reward, done
