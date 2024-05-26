import numpy as np
import parameters as param
import math
import torch
import torch.nn as nn
import parameters as params

class Env():
    def __init__(self):
        super(Env, self).__init__()
        
        # Define action and observation space
        self.numEdge = param.numEdge
        self.numVeh = param.numVeh
        
        self.taskInfo = np.zeros(3)
        self.mobilityInfo = np.zeros(4)
        self.taskEnd = np.zeros(self.numVeh * param.maxStep) # 처리 완료까지 남은 시간 기록
        self.alloRes_loc = np.zeros(self.numVeh * param.maxStep) # local에서 할당받은 자원량 기록
        self.alloRes_neighbor = np.zeros(self.numVeh * param.maxStep) # offload 서버에게 빌려온 자원량 기록
        self.allo_loc = np.zeros(self.numVeh * param.maxStep)  # local 서버 번호 기록
        self.allo_neighbor = np.zeros(self.numVeh * param.maxStep) # offload한 서버 번호 기록

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
        for i in range (self.numEdge):
            distance = self.calculate_distance(vehicle_position[0], vehicle_position[1], param.edge_pos[i][0], param.edge_pos[i][1])
            if min_distance > distance:
                min_distance = distance
                closest_rsu_index = i
        return closest_rsu_index
    
    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def reset(self, step):
        ''' 초기 자원 할당, 자원 release 구현 '''
        if step == -1: # episode 시작되고 첫 reset call
            param.remains = self.random_list(self.numEdge, param.resource_avg, param.resource_std)
        else:
            for i in range (step+1):
                if self.taskEnd[i] > 0: # 아직 리소스 사용 중, 실패하더라도 -1로 주지말고 max time 동안 자원 차지하고 있도록 해보자
                    self.taskEnd[i] = max(0, self.taskEnd[i]-1/self.numVeh)
                    if self.taskEnd[i] == 0:
                        param.remains[int(self.allo_loc[i])] += self.alloRes_loc[i]
                        param.remains[int(self.allo_neighbor[i])] +=self.alloRes_neighbor[i]

        ''' 작업 초기화 '''
        self.taskInfo[0] = np.random.uniform(param.min_size, param.max_size)
        self.taskInfo[1] = np.random.uniform(param.min_cpu, param.max_cpu)
        self.taskInfo[2] = np.random.uniform(param.min_time, param.max_time)
        param.task = self.taskInfo

        ''' 차량 모빌리티 모델 초기화 (24.05.22 기준으로 정해진 구역 내 완전히 random -> 추후에 more realistic한 mobility model로 업데이트 필요) '''
        self.mobilityInfo[0] = np.random.uniform(0, 4) #x-axis
        self.mobilityInfo[1] = np.random.uniform(0, 4) #x-axis#y-axis
        self.mobilityInfo[2] = np.random.uniform(0.0083, 0.02) #velocity
        self.mobilityInfo[3] = np.random.choice([0, 1, 2, 3]) #angle

        ''' 차량 가장 가까운 edge server 찾기 '''
        self.nearest = self.find_closest_rsu(self.mobilityInfo[:2])

        return self.taskInfo

    def step(self, action1, action2, stepnum):
        #action1: offloading fraction (0~1) from ppo (continous)
        #action2: offloading decision (0~N) from sac (discrete)

        # 1. offloading fraction에 따라 local computing량과 offloading computing량 계산
        tasksize = param.task[0]
        taskcpu = param.task[1]
        tasktime = param.task[2]

        local_amount = taskcpu * action1 
        off_amount = taskcpu * (1-action1)
        #print("fraction: ", action1, ", local: ", local_amount, "Gigacycle, offload: ", off_amount, "Gigacycle")

        # 2. local computing time 계산
        # 2.1 Tloc = computing time (우선 모든 자원 - remains 다 할당한다고 가정하고 코드 짜기) version 1
        Tloc = local_amount/param.remains[self.nearest]
        # 3. offloading time 계산
        # 3.1 transmission delay 계산 (hop count 가지고 init -> comp까지)
        
        hopcount = self.calculate_hopcount(param.edge_pos[self.nearest], param.edge_pos[int(action1)])
        bandwidth = np.random.uniform(0.07, 0.1)
        Ttrans = (tasksize*(1-action1))/(bandwidth*1000)*hopcount
        # 3.2 computing delay 계산
        Tcomp = off_amount/param.remains[int(action2)]
        # 3.3 Toff = trans +  comp
        Toff = Ttrans + Tcomp

        # 4. local과 off 중 더 긴 시간을 최종 Ttotal로 결정
        #print("Tloc: ", Tloc, "Toff: ", Toff)
        Ttotal = max(Tloc, Toff)

        # 5. 성공 여부 체크 
        if tasktime < Ttotal:
            #print("failue")
            reward = -1
            self.taskEnd[stepnum] = tasktime #실패한 경우 할당받은 자원 사용 시간 = latency 요구사항
        else:
            #print("success")
            reward = 0
            self.taskEnd[stepnum] = Ttotal

        self.allo_loc[stepnum] = self.nearest
        self.allo_neighbor[stepnum] = action2

        self.alloRes_loc[stepnum]= param.remains[self.nearest]
        self.alloRes_neighbor[stepnum]= param.remains[int(action2)]

        # 6. reward 계산
        profit = taskcpu*param.unitprice_cpu + tasksize*param.unitprice_size
        
        energy_coeff = 10 ** -26 # effective energy coefficient
        cost_comp1 = energy_coeff * param.remains[self.nearest] ** 2 * taskcpu*action1 #local에서 연산하려고 사용한 에너지 소모량
        cost_comp2 = energy_coeff * param.remains[int(action2)] ** 2* taskcpu*(1-action1) #neighbor에서 연산하려고 사용한 에너지 소모량
        cost_comp = param.wcomp*cost_comp1+cost_comp2
        cost_trans = param.wtrans*(1-action1)*tasksize*hopcount
        cost = cost_comp + cost_trans

        
        # reward = profit - cost #objective function of original problem
    
        #print(reward)
        # 7. 서버들 가용 자원량 조정 (위에서 할당한 만큼 빼는 것)
        param.remains[self.nearest] -= param.remains[self.nearest]
        param.remains[int(action2)] -= param.remains[int(action2)]

        done = False
        new_task = self.reset(stepnum)
        key = np.vstack((params.remains, params.hop_count, params.temp)) # 3x10 크기의 배열
        query = new_task # 1x3 크기
        new_query = query.reshape(1,3)

        key_tensor = torch.tensor(key, dtype=torch.float32)
        query_tensor = torch.tensor(new_query, dtype=torch.float32)
        #print(key_tensor.shape, query_tensor.shape)

        scores = torch.matmul(query_tensor, key_tensor)
        #print(scores.shape)

        attn_weights = nn.functional.softmax(scores, dim=-1)
        #task_tensor = torch.tensor(params.task, dtype=torch.float32)

        encoded_state = torch.cat((attn_weights, query_tensor), dim=1)
        new_state = np.concatenate((encoded_state.reshape(-1), action1.reshape(-1)), axis=-1)
        
        return new_state, reward, done

    def calculate_hopcount (self, mob1, mob2):
        diffx = abs(mob1[0] - mob2[0]) / param.radius*2
        diffy = abs(mob1[1] - mob2[1]) / param.radius*2
        hop = diffx+diffy
        return hop
        