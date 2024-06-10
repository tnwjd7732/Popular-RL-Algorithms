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
        self.wrong_cnt = 0
        
        self.taskInfo = np.zeros(3)
        self.mobilityInfo = np.zeros(4)
        self.taskEnd = np.zeros(self.numVeh * param.STEP) # 처리 완료까지 남은 시간 기록
        self.alloRes_loc = np.zeros(self.numVeh * param.STEP) # local에서 할당받은 자원량 기록
        self.alloRes_neighbor = np.zeros(self.numVeh * param.STEP) # offload 서버에게 빌려온 자원량 기록
        self.allo_loc = np.zeros(self.numVeh * param.STEP)  # local 서버 번호 기록
        self.allo_neighbor = np.zeros(self.numVeh * param.STEP) # offload한 서버 번호 기록
    
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
        if step!=-1:
            for i in range (step+1):
                if self.taskEnd[i] > 0: # 아직 리소스 사용 중, 실패하더라도 -1로 주지말고 max time 동안 자원 차지하고 있도록 해보자
                    self.taskEnd[i] = max(0, self.taskEnd[i]-1/self.numVeh)
                    if self.taskEnd[i] == 0:
                        param.remains[int(self.allo_loc[i])] += self.alloRes_loc[i]
                        param.remains[int(self.allo_neighbor[i])] +=self.alloRes_neighbor[i]
        else:
            print("이전 에피소드에서의 wrong action 횟수: ", self.wrong_cnt)
            self.wrong_cnt = 0
        for i in range(params.numEdge):
            params.remains_lev[i] = int(params.remains[i]/10)
        #print("remains level: ", params.remains_lev)       

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
        self.calculate_hopcount2(param.edge_pos[int(self.nearest)])
        myClusterId = self.find_my_cluster(int(self.nearest))
        #print("My edge: ", self.nearest, "My CH: ", myClusterId)
        self.myCH = myClusterId

        if myClusterId is not None:
            cluster_servers = [myClusterId] + params.CMs[myClusterId]
            cluster_servers.sort()
            self.cluster = cluster_servers

            cluster_remains_lev = [params.remains_lev[i] for i in cluster_servers]
            cluster_size = len(cluster_servers)
            dummy_size = max(0, params.maxEdge - cluster_size)  # 더미값의 크기 결정
            dummy_values = np.zeros(dummy_size)
            state = np.concatenate((cluster_remains_lev, dummy_values, self.taskInfo))
        else:
            dummy_values = np.zeros(params.maxEdge)  # 클러스터에 속하지 않은 경우 더미값으로 채움
            state = np.concatenate((dummy_values, self.taskInfo))

        if myClusterId is not None:
            cluster_servers = [myClusterId] + params.CMs[myClusterId]
            cluster_servers.sort()

            cluster_remains_lev = [params.remains_lev[i] for i in cluster_servers]
            cluster_hop_count = [params.hop_count[i] for i in cluster_servers]  # 클러스터에 속한 서버들의 홉 카운트 정보
            cluster_size = len(cluster_servers)
            dummy_size = max(0, params.maxEdge - cluster_size)  # 더미값의 크기 결정
            dummy_values = np.zeros(dummy_size)
            state2 = np.concatenate((cluster_remains_lev, dummy_values, cluster_hop_count, dummy_values, params.task))
        else:
            dummy_values = np.zeros(params.maxEdge)  # 클러스터에 속하지 않은 경우 더미값으로 채움
            state2 = np.concatenate((dummy_values, params.hop_count, params.task))

        return state, state2


        
    def find_my_cluster(self, server_id):
        for ch_id in params.CHs:
            if server_id == ch_id or server_id in params.CMs[ch_id]:
                return ch_id
        return None
    
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
        optimal_resource_loc =local_amount/tasktime #state[5] = required CPU cycles, state[6] = time deadline
        optimal_resource_loc = min(param.remains[self.nearest], optimal_resource_loc*1.5)
        optimal_resource_off = off_amount/tasktime #state[5] = required CPU cycles, state[6] = time deadline
        #print("my CH: ", self.myCH, "action2: ", action2, "params.CMs[myCH]: ", self.cluster)
        
        if(len(self.cluster)) <= action2:
            r1 = 0
            r2 = -10
            reward = 0
            self.allo_loc[stepnum] = 0
            self.allo_neighbor[stepnum] = 0
            self.alloRes_loc[stepnum]=0
            self.alloRes_neighbor[stepnum]= 0
            self.taskEnd[stepnum] = 0
            print("wrong action2 ...")
            self.wrong_cnt+=1
            
        else:
            '''여기서 action2가 실제로 선택 불가한 (len(self.cluster)보다 큰 값일 경우 어케야할까?)'''
            action_globid_ver = self.cluster[int(action2)]
            #print("action glob: ", action_globid_ver)
            optimal_resource_off = min(param.remains[action_globid_ver], optimal_resource_off*1.5)

            # 2. local computing time 계산
            # 2.1 Tloc = computing time (우선 모든 자원 - remains 다 할당한다고 가정하고 코드 짜기) version 1
            Tloc = local_amount/optimal_resource_loc
            # 3. offloading time 계산
            # 3.1 transmission delay 계산 (hop count 가지고 init -> comp까지)
            
            hopcount = self.calculate_hopcount(param.edge_pos[self.nearest], param.edge_pos[action_globid_ver])
            bandwidth = np.random.uniform(0.07, 0.1)
            Ttrans = (tasksize*(1-action1))/(bandwidth*1000)*hopcount
            # 3.2 computing delay 계산
        

            Tcomp = off_amount/optimal_resource_off
            # 3.3 Toff = trans +  comp
            Toff = Ttrans + Tcomp

            # 4. local과 off 중 더 긴 시간을 최종 Ttotal로 결정
            #print("Tloc: ", Tloc, "Toff: ", Toff)
            Ttotal = max(Tloc, Toff)
            #print("  size: ", tasksize, "cpu: ", taskcpu, "latency: ", tasktime)
            # 5. 성공 여부 체크 
            if tasktime < Ttotal:
                print("\n----> Fraction: ", action1)
                print("----> MyId: ", self.nearest,"(lev:", param.remains_lev[self.nearest], "-", (1-action1), ") Offloading: ", action2, "(lev:", param.remains_lev[action_globid_ver], "-", action1, ")")
                print("Failue - Ttotal:" , Ttotal, "Tloc: ", Tloc, "Toff: ", Toff, "hopcnt: ", hopcount)
                reward = 0
                r1 = 0
                r2 = 0
                if math.isinf(Ttotal):
                    if math.isinf(Tloc):
                        r1 = -1
                    if math.isinf(Toff):
                        r2 = -1
                    if Tloc > tasktime:
                        r1 = -0.5
                    if Toff > tasktime:
                        r2 = -0.5
                self.taskEnd[stepnum] = tasktime #실패한 경우 할당받은 자원 사용 시간 = latency 요구사항
            else:
                print("Success - Ttotal:" , Ttotal, "Tloc: ", Tloc, "Toff: ", Toff)
                reward = 1
                r1 = 1 - min(abs(Tloc-Toff), 0.5)+(tasktime-Ttotal)
                #print("latency-total:", tasktime-Ttotal)
                r2 = 1
                self.taskEnd[stepnum] = Ttotal

            self.allo_loc[stepnum] = self.nearest
            self.allo_neighbor[stepnum] = action_globid_ver

            self.alloRes_loc[stepnum]=optimal_resource_loc
            self.alloRes_neighbor[stepnum]= optimal_resource_off

            # 6. reward 계산
            profit = taskcpu*param.unitprice_cpu + tasksize*param.unitprice_size
            
            energy_coeff = 10 ** -26 # effective energy coefficient
            cost_comp1 = energy_coeff * optimal_resource_loc ** 2 * taskcpu*action1 #local에서 연산하려고 사용한 에너지 소모량
            cost_comp2 = energy_coeff * optimal_resource_off ** 2* taskcpu*(1-action1) #neighbor에서 연산하려고 사용한 에너지 소모량
            cost_comp = param.wcomp*cost_comp1+cost_comp2
            cost_trans = param.wtrans*(1-action1)*tasksize*hopcount
            cost = cost_comp + cost_trans

            
            # reward = profit - cost #objective function of original problem
        
            '''action2가 out of range여도 기록해야 하는 것들'''

            # 7. 서버들 가용 자원량 조정 (위에서 할당한 만큼 빼는 것)
            param.remains[self.nearest] -= optimal_resource_loc
            param.remains[action_globid_ver] -= optimal_resource_off
        
        done = False
        new_state1, new_state2_temp = self.reset(stepnum)

       
        new_state2 = np.concatenate((new_state2_temp, action1))
        done = False
        return new_state1, new_state2, reward, r1, r2, done

    def calculate_hopcount (self, mob1, mob2):
        diffx = abs(mob1[0] - mob2[0]) / (param.radius*2)
        diffy = abs(mob1[1] - mob2[1]) / (param.radius*2)
        hop = diffx+diffy
        return hop
    
    def calculate_hopcount2 (self, mob1):
        for i in range(param.numEdge):
            diffx = abs(mob1[0] - param.edge_pos[i][0]) / (param.radius*2)
            diffy = abs(mob1[1] - param.edge_pos[i][1]) / (param.radius*2)
            hop = diffx+diffy
            param.hop_count[i] = hop 

            # hop_count task i가 생성된 nearest 서버로부터 클러스터 내 모든 엣지 서버와의 hop count를 기록하는 것 
            # ex) hop_count[1]의 의미: nearest 서버 <-> 서버 1과의 홉 카운트
        return hop
        