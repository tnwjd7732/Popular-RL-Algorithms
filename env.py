import numpy as np
import parameters as param
import math
import torch
import torch.nn as nn
import parameters as params
import random

class Env():
    def __init__(self):
        super(Env, self).__init__() 
        
        # Define action and observation space
        self.loc_res = 0
        self.stepNum = -1
        self.off_res = 0
        self.action_frac = 0
        self.numEdge = param.numEdge
        self.numVeh = param.numVeh
        self.wrong_cnt = 0
        self.cloud_selection = 0
        self.taskInfo = np.zeros(3)
        self.mobilityInfo = np.zeros(4)
        self.taskEnd = np.zeros(self.numVeh * param.STEP) # 처리 완료까지 남은 시간 기록
        self.alloRes_loc = np.zeros(self.numVeh * param.STEP) # local에서 할당받은 자원량 기록
        self.alloRes_neighbor = np.zeros(self.numVeh * param.STEP) # offload 서버에게 빌려온 자원량 기록
        self.allo_loc = np.zeros(self.numVeh * param.STEP)  # local 서버 번호 기록
        self.allo_neighbor = np.zeros(self.numVeh * param.STEP) # offload한 서버 번호 기록
        self.plan_info = np.zeros(self.numVeh) # 각 차량들의 플랜 정보를 담고 있음 - 0: basic, 1: premium
        self.credit_info = np.zeros(self.numVeh) # 각 차랭들의 현재 크레딧 정보를 담고있음
    
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


    def reset(self, step, cloud): # if cloud==0: does not use, if cloud ==1: use cloud
        
        #print("Credit information ---- ")
        #print("userplan:", params.userplan, "and!!!", self.credit_info, "and!!!", params.credit)
        #print("remains: ", param.remains_lev)
        ''' 초기 자원 할당, 자원 release 구현 '''
       
        if step!=-1:
            for i in range (step+1):
                if self.taskEnd[i] > 0: # 아직 리소스 사용 중, 실패하더라도 -1로 주지말고 max time 동안 자원 차지하고 있도록 해보자
                    self.taskEnd[i] = max(0, self.taskEnd[i]-param.time_slot/self.numVeh)
                    if self.taskEnd[i] == 0:
                        param.remains[int(self.allo_loc[i])] += self.alloRes_loc[i]
                        param.remains[int(self.allo_neighbor[i])] +=self.alloRes_neighbor[i]
        else:
                # Define action and observation space
            self.loc_res = 0
            self.stepNum = -1
            self.off_res = 0
            self.action_frac = 0
            self.numEdge = param.numEdge
            self.numVeh = param.numVeh
            self.taskInfo = np.zeros(3)
            self.mobilityInfo = np.zeros(4)
            self.taskEnd = np.zeros(self.numVeh * param.STEP) # 처리 완료까지 남은 시간 기록
            self.alloRes_loc = np.zeros(self.numVeh * param.STEP) # local에서 할당받은 자원량 기록
            self.alloRes_neighbor = np.zeros(self.numVeh * param.STEP) # offload 서버에게 빌려온 자원량 기록
            self.allo_loc = np.zeros(self.numVeh * param.STEP)  # local 서버 번호 기록
            self.allo_neighbor = np.zeros(self.numVeh * param.STEP) # offload한 서버 번호 기록
            self.plan_info = np.zeros(self.numVeh) # 각 차량들의 플랜 정보를 담고 있음 - 0: basic, 1: premium
            self.credit_info = np.zeros(self.numVeh) # 각 차랭들의 현재 크레딧 정보를 담고있음
            params.premium_alloc = []  # 프리미엄 유저 자원 할당 기록
            params.basic_alloc = []  # 베이직 유저 자원 할당 기록
            #print("이전 에피소드에서의 wrong action 횟수: ", self.wrong_cnt)

            param.wrong_cnt.append(self.wrong_cnt)
            param.cloud_cnt.append(self.cloud_selection)

            #param.wrong_cnt.clear()
            #param.cloud_cnt.clear()
            

            self.wrong_cnt = 0
            self.cloud_selection = 0
            for i in range(param.numVeh):
                if params.userplan == 1:  # 크레딧 기반이면 0또는 1로 랜덤하게 설정 (50%)
                    self.plan_info[i] = random.randint(0, 1)
                    if self.plan_info[i] == 0:
                        self.credit_info[i] = param.basic_init
                    else:
                        self.credit_info[i] = param.premium_init
                else:  # 비교 스킴 (고정 1.6배)
                    if params.credit == 1:
                        self.plan_info[i] = 0  # 모든 유저가 동일한 자원 할당 방식
                        self.credit_info[i] = 1.35  # 모든 사용자에게 1.35배 자원 할당
                    elif params.credit == 2:
                        self.plan_info[i] = 1  # 모든 유저가 동일한 자원 할당 방식
                        self.credit_info[i] = 1.75  # 모든 사용자에게 1.75배 자원 할당
            param.credit_info = self.credit_info
                        
        for i in range(params.numEdge):
            params.remains_lev[i] = int(params.remains[i]/1)
        #print("remains level: ", params.remains_lev)       

        ''' 작업 초기화 '''
        #print(params.min_size, params.min_cpu, params.min_time, param.max_size, param.max_cpu, param.max_time)
        self.taskInfo[0] = np.random.uniform(param.min_size, param.max_size)
        self.taskInfo[1] = np.random.uniform(param.min_cpu, param.max_cpu)
        self.taskInfo[2] = np.random.uniform(param.min_time, param.max_time)
        param.task = self.taskInfo

        ''' 차량 모빌리티 모델 초기화 (24.05.22 기준으로 정해진 구역 내 완전히 random -> 추후에 more realistic한 mobility model로 업데이트 필요) '''
        self.mobilityInfo[0] = np.random.uniform(0, math.sqrt(param.numEdge)) #x-axis
        self.mobilityInfo[1] = np.random.uniform(0, math.sqrt(param.numEdge)) #x-axis#y-axis
        self.mobilityInfo[2] = np.random.uniform(0.0083, 0.02) #velocity
        self.mobilityInfo[3] = np.random.choice([0, 1, 2, 3]) #angle

        ''' 차량 가장 가까운 edge server 찾기 '''
        self.nearest = self.find_closest_rsu(self.mobilityInfo[:2])
        param.nearest = self.nearest
        self.calculate_hopcount2(param.edge_pos[int(self.nearest)])
        myClusterId = self.find_my_cluster(int(self.nearest))
        #print("My edge: ", self.nearest, "My CH: ", myClusterId)
        self.myCH = myClusterId

        
        cloud_resource = [100]
        cloud_hop = [100]

        if myClusterId is not None:
            if cloud == 0:
                myResource = np.array([param.remains[self.nearest]]) #클라우드에서는 리소스 넣을 때 *10
                myResource *= 10
            else:
                myResource = np.array([param.remains[self.nearest]])  

            sum = 0 

            cluster_servers = [myClusterId] + params.CMs[myClusterId]
            cluster_servers.sort()
            self.cluster = cluster_servers
            for server in cluster_servers:
                sum += param.remains_lev[int(server)]
            avgResource = np.array([sum/len(cluster_servers)])
            if cloud == 0:
                avgResource *= 10
            taskstate = np.zeros(3)
            taskstate = [self.taskInfo[i]*2 for i in range(3)]

            state = np.concatenate((myResource, avgResource, taskstate))

            
        else:
            dummy_values = np.full(2, -10)  # 클러스터에 속하지 않은 경우 더미값으로 채움
            state = np.concatenate((dummy_values, taskstate))

        if myClusterId is not None:
            cluster_servers = [myClusterId] + params.CMs[myClusterId]
            cluster_servers.sort()
            if cloud == 0:
                cluster_remains_lev = [params.remains_lev[i]*10 for i in cluster_servers]
            else:
                cluster_remains_lev = [params.remains_lev[i] for i in cluster_servers]
            if cloud == 0:
                cluster_hop_count = [params.hop_count[i] for i in cluster_servers]  # 클러스터에 속한 서버들의 홉 카운트 정보
            else:
                cluster_hop_count = [params.hop_count[i] for i in cluster_servers]  # 클러스터에 속한 서버들의 홉 카운트 정보
    
            cluster_size = len(cluster_servers)
            dummy_size = max(0, params.maxEdge - cluster_size)  # 더미값의 크기 결정
            dummy_values = np.full(dummy_size, -10)  # 더미값을 -10으로 설정
            if cloud == 1:
                state2 = np.concatenate((cloud_resource, cluster_remains_lev, dummy_values,cloud_hop, cluster_hop_count, dummy_values, taskstate))
            else:
                state2 = np.concatenate((cluster_remains_lev, dummy_values, cluster_hop_count, dummy_values, taskstate))

        else:
            if cloud == 1:
                dummy_values = np.full(params.maxEdge+1, -10)  # 클러스터에 속하지 않은 경우 더미값으로 채움
                state2 = np.concatenate((dummy_values, dummy_values, taskstate))
            else:
                dummy_values = np.full(params.maxEdge, -10)  # 클러스터에 속하지 않은 경우 더미값으로 채움
                state2 = np.concatenate((dummy_values, dummy_values, taskstate))

        return state, state2
   
    def find_my_cluster(self, server_id):
        # 먼저 서버가 CH인지 확인하고, CH라면 바로 리턴
        if server_id in params.CHs:
            return server_id
        
        # 서버가 CH가 아닌 경우, CM으로 속한 클러스터를 확인
        for ch_id in params.CHs:
            if server_id in params.CMs[ch_id]:
                return ch_id
        return None


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

    def step(self, action1, action2, stepnum, cloud):
        def calculate_rewards_and_costs(Tloc, Toff, vehId, taskcpu, tasksize, tasktime, optimal_resource_loc, optimal_resource_off, action1, hopcount):
            Ttotal = max(Tloc, Toff)
            reward = 0
            r1 = 0
            r2 = 0

            if tasktime < Ttotal:  # 실패
                reward = 0
                if params.userplan == 1: #크레딧 기반에서만 바꾸기
                    self.credit_info[vehId] = min(self.credit_info[vehId] + 0.1, param.premium_max if self.plan_info[vehId] else param.basic_max)
                
                # Determine the cause of failure and assign penalties
                if Tloc > tasktime and Toff > tasktime:
                    # Both actions are responsible
                    r1 = r2 = -3
                elif Tloc > tasktime:
                    # action1 is responsible
                    if math.isinf(Tloc):
                        r1 = -3 #할 수 없으면서 지가 한다고 fraction 크게 잡은거니까 model1은 확실하게 잘못한 것임
                    else:
                        r1 = min(max(-3, Tloc-tasktime), -1)
                    r2 = 1
                elif Toff > tasktime:
                    if param.remains[self.nearest] >= optimal_resource_off+optimal_resource_loc:
                        r1 = -1.5 #자기가 하면 됐는데도 보낸거니까 잘못한거임
                    else:
                        r1 = 1
                    r2 = -3
                        
      
                self.taskEnd[stepnum] = tasktime
            else:  # 성공
                if params.userplan == 1:
                    self.credit_info[vehId] = max(self.credit_info[vehId] - 0.1, param.premium_min if self.plan_info[vehId] else param.basic_min)
                cost_weight = 1.5 if self.plan_info[vehId] else 1.0
                profit = taskcpu * param.unitprice_cpu * cost_weight + tasksize * param.unitprice_size * cost_weight
                energy_coeff = 10 ** -26
                cost_comp1 = energy_coeff * optimal_resource_loc ** 2 * taskcpu * action1
                cost_comp2 = energy_coeff * optimal_resource_off ** 2 * taskcpu * (1 - action1)
                cost_comp = param.wcomp * (cost_comp1 + cost_comp2)
                cost_trans = param.cloud_trans_price if action2 == 0 else param.wtrans * (1 - action1) * tasksize * hopcount
                cost = cost_comp + cost_trans
                reward = profit - cost[0] + tasktime - Ttotal
                
                # Adjust reward based on revenue
                reward = max(1, reward / 2)  # Scale down the reward to make it manageable
                r1 = r2 = min(reward, 5)

                
                self.taskEnd[stepnum] = Ttotal

            return reward, r1, r2, Ttotal

        if cloud == 0:
            action2+=1
        
        # 기존 step 함수
        vehId = stepnum % param.numVeh
        current_credit = self.credit_info[vehId]
        tasksize, taskcpu, tasktime = param.task

        # Calculate local and offloading resources
        local_amount = taskcpu * action1
        off_amount = taskcpu * (1 - action1)
        optimal_resource_loc = local_amount / tasktime if local_amount else 0
        optimal_resource_off = off_amount / tasktime if off_amount else 0
        optimal_resource_loc = min(param.remains[self.nearest], optimal_resource_loc * current_credit)

        #print(f"--> fraction: {action1}, offloading decision: {action2}")
        #print(f"local edge: {self.nearest} (resource lev: {param.remains_lev[self.nearest]} - {action1}%)")
        
        if len(self.cluster) < action2 and action1 != 1:
            self.wrong_cnt += 1
            Tloc = local_amount / optimal_resource_loc if action1 else 0
            if Tloc < tasktime:
                r1, r2, reward, Ttotal = 2, -5, 0, 0
            else:
                r1, r2, reward, Ttotal = -2, -5, 0, 0
            self.allo_loc[stepnum] = self.allo_neighbor[stepnum] = -1
            self.alloRes_loc[stepnum] = self.alloRes_neighbor[stepnum] = 0
            self.taskEnd[stepnum] = -1
            Tloc, Toff = 0, 0
        else:
            if action2 == 0:
                Ttrans = 0.8
                Tloc = local_amount / optimal_resource_loc if action1 else 0
                Toff = Ttrans if action1 != 1 else 0
                reward, r1, r2, Ttotal = calculate_rewards_and_costs(Tloc, Toff, vehId, taskcpu, tasksize, tasktime, optimal_resource_loc, optimal_resource_off, action1, 0)
                self.allo_loc[stepnum] = self.nearest
                self.allo_neighbor[stepnum] = -1
                self.alloRes_loc[stepnum] = optimal_resource_loc if Tloc else 0
                self.alloRes_neighbor[stepnum] = 0
                param.remains[self.nearest] -= optimal_resource_loc if Tloc else 0
                self.cloud_selection += 1
                #print(f"offloading: cloud {1 - action1}%)")
            else:
                if action2 > len(self.cluster):
                    action_globid_ver = 0
                    optimal_resource_off = 0
                else:
                    #print("hihi ,", self.cluster[int(action2) - 1], int(action2) - 1)
                    action_globid_ver = self.cluster[int(action2) - 1]
                    #print("action glob:", action_globid_ver)
                    optimal_resource_off = min(param.remains[action_globid_ver], optimal_resource_off * current_credit)

                hopcount = self.calculate_hopcount(param.edge_pos[self.nearest], param.edge_pos[action_globid_ver])
                param.hop_counts.append(hopcount)

                bandwidth = np.random.uniform(0.07, 0.1)
                Ttrans = (tasksize * (1 - action1)) / (bandwidth * 1000) * hopcount
                Tcomp = off_amount / optimal_resource_off if action1 != 1 else 0
                Tloc = local_amount / optimal_resource_loc if action1 else 0
                Toff = Ttrans + Tcomp
                reward, r1, r2, Ttotal = calculate_rewards_and_costs(Tloc, Toff, vehId, taskcpu, tasksize, tasktime, optimal_resource_loc, optimal_resource_off, action1, hopcount)
                self.allo_loc[stepnum] = self.nearest
                self.allo_neighbor[stepnum] = action_globid_ver
                self.alloRes_loc[stepnum] = optimal_resource_loc
                self.alloRes_neighbor[stepnum] = optimal_resource_off
                param.remains[self.nearest] -= optimal_resource_loc
                param.remains[action_globid_ver] -= optimal_resource_off
                #rint(f"offloading: {action2} (resource lev: {param.remains_lev[action_globid_ver]} - {1 - action1}%)")
        #print(f"reward: {reward}, r1: {r1}, r2: {r2}")
        #print(f"req. latency: {tasktime}, Ttotal: {Ttotal}, Tloc: {Tloc}, Toff: {Toff}\n")

        new_state1, new_state2_temp = self.reset(stepnum, cloud)
        new_state2 = np.concatenate((new_state2_temp, action1))

        # Ensure reward is a float
        reward = float(reward)
        r1 = float(r1)
        r2 = float(r2)

        #print(param.remains)
        return new_state1, new_state2, reward, r1, r2, False

    def step_single(self, action1, action2, stepnum, cloud):

        def calculate_rewards_and_costs(Tloc, Toff, vehId, taskcpu, tasksize, tasktime, optimal_resource_loc, optimal_resource_off, action1, hopcount):
            Ttotal = max(Tloc, Toff)
            reward = 0
            r1 = 0
            if tasktime < Ttotal:  # 실패
                if params.userplan == 1:
                    self.credit_info[vehId] = min(self.credit_info[vehId] + 0.1, param.premium_max if self.plan_info[vehId] else param.basic_max)

                # 실패에 대한 책임을 r1에게 부여
                if Tloc > tasktime and Toff > tasktime:
                    # 둘 다 책임이 있을 때
                    r1 = -4
                elif Tloc > tasktime:
                    # action1이 주로 잘못했을 때
                    if math.isinf(Tloc):
                        r1 = -3  # 할 수 없으면서 큰 fraction을 선택한 경우
                    else:
                        r1 = -3 #min(max(-3, Tloc-tasktime), -1)
                elif Toff > tasktime:
                    # action2가 주로 잘못했을 때도 이제 r1에게 부여
                    if param.remains[self.nearest] >= optimal_resource_off:
                        r1 = -3  # action1이 자기가 처리할 수 있었는데도 오프로드한 경우
                    else:
                        r1 = -3 if tasktime > 1 else -3

                self.taskEnd[stepnum] = tasktime
            else:  # 성공
                if params.userplan == 1:
                    self.credit_info[vehId] = max(self.credit_info[vehId] - 0.1, param.premium_min if self.plan_info[vehId] else param.basic_min)
                cost_weight = 1.5 if self.plan_info[vehId] else 1.0
                profit = taskcpu * param.unitprice_cpu * cost_weight + tasksize * param.unitprice_size * cost_weight
                energy_coeff = 10 ** -26
                cost_comp1 = energy_coeff * optimal_resource_loc ** 2 * taskcpu * action1
                cost_comp2 = energy_coeff * optimal_resource_off ** 2 * taskcpu * (1 - action1)
                cost_comp = param.wcomp * (cost_comp1 + cost_comp2)
                cost_trans = param.cloud_trans_price if action2 == 0 else param.wtrans * (1 - action1) * tasksize * hopcount
                cost = cost_comp + cost_trans
                reward = profit - cost[0] +  tasktime - Ttotal

                # 성공 시 리워드도 r1에 모두 부여
                reward = max(1, reward / 2)  # 보상 스케일 조정
                r1 = min(reward, 5)

                self.taskEnd[stepnum] = Ttotal

            return reward, r1, Ttotal

        if cloud == 0:
            action2 += 1

        vehId = stepnum % param.numVeh
        current_credit = self.credit_info[vehId]
        tasksize, taskcpu, tasktime = param.task

        local_amount = taskcpu * action1
        off_amount = taskcpu * (1 - action1)
        optimal_resource_loc = local_amount / tasktime if local_amount else 0
        optimal_resource_off = off_amount / tasktime if off_amount else 0

        optimal_resource_loc = min(param.remains[self.nearest], optimal_resource_loc * current_credit)

        if len(self.cluster) < action2 and action1 != 1:
            self.wrong_cnt += 1
            Tloc = local_amount / optimal_resource_loc if action1 else 0
            if Tloc < tasktime:
                r1, reward, Ttotal = -5, 0, 0
            else:
                r1, reward, Ttotal = -4, 0, 0
            self.allo_loc[stepnum] = self.allo_neighbor[stepnum] = -1
            self.alloRes_loc[stepnum] = self.alloRes_neighbor[stepnum] = 0
            self.taskEnd[stepnum] = -1
            Tloc, Toff = 0, 0
        else:
            if action2 == 0:
                Ttrans = 0.8
                Tloc = local_amount / optimal_resource_loc if action1 else 0
                Toff = Ttrans if action1 != 1 else 0
                reward, r1, Ttotal = calculate_rewards_and_costs(Tloc, Toff, vehId, taskcpu, tasksize, tasktime, optimal_resource_loc, optimal_resource_off, action1, 0)
                self.allo_loc[stepnum] = self.nearest
                self.allo_neighbor[stepnum] = -1
                self.alloRes_loc[stepnum] = optimal_resource_loc if Tloc else 0
                self.alloRes_neighbor[stepnum] = 0
                param.remains[self.nearest] -= optimal_resource_loc if Tloc else 0
                self.cloud_selection += 1
            else:
                if action2 > len(self.cluster):
                    action_globid_ver = 0
                    optimal_resource_off = 0
                else:
                    action_globid_ver = self.cluster[int(action2) - 1]
                    optimal_resource_off = min(param.remains[action_globid_ver], optimal_resource_off * current_credit)

                hopcount = self.calculate_hopcount(param.edge_pos[self.nearest], param.edge_pos[action_globid_ver])
                bandwidth = np.random.uniform(0.07, 0.1)
                Ttrans = (tasksize * (1 - action1)) / (bandwidth * 1000) * hopcount
                Tcomp = off_amount / optimal_resource_off if action1 != 1 else 0
                Tloc = local_amount / optimal_resource_loc if action1 else 0
                Toff = Ttrans + Tcomp
                reward, r1, Ttotal = calculate_rewards_and_costs(Tloc, Toff, vehId, taskcpu, tasksize, tasktime, optimal_resource_loc, optimal_resource_off, action1, hopcount)
                self.allo_loc[stepnum] = self.nearest
                self.allo_neighbor[stepnum] = action_globid_ver
                self.alloRes_loc[stepnum] = optimal_resource_loc
                self.alloRes_neighbor[stepnum] = optimal_resource_off
                param.remains[self.nearest] -= optimal_resource_loc
                param.remains[action_globid_ver] -= optimal_resource_off

        new_state1, new_state2_temp = self.reset(stepnum, cloud)
        new_state2 = np.concatenate((new_state2_temp, action1))

        # Ensure reward and r1 are floats
        reward = float(reward)
        r1 = float(r1)
        
        return reward, r1, False

    def step2(self, action1, action2, stepnum): # nearest, greedy 전용 함수 (클러스터 활용 X)
        vehId = stepnum % param.numVeh
        current_credit = self.credit_info[vehId]
       
        tasksize = param.task[0]
        taskcpu = param.task[1]
        tasktime = param.task[2]

        local_amount = taskcpu * action1 
        off_amount = taskcpu * (1-action1)
        optimal_resource_loc =local_amount/tasktime #state[5] = required CPU cycles, state[6] = time deadline
        optimal_resource_loc = min(param.remains[self.nearest], optimal_resource_loc*current_credit)
        
        optimal_resource_off = off_amount/tasktime #state[5] = required CPU cycles, state[6] = time deadline

        #print("--> fraction: ", action1, "offloading decision: ", action2)
        #print("local edge: ", self.nearest,"(resource lev:", param.remains_lev[self.nearest], "-", (action1),"%)")

        optimal_resource_off = min(param.remains[action2], optimal_resource_off*current_credit)

        if action1 == 0: 
            Tloc = 0
        else:
            Tloc = local_amount/optimal_resource_loc
        hopcount = self.calculate_hopcount(param.edge_pos[self.nearest], param.edge_pos[action2])
        param.hop_counts.append(hopcount)
        bandwidth = np.random.uniform(0.07, 0.1)
        Ttrans = (tasksize*(1-action1))/(bandwidth*1000)*hopcount
        if action1 == 1: 
            Tcomp=0 
        else:
            Tcomp = off_amount/optimal_resource_off
        
        Toff = Ttrans + Tcomp
        Ttotal = max(Tloc, Toff)

        if tasktime < Ttotal: #failure
            reward = 0
            if params.userplan == 1:
                if self.plan_info[vehId] == 0: #basic
                    self.credit_info[vehId] = min(self.credit_info[vehId]+0.1, param.basic_max)
                else:
                    self.credit_info[vehId] = min(self.credit_info[vehId]+0.1, param.premium_max)
            
            self.allo_loc[stepnum] = self.nearest       
            self.allo_neighbor[stepnum] = action2     
            self.taskEnd[stepnum] = tasktime #실패한 경우 할당받은 자원 사용 시간 = latency 요구사항
        else: # 성공 -- (이웃 엣지서버 쓴 경우)
            if params.userplan == 1:
                if self.plan_info[vehId] == 0: #basic
                    self.credit_info[vehId] = max(self.credit_info[vehId]-0.1, param.basic_min)
                else:
                    self.credit_info[vehId] = max(self.credit_info[vehId]-0.1, param.premium_min)

            plan = self.plan_info[int(vehId)]
            if plan == 0:
                cost_weight = 1.0
            else:
                cost_weight = 1.5
            profit = taskcpu*param.unitprice_cpu*cost_weight + tasksize*param.unitprice_size*cost_weight
        
            energy_coeff = 10 ** -26 # effective energy coefficient - 관련 연구 논문에서 참고한 값 (후에 레퍼런스 확인하고 더 정확하게 수정할 것)
            cost_comp1 = energy_coeff * optimal_resource_loc ** 2 * taskcpu*action1 #local에서 연산하려고 사용한 에너지 소모량
            cost_comp2 = energy_coeff * optimal_resource_off ** 2* taskcpu*(1-action1) #neighbor에서 연산하려고 사용한 에너지 소모량
            cost_comp = param.wcomp*(cost_comp1+cost_comp2)
            cost_trans = param.wtrans*(1-action1)*tasksize*hopcount
            cost = cost_comp + cost_trans
            

            reward = profit - cost + tasktime - Ttotal#objective function of original problem
            #reward = max(0.1, reward /5) #old one

            # Adjust reward based on revenue
            reward = max(1, reward / 2)  # Scale down the reward to make it manageable
    
            self.taskEnd[stepnum] = Ttotal

            self.allo_loc[stepnum] = self.nearest
            self.allo_neighbor[stepnum] = action2

            if Tloc == 0:
                optimal_resource_loc = 0
            if Tcomp == 0:
                optimal_resource_off = 0

        self.alloRes_loc[stepnum]=optimal_resource_loc
        self.alloRes_neighbor[stepnum]= optimal_resource_off
            
        param.remains[self.nearest] -= optimal_resource_loc
        param.remains[action2] -= optimal_resource_off

        #print("offloading: ", action2, "(resource lev:", param.remains_lev[action2], "-", 1-action1, "%)")

        #print("req. latency: ", tasktime, "Ttotal: ",Ttotal, "Tloc: ", Tloc, "Toff", Toff, "\n")
        

        done = False
        new_state1, new_state2_temp = self.reset(stepnum, 1)

        action1 = np.array(action1, ndmin=1)
        new_state2 = np.concatenate((new_state2_temp, action1))
        done = False
        return new_state1, new_state2, reward, 0, 0, done