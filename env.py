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
    
    def reset(self, step):
        #print("Credit information ---- ")
        #print(self.credit_info)
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
            #print("이전 에피소드에서의 wrong action 횟수: ", self.wrong_cnt)
            param.wrong_cnt.append(self.wrong_cnt)
            param.cloud_cnt.append(self.cloud_selection)
            self.wrong_cnt = 0
            self.cloud_selection = 0
            for i in range(param.numVeh):
                self.plan_info[i] = random.randint(0,1)
                if self.plan_info[i] == 0:
                    self.credit_info[i] = param.basic_init
                else:
                    self.credit_info[i] = param.premium_init
                        
        for i in range(params.numEdge):
            params.remains_lev[i] = int(params.remains[i]/10)
        #print("remains level: ", params.remains_lev)       

        ''' 작업 초기화 '''
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
        self.calculate_hopcount2(param.edge_pos[int(self.nearest)])
        myClusterId = self.find_my_cluster(int(self.nearest))
        #print("My edge: ", self.nearest, "My CH: ", myClusterId)
        self.myCH = myClusterId
        
        cloud_resource = [100]
        cloud_hop = [100]

        if myClusterId is not None:
            cluster_servers = [myClusterId] + params.CMs[myClusterId]
            cluster_servers.sort()
            self.cluster = cluster_servers

            cluster_remains_lev = [params.remains_lev[i] for i in cluster_servers]
            cluster_size = len(cluster_servers)
            dummy_size = max(0, params.maxEdge - cluster_size)  # 더미값의 크기 결정
            dummy_values = np.full(dummy_size, -10)  # 더미값을 -10으로 설정
            state = np.concatenate((cloud_resource, cluster_remains_lev, dummy_values, self.taskInfo))
        else:
            dummy_values = np.full(params.maxEdge+1, -10)  # 클러스터에 속하지 않은 경우 더미값으로 채움
            state = np.concatenate((dummy_values, self.taskInfo))

        if myClusterId is not None:
            cluster_servers = [myClusterId] + params.CMs[myClusterId]
            cluster_servers.sort()

            cluster_remains_lev = [params.remains_lev[i] for i in cluster_servers]
            cluster_hop_count = [params.hop_count[i] for i in cluster_servers]  # 클러스터에 속한 서버들의 홉 카운트 정보
            cluster_size = len(cluster_servers)
            dummy_size = max(0, params.maxEdge - cluster_size)  # 더미값의 크기 결정
            dummy_values = np.full(dummy_size, -10)  # 더미값을 -10으로 설정
            state2 = np.concatenate((cloud_resource, cluster_remains_lev, dummy_values,cloud_hop, cluster_hop_count, dummy_values, self.taskInfo))
        else:
            dummy_values = np.full(params.maxEdge+1, -10)  # 클러스터에 속하지 않은 경우 더미값으로 채움
            state2 = np.concatenate((dummy_values, dummy_values, self.taskInfo))

        return state, state2
   
    def find_my_cluster(self, server_id):
        for ch_id in params.CHs:
            if server_id == ch_id or server_id in params.CMs[ch_id]:
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

    def step(self, action1, action2, stepnum):
        #action1: offloading fraction (0~1) from ppo (continous)
        #action2: offloading decision (0~N) from sac (discrete)

        # 0. vehicle의 plan info 가져오기
        vehId = stepnum % param.numVeh
        current_credit = self.credit_info[vehId]
       
        # 1. offloading fraction에 따라 local computing량과 offloading computing량 계산
        tasksize = param.task[0]
        taskcpu = param.task[1]
        tasktime = param.task[2]


        local_amount = taskcpu * action1 
        off_amount = taskcpu * (1-action1)
        #print("fraction: ", action1, ", local: ", local_amount, "Gigacycle, offload: ", off_amount, "Gigacycle")
        optimal_resource_loc =local_amount/tasktime #state[5] = required CPU cycles, state[6] = time deadline
        optimal_resource_loc = min(param.remains[self.nearest], optimal_resource_loc*current_credit)
        
        optimal_resource_off = off_amount/tasktime #state[5] = required CPU cycles, state[6] = time deadline

        print("--> fraction: ", action1, "offloading decision: ", action2, "cluster: ", self.cluster)
        print("local edge: ", self.nearest,"(resource lev:", param.remains_lev[self.nearest], "-", (action1),"%)")
        action_globid_ver = -1 #default value  
        if(len(self.cluster)) < action2:
            r1 = 0
            r2 = -5
            reward = 0
            self.allo_loc[stepnum] = 0
            self.allo_neighbor[stepnum] = 0
            self.alloRes_loc[stepnum]=0
            self.alloRes_neighbor[stepnum]= 0
            self.taskEnd[stepnum] = -1
            #print("wrong action2 ...")
            self.wrong_cnt+=1
            Ttotal = 0
            Tloc = 0
            Toff = 0

        elif action2==0:
            #print("select cloud server")
            self.cloud_selection += 1
            Ttrans = 1
            if action1 == 0: #자기에서 하나도 안한다는 것 (전부 클라우드에게)
                Tloc = 0
                Toff = Ttrans
            elif action1 == 1: #로컬에서 전부하겠다는 것 
                Tloc = local_amount/optimal_resource_loc
                Toff = 0
            else:
                Tloc=local_amount/optimal_resource_loc
                Toff = Ttrans

            Ttotal=max(Tloc, Toff)

            if tasktime < Ttotal: #실패
                if self.plan_info[vehId] == 0: #basic
                    self.credit_info[vehId] = min(self.credit_info[vehId]+0.1, param.basic_max)
                else:
                    self.credit_info[vehId] = min(self.credit_info[vehId]+0.1, param.premium_max)

                #print("\n----> Fraction: ", action1)
                #print("----> MyId: ", self.nearest,"(lev:", param.remains_lev[self.nearest], "-", (1-action1), ") Offloading: Cloud")
                #print("Failue - Ttotal:" , Ttotal, "Tloc: ", Tloc, "Toff: ", Toff)
                reward = 0
                r1 = 0
                r2 = 0

                if math.isinf(Ttotal):
                    if math.isinf(Tloc):
                        r1 = -1  # Penalty for incorrect local fraction decision
                    if math.isinf(Toff):
                        r1 = -0.5  # Penalty for incorrect local fraction decision
                        r2 = -1  # Penalty for incorrect offloading decision
                elif Ttotal > tasktime:
                    if Tloc > tasktime:
                        r1 = -0.5  # Penalty for exceeding local processing time
                    if Toff > tasktime:
                        r2 = -0.5  # Penalty for exceeding offloading processing time
                
                if Toff < tasktime:
                    r2 = 0.5

                self.taskEnd[stepnum] = tasktime #실패한 경우 할당받은 자원 사용 시간 = latency 요구사항
            else: #성공 -- (클라우드 쓴 경우 )
                if self.plan_info[vehId] == 0: #basic
                    self.credit_info[vehId] = max(self.credit_info[vehId]-0.1, param.basic_min)
                else:
                    self.credit_info[vehId] = max(self.credit_info[vehId]-0.1, param.premium_min)

                #print("Success - Ttotal:" , Ttotal, "Tloc: ", Tloc, "Toff: ", Toff)
                #objective function (maximize하고 싶었던 것)
                # 우리 논문에서는 revenue를 최대화 하고자함
                # revenue = profit - cost
                # profit = vehicle에게 받아오는 offloading 비용 (두개의 unit price plan을 고려하고 있음)
                # cost = 계산에 사용한 에너지, 통신 비용 (서버 간 유선 통신, 클라우드까지 접근 비용)
                
                plan = self.plan_info[int(vehId)]
                if plan == 0:
                    cost_weight = 1.0
                else:
                    cost_weight = 1.5
                #profit = 차량에게 받아오는 돈 (요금제 별 unit price를 다르게 설정, task size, resource 요구량에 따라 비용 증가)
                profit = taskcpu*param.unitprice_cpu*cost_weight + tasksize*param.unitprice_size*cost_weight
            
                energy_coeff = 10 ** -26 # effective energy coefficient - 관련 연구 논문에서 참고한 값 (후에 레퍼런스 확인하고 더 정확하게 수정할 것)
                cost_comp1 = energy_coeff * optimal_resource_loc ** 2 * taskcpu*action1 #local에서 연산하려고 사용한 에너지 소모량
                cost_comp2 = energy_coeff * optimal_resource_off ** 2* taskcpu*(1-action1) #neighbor에서 연산하려고 사용한 에너지 소모량
                cost_comp = param.wcomp*(cost_comp1+cost_comp2)
                cost_trans = param.cloud_trans_price
                cost = cost_comp + cost_trans
                cost = cost[0]

                reward = profit - cost #objective function of original problem
                reward = max(0.1, reward /5)
               
                print(abs(Tloc-Toff), Tloc, Toff)
                r1 = max(1, reward)- min(abs(Tloc-Toff),0.5)
                r2 = max(1, reward)- min(abs(Tloc-Toff),0.5)
            
                self.taskEnd[stepnum] = Ttotal

            self.allo_loc[stepnum] = self.nearest
            

            self.alloRes_loc[stepnum]=optimal_resource_loc
            param.remains[self.nearest] -= optimal_resource_loc
            
        else:
            '''여기서 action2가 실제로 선택 불가한 (len(self.cluster)보다 큰 값일 경우 어케야할까?)'''
            action_globid_ver = self.cluster[int(action2)-1]
            optimal_resource_off = min(param.remains[action_globid_ver], optimal_resource_off*current_credit)

            # 2. local computing time 계산
            # 2.1 Tloc = computing time (우선 모든 자원 - remains 다 할당한다고 가정하고 코드 짜기) version 1
            if action1 == 0: #로컬에서 하나도 안하고 전부 오프로딩
                Tloc = 0
            else:
                Tloc = local_amount/optimal_resource_loc
            # 3. offloading time 계산
            # 3.1 transmission delay 계산 (hop count 가지고 init -> comp까지)
            
            hopcount = self.calculate_hopcount(param.edge_pos[self.nearest], param.edge_pos[action_globid_ver])
            bandwidth = np.random.uniform(0.07, 0.1)
            Ttrans = (tasksize*(1-action1))/(bandwidth*1000)*hopcount
            # 3.2 computing delay 계산
        
            if action1 == 1: #아예 내가 계산할 일 없음
                Tcomp=0 
            else:#조금은 계산해야 함 (혹은 전체?)
                Tcomp = off_amount/optimal_resource_off
            
            # 3.3 Toff = trans +  comp
            Toff = Ttrans + Tcomp

            # 4. local과 off 중 더 긴 시간을 최종 Ttotal로 결정
            #print("Tloc: ",  Tloc, "Toff: ", Toff)
            Ttotal = max(Tloc, Toff)
            #print("  size: ", tasksize, "cpu: ", taskcpu, "latency: ", tasktime)

            # 5. 성공 여부 체크 
            if tasktime < Ttotal: #실패
                if self.plan_info[vehId] == 0: #basic
                    self.credit_info[vehId] = min(self.credit_info[vehId]+0.1, param.basic_max)
                else:
                    self.credit_info[vehId] = min(self.credit_info[vehId]+0.1, param.premium_max)

                #print("\n----> Fraction: ", action1)
                #print("Failue - Ttotal:" , Ttotal, "Tloc: ", Tloc, "Toff: ", Toff, "hopcnt: ", hopcount)
                reward = 0
                r1 = 0
                r2 = 0

                if math.isinf(Ttotal):
                    if math.isinf(Tloc):
                        r1 = -1  # Penalty for incorrect local fraction decision
                    if math.isinf(Toff):
                        r1 = -0.5  # Penalty for incorrect local fraction decision
                        r2 = -1  # Penalty for incorrect offloading decision
                elif Ttotal > tasktime:
                    if Tloc > tasktime:
                        r1 = -0.5  # Penalty for exceeding local processing time
                    if Toff > tasktime:
                        r2 = -0.5  # Penalty for exceeding offloading processing time
                
                if Toff < tasktime:
                    r2 = 0.5
                self.taskEnd[stepnum] = tasktime #실패한 경우 할당받은 자원 사용 시간 = latency 요구사항
            else: # 성공 -- (이웃 엣지서버 쓴 경우)
                if self.plan_info[vehId] == 0: #basic
                    self.credit_info[vehId] = max(self.credit_info[vehId]-0.1, param.basic_min)
                else:
                    self.credit_info[vehId] = max(self.credit_info[vehId]-0.1, param.premium_min)

                #print("Success - Ttotal:" , Ttotal, "Tloc: ", Tloc, "Toff: ", Toff)
                plan = self.plan_info[int(vehId)]
                if plan == 0:
                    cost_weight = 1.0
                else:
                    cost_weight = 1.5
                #profit = 차량에게 받아오는 돈 (요금제 별 unit price를 다르게 설정, task size, resource 요구량에 따라 비용 증가)
                profit = taskcpu*param.unitprice_cpu*cost_weight + tasksize*param.unitprice_size*cost_weight
            
                energy_coeff = 10 ** -26 # effective energy coefficient - 관련 연구 논문에서 참고한 값 (후에 레퍼런스 확인하고 더 정확하게 수정할 것)
                cost_comp1 = energy_coeff * optimal_resource_loc ** 2 * taskcpu*action1 #local에서 연산하려고 사용한 에너지 소모량
                cost_comp2 = energy_coeff * optimal_resource_off ** 2* taskcpu*(1-action1) #neighbor에서 연산하려고 사용한 에너지 소모량
                cost_comp = param.wcomp*(cost_comp1+cost_comp2)
                cost_trans = param.wtrans*(1-action1)*tasksize*hopcount
                cost = cost_comp + cost_trans
                cost = cost[0]

                reward = profit - cost #objective function of original problem
                reward = max(0.1, reward /5)
                

                r1 = max(1, reward)- min(abs(Tloc-Toff),0.5)
                r2 = max(1, reward)- min(abs(Tloc-Toff),0.5)
              
                self.taskEnd[stepnum] = Ttotal

            self.allo_loc[stepnum] = self.nearest
            self.allo_neighbor[stepnum] = action_globid_ver

            self.alloRes_loc[stepnum]=optimal_resource_loc
            self.alloRes_neighbor[stepnum]= optimal_resource_off

            '''action2가 out of range여도 기록해야 하는 것들'''

            # 7. 서버들 가용 자원량 조정 (위에서 할당한 만큼 빼는 것)
            param.remains[self.nearest] -= optimal_resource_loc
            param.remains[action_globid_ver] -= optimal_resource_off
        
        print("offloading: ", action2, "(resource lev:", param.remains_lev[action_globid_ver], "-", 1-action1, "%)")

        print("reward: ", reward, "r1: ", r1, "r2: ", r2)
        print("req. latency: ", tasktime, "Ttotal: ",Ttotal, "Tloc: ", Tloc, "Toff", Toff, "\n")
        

        done = False
        new_state1, new_state2_temp = self.reset(stepnum)

       
        new_state2 = np.concatenate((new_state2_temp, action1))
        done = False

        # Ensure reward is a float
        reward = float(reward) if not isinstance(reward, float) else reward

        # Ensure r1 is a float
        r1 = float(r1) if not isinstance(r1, float) else r1

        # Ensure r2 is a float
        r2 = float(r2) if not isinstance(r2, float) else r2

        return new_state1, new_state2, reward, r1, r2, done