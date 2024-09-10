import numpy as np
import math
'''basic units'''
Byte = 8
Kilo = 1000
Mega = 1000 * Kilo
Giga = 1000 * Mega 
task = np.zeros(3)
time_slot = 1

GPU = False


'''training parameters'''
EPS=500
STEP = 20
GAMMA = 0.99

'''numEdge를 바꾸면 lamb, maxEdge를 함께 조정해주어야 함!'''
numEdge = 25
numVeh = 200
lamb = 2 #시스템 내 클러스터 개수 (K-means에서 K와 같은 역할함)
maxEdge = int(numEdge/lamb)+3 #하나의 클러스터에 최대 몇개의 엣지가 포함될 수 있는지를 정의함
CHs = []
CMs = [[]]
nearest = -1


'''model path'''
# 9월 10일에 새롭게 트레이닝하면서 1자를 붙임
# 즉, 이전에 학습된 모델은 ppo_, dqn_, woclst_dqn_,...의 이름으로 저장되어 있음!!
ppo_path = './model/ppo1_'+str(int(numEdge))+str(int(lamb))
dqn_path = './model/dqn1_'+str(int(numEdge))+str(int(lamb))

woClst_dqn_path = './model/woclst_dqn1_'+str(int(numEdge))+str(int(lamb))
woClst_ppo_path = './model/woclst_ppo1_'+str(int(numEdge))+str(int(lamb))

woCloud_dqn_path = './model/wocloud_dqn1_'+str(int(numEdge))+str(int(lamb))
woCloud_ppo_path = './model/wocloud_ppo1_'+str(int(numEdge))+str(int(lamb))

'''global state'''
state1 = np.zeros(1+3+1) 
state2 = np.zeros((maxEdge+1)*2+3+1)

'''state, action dimension'''
state_dim1 = 2+3  #ppo
state_dim2 = (maxEdge+1)*2+3+1#ppo
action_dim1 = 1 #sac
action_dim2 = maxEdge + 1 #sac
hidden_dim = 128

'''wocloud dimension'''
wocloud_state_dim2 = (maxEdge)*2+3+1#ppo
wocloud_action_dim2 = maxEdge#sac

'''edge server resources'''
remains = np.zeros(numEdge)
remains_lev = np.zeros(numEdge)
hop_count = np.zeros(numEdge)
temp = np.zeros(numEdge)
resource_avg = 10
resource_std = 10

'''task information'''
min_size = 0.1  * Byte
max_size = 1  * Byte
min_cpu = 0.1 
max_cpu = 3
min_time = 0.1
max_time = 2
unitprice_size = 2 # 차량 지불 함수에서 가중치
unitprice_cpu = 2 # 차량 지불 함수에서 가중치
wcomp = 10**26 # 소모 함수에서 가중치
wtrans = 0.5 # 소모 함수에서 가중치

grid_size = int(math.sqrt(numEdge))
edge_pos = np.zeros((numEdge, 2))

radius = 0.5


'''graph'''
wrong_cnt = []
epsilon_logging = []
cloud_cnt = []

'''Resource allocation algorithm params'''
basic_min = 1.2
basic_max = 1.6
basic_init = (basic_min+basic_max)/2 #now: 1.4

premium_min = 1.6
premium_max = 2.4
premium_init = (premium_max+premium_min)/2 #now: 2
cloud_trans_price = 2.5
'''plot parameters '''
font_size = 18
credit_info = np.zeros(numVeh) 


# model hyper parameters
# we use two RL model for making two actions 
# action1: offloading fraction decision (continous action space 0~1) making from PPO
# action2: offloaing server decision (discrete action space 0~N, 0: cloud, 1~N: edge servers) making from DQN

dqnlr = 1e-4  # dqn의 Q 네트워크 학습률
actorlr = 1e-4  # ppo - actor 학습률
criticlr = 5e-4  # ppo - critic 학습률
scheduler_step = 1000  # 학습률 스케줄러 단계
scheduler_gamma = 0.995  # 학습률 스케줄러 감쇠 계수
dqn_batch = 32  # dqn 배치 크기
ppo_batch = 512  # ppo 배치 크기


# The number of total step per one episode is 2000

cloud =1 #default values
distribution_mode = 0
repeat = 4