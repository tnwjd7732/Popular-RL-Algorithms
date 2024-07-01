import numpy as np
import math
'''basic units'''
Byte = 8
Kilo = 1000
Mega = 1000 * Kilo
Giga = 1000 * Mega 
task = np.zeros(3)
time_slot = 0.5

GPU = False

'''model path'''
ppo_path = './model/ppo' 
sac_interval  = 199

'''training parameters'''
EPS=1000
STEP = 50
dqn_batch = 128
update_itr = 199
ppo_batch = 256
GAMMA = 0.99
''''''
numEdge = 16
numVeh = 50
maxEdge = 5 #하나의 클러스터에 최대 몇개의 엣지가 포함될 수 있는지를 정의함


'''global state'''
state1 = np.zeros(maxEdge+3+1) 
state2 = np.zeros((maxEdge+1)*2+3+1)

'''state, action dimension'''
state_dim1 = maxEdge+1+3  #ppo
state_dim2 = (maxEdge+1)*2+3+1#ppo
action_dim1 = 1 #sac
action_dim2 = maxEdge + 1 #sac
hidden_dim = 256

'''edge server resources'''
remains = np.zeros(numEdge)
remains_lev = np.zeros(numEdge)


hop_count = np.zeros(numEdge)

temp = np.zeros(numEdge)

resource_avg = 10
resource_std = 10

'''clustering params'''
lamb = 4 #number of clusters in the system
CHs = []
CMs = [[]]

'''task information'''
min_size = 0.1  * Byte
max_size = 1  * Byte
min_cpu = 0.1 
max_cpu = 3
min_time = 0.1
max_time = 3
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
cloud_trans_price = 1
'''plot parameters '''
font_size = 10