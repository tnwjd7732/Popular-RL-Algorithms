import numpy as np
import math
'''basic units'''
Byte = 8
Kilo = 1000
Mega = 1000 * Kilo
Giga = 1000 * Mega 
task = np.zeros(3)

GPU = False

'''model path'''
sac_path = './model/sac'
ppo_path = './model/ppo'
sac_interval  = 199

'''training parameters'''
EPS=1000
STEP = 10
sac_batch = 128
dqn_batch = 64
update_itr = 199
ppo_batch = 512
GAMMA = 0.99
''''''
numEdge = 25
numVeh = 100
maxEdge = 10 #하나의 클러스터에 최대 몇개의 엣지가 포함될 수 있는지를 정의함


'''global state'''
state1 = np.zeros(maxEdge+3) 
state2 = np.zeros(maxEdge*2+3+1)

'''state, action dimension'''
state_dim1 = maxEdge+3  #ppo
state_dim2 = maxEdge*2+3+1#ppo
action_dim1 = 1 #sac
action_dim2 = maxEdge #sac
hidden_dim = 128

'''edge server resources'''
remains = np.zeros(numEdge)
remains_lev = np.zeros(numEdge)


hop_count = np.zeros(numEdge)

temp = np.zeros(numEdge)

resource_avg = 20
resource_std = 15

'''clustering params'''
lamb = 5 #number of clusters in the system
CHs = []
CMs = [[]]

'''task information'''
min_size = 0.1  * Byte
max_size = 1  * Byte
min_cpu = 0.1 
max_cpu = 3
min_time = 0.5
max_time = 3
unitprice_size = 1 # 차량 지불 함수에서 가중치
unitprice_cpu = 1 # 차량 지불 함수에서 가중치
wcomp = 1 # 소모 함수에서 가중치
wtrans = 1 # 소모 함수에서 가중치

grid_size = int(math.sqrt(numEdge))
edge_pos = np.zeros((numEdge, 2))

radius = 0.5


'''graph'''
wrong_cnt = []