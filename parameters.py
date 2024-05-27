import numpy as np
'''basic units'''
Byte = 8
Kilo = 1000
Mega = 1000 * Kilo
Giga = 1000 * Mega 
task = np.zeros(3)

'''model path'''
sac_path = './model/sac'
ppo_path = './model/ppo'
sac_interval  = 10

'''training parameters'''
EPS=1000
STEP = 10
sac_batch = 512
update_itr = 1
ppo_batch = 128
GAMMA = 0.99
''''''
numEdge = 16
numVeh = 5


'''global state'''
state1 = np.zeros(numEdge) 
state2 = np.zeros(numEdge+1)

'''state, action dimension'''
state_dim1 = numEdge  #ppo
state_dim2 = numEdge+1 #ppo
action_dim1 = 1 #sac
action_dim2 = numEdge #sac
hidden_dim = 128

'''edge server resources'''
remains = np.zeros(numEdge)
remains_lev = np.zeros(numEdge)

hop_count = np.zeros(numEdge)
temp = np.zeros(numEdge)

resource_avg = 50
resource_std = 20

'''clustering params'''

'''task information'''
min_size = 0.1  * Byte
max_size = 1  * Byte
min_cpu = 0.1 
max_cpu = 1 
min_time = 0.1
max_time = 5
unitprice_size = 1 # 차량 지불 함수에서 가중치
unitprice_cpu = 1 # 차량 지불 함수에서 가중치
wcomp = 1 # 소모 함수에서 가중치
wtrans = 1 # 소모 함수에서 가중치

edge_pos = np.array([[0.5, 0.5], [1.5, 0.5],[2.5, 0.5], [3.5, 0.5],
                    [0.5, 1.5], [1.5, 1.5],[2.5, 1.5], [3.5, 1.5],
                    [0.5, 2.5], [1.5, 2.5],[2.5, 2.5], [3.5, 2.5],
                    [0.5, 3.5], [1.5, 3.5],[2.5, 3.5], [3.5, 3.5]])
radius = 0.5

