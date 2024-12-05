import numpy as np
import math
'''basic units'''
Byte = 8
Kilo = 1000
Mega = 1000 * Kilo
Giga = 1000 * Mega 
task = np.zeros(3)
time_slot = 1

'''network scale'''
numEdge = 64
numVeh = 700
lamb = 4 ## of clusters in the network
maxEdge = int(numEdge/lamb)+int(math.sqrt(numEdge)/2) # max number of edge servers in a cluster
grid_size = int(math.sqrt(numEdge))

'''initialize edge server position'''
edge_pos = np.zeros((numEdge, 2))
radius = 0.5
x = -1
y = 0
for i in range(numEdge):
    if i % grid_size == 0:
        x += radius*2
        y = 0
    edge_pos[i] = [radius + y, radius + x]
    y += radius*2


'''task profile range'''
min_size = 0.1  * Byte #1 MB
max_size = 2  * Byte #20MB
min_cpu = 1
max_cpu = 3 # 5Giga cycle per second
min_time = 0.1
max_time = 1

'''cost function parameter'''
unitprice_size = 0.1 # 차량 지불 함수에서 가중치
unitprice_cpu = 2 # 차량 지불 함수에서 가중치
wcomp = 2e+23 # 소모 함수에서 가중치
wtrans = 0.01 #12 소모 함수에서 가중치
cloud_trans_price = 0.1

'''credit based RA (resource allocation) algorithm params'''
basic_min = 1.2
basic_max = 1.6
basic_init = (basic_min+basic_max)/2 
premium_min = 1.6
premium_max = 2.0
premium_init = (premium_max+premium_min)/2 

'''hyper parameters'''
dqnlr = 5e-4 # lr for dqn Q network
actorlr = 5e-5  # lr for ppo actor net (=policy net) 이걸  1e-4, -5 이상 수준으로 줄이면 오히려 성능이 오르다가 다시 줄어드는 현상 보임
criticlr = 5e-4  # lr for ppo critic net (=value net)
cloud = 1 # 1: using cloud, 0: did not use cloud
dqn_batch = 256  
ppo_batch = 2048 

'''learning rate scheduler parameter'''
scheduler_step = 1000 
scheduler_gamma = 0.999 # did not use (current)

# ------------------------------- DO NOT MODIFY ------------------------ #

'''model path'''
ppo_path = './model/ppo1_'+str(int(numEdge))+str(int(lamb))
dqn_path = './model/dqn1_'+str(int(numEdge))+str(int(lamb))

baseline_ppo_path = './model/base_ppo1_'+str(int(numEdge))+str(int(lamb))
baseline_dqn_path = './model/base_dqn1_'+str(int(numEdge))+str(int(lamb))

ppo_single_path = './model/single_ppo1_'+str(int(numEdge))+str(int(lamb))
dqn_single_path = './model/single_dqn1_'+str(int(numEdge))+str(int(lamb))

woClst_dqn_path = './model/woclst_dqn1_'+str(int(numEdge))+str(int(lamb))
woClst_ppo_path = './model/woclst_ppo1_'+str(int(numEdge))+str(int(lamb))

woCloud_dqn_path = './model/wocloud_dqn1_'+str(int(numEdge))+str(int(lamb))
woCloud_ppo_path = './model/wocloud_ppo1_'+str(int(numEdge))+str(int(lamb))

staticClst_dqn_path = './model/staticClst_dqn1_'+str(int(numEdge))+str(int(lamb))
staticClst_ppo_path = './model/staticClst_ppo1_'+str(int(numEdge))+str(int(lamb))

'''define state, action dimension'''
state1 = np.zeros(1+3+1) 
state2 = np.zeros((maxEdge+1)*2+3+1)
state_dim1 = 2+3  #ppo
state_dim2 = (maxEdge+1)*2+3+1#ppo
action_dim1 = 1 #sac
action_dim2 = maxEdge + 1 #sac
hidden_dim = 128 # change 128 to 256
single_state_dim=(maxEdge+1)*2+3
single_action1_dim=1
single_action2_dim=maxEdge + 1

'''graph'''
wrong_cnt = []
epsilon_logging = []
cloud_cnt = []
valid_cnt = []

'''clustering parameter'''
CHs = []
CMs = [[]]

'''edge server information'''
remains = np.zeros(numEdge)
remains_lev = np.zeros(numEdge)
hop_count = np.zeros(numEdge)
temp = np.zeros(numEdge)
resource_avg = 10
resource_std = 7

'''enable GPU(mps or cuda)'''
GPU = False

'''load pre-trained model'''
pre_trained = False

'''basic RL setting'''
EPS = 500
STEP = 5
GAMMA = 0.99

'''plot parameters '''
font_size = 18
credit_info = np.zeros(numVeh) 

'''shared variable across all of the files'''
nearest = -1
userplan=1 # userplan: 1 - 우리 알고리즘 (프리미엄 베이직 유저로 나누고 할당량 다르게)
# 1 이외의 값 > 그 값이 바로 고정 할당량으로 설정되고 변화하지 않음
mycluster = []
glob = 0
CH_glob_ID = -1
distribution_mode = 0
repeat = 6
hop_counts= []
std_exp = 1
realtime = 1
costpaid = 1
success = True 
stepnum = 0
actor_loss = []
critic_loss = []