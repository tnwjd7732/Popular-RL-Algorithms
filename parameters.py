import numpy as np
'''basic units'''
Byte = 8
Kilo = 1000
Mega = 1000 * Kilo
Giga = 1000 * Mega 

'''model path'''
sac_path = './model/sac'
ppo_path = './model/ppo'


'''training parameters'''
EPS=100
STEP = 10
sac_batch = 256
update_itr = 1
ppo_batch = 1024
GAMMA = 0.99
''''''
numEdge = 16
numVeh = 10
maxStep = 10


'''edge server resources'''
remains = np.zeros(numEdge)
resource_avg = 50
resource_std = 20

'''clustering params'''

'''task information'''
min_size = 0.1 * Mega * Byte
max_size = 1 * Mega * Byte
min_cpu = 0.1 * Giga
max_cpu = 1 * Giga
min_time = 0.1
max_time = 5

edge_pos = np.array([[0.5, 0.5], [1.5, 0.5],[2.5, 0.5], [3.5, 0.5],
                    [0.5, 1.5], [1.5, 1.5],[2.5, 1.5], [3.5, 1.5],
                    [0.5, 2.5], [1.5, 2.5],[2.5, 2.5], [3.5, 2.5],
                    [0.5, 3.5], [1.5, 3.5],[2.5, 3.5], [3.5, 3.5]])

