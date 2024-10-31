import parameters as param
import numpy as np
import env 
environment = env.Env()


class Nearest():
    def choose_action(self):
        action2 = param.nearest
        action1 = 1 # always process itself (no offloading)
        return action1, action2

class Greedy():
    def choose_action(self, n, stepnum): #n: h-hop 이웃 중 최대 노드 고르기
        myId = param.nearest
        taskcpu = param.task[1]

        vehId = stepnum % param.numVeh
        credit = param.credit_info[vehId]

        #print("Credit:", credit)
        req_taskcpu = taskcpu #* credit

        # 만약 자신의 자원이 충분하다면, 자기가 처리하기
        if param.remains[myId] >= req_taskcpu:
            action2 = myId
            action1 = 1
        
        # 그렇지 않은 경우, 자신의 n-hop 이웃 중 가장 자원이 많은 서버에게 toss
        else:
            action2 = self.greedy_action(myId, n)
            #print("best neighbor =  ", action2)
            action1 = 0 #모두 오프로딩하는 것 (자신이 처리하는 것 전혀 없음)

        action1 = 0 # no partial offloading strategy
        return action1, action2
    
    def greedy_action(self, nodeId, n):
        numEdge = param.numEdge
        grid_size = int(np.sqrt(numEdge))

        nodeId=int(nodeId)
        
        # Get coordinates of the nodeId
        node_x, node_y = param.edge_pos[nodeId]
        
        def get_neighbors(nodeId, n):
            """ Get n-hop neighbors of the given nodeId """
            node_x, node_y = param.edge_pos[nodeId]
            neighbors = []
            for i in range(numEdge):
                if i == nodeId:
                    continue
                x, y = param.edge_pos[i]
                distance = np.sqrt((node_x - x)**2 + (node_y - y)**2)
                if distance <= n * (param.radius*2):
                    neighbors.append(i)
            #print(neighbors)
            return neighbors
        
        neighbors = get_neighbors(nodeId, n)
        
        # Initialize the max resource and the best neighbor
        max_resource = -1
        best_neighbor = -1
        
        # Iterate through the neighbors to find the one with the maximum available resources
        for neighbor in neighbors:
            if param.remains_lev[neighbor] > max_resource:
                max_resource = param.remains_lev[neighbor]
                best_neighbor = neighbor
        
        return best_neighbor
    
class GreedyCloud():
    def choose_action(self, n, stepnum):  # n: h-hop 이웃 중 최대 노드 고르기
        myId = param.nearest
        taskcpu = param.task[1]
        vehId = stepnum % param.numVeh
        credit = param.credit_info[vehId]

        # 요청된 작업에 필요한 CPU 자원
        req_taskcpu = taskcpu #* credit

        # 만약 자신의 자원이 충분하다면, 자기가 처리
        if param.remains[myId] >= req_taskcpu:
            action2 = myId
            action1 = 1  # 자기 자신이 처리
        else:
            # 자신의 n-hop 이웃 중 가장 자원이 많은 서버에게 오프로딩
            action2 = self.greedy_action(myId, n)
            
            # 이웃 중 가장 자원이 많은 서버도 처리할 수 없는 경우 클라우드로 보냄
            if action2 == -1 or param.remains[action2] < req_taskcpu:
                action2 = 0  # 클라우드로 오프로딩
                action1 = 0  # 모두 오프로딩
            else:
                action1 = 0  # 이웃에게 오프로딩

        return action1, action2
    
    def greedy_action(self, nodeId, n):
        numEdge = param.numEdge
        grid_size = int(np.sqrt(numEdge))
        nodeId = int(nodeId)

        # Get coordinates of the nodeId
        node_x, node_y = param.edge_pos[nodeId]

        def get_neighbors(nodeId, n):
            """Get n-hop neighbors of the given nodeId."""
            node_x, node_y = param.edge_pos[nodeId]
            neighbors = []
            for i in range(numEdge):
                if i == nodeId:
                    continue
                x, y = param.edge_pos[i]
                distance = np.sqrt((node_x - x) ** 2 + (node_y - y) ** 2)
                if distance <= n * (param.radius * 2):
                    neighbors.append(i)
            return neighbors

        neighbors = get_neighbors(nodeId, n)

        # Initialize the max resource and the best neighbor
        max_resource = -1
        best_neighbor = -1

        # Iterate through the neighbors to find the one with the maximum available resources
        for neighbor in neighbors:
            if param.remains_lev[neighbor] > max_resource:
                max_resource = param.remains_lev[neighbor]
                best_neighbor = neighbor

        return best_neighbor
class NearestCloud():
    def choose_action(self, stepnum):
        myId = param.nearest
        taskcpu = param.task[1]
        vehId = stepnum % param.numVeh
        credit = param.credit_info[vehId]

        # 요청된 작업에 필요한 CPU 자원
        req_taskcpu = taskcpu# * credit

        # 만약 자신의 자원이 충분하다면, 자기가 처리
        if param.remains[myId] >= req_taskcpu:
            action2 = myId
            action1 = 1  # 자기 자신이 처리
        else:
            # 자신의 자원이 부족하면 클라우드로 보냄
            action2 = 0  # 클라우드의 아이디
            action1 = 0  # 모두 오프로딩

        return action1, action2


