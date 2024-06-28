import numpy as np
import skfmm
from scipy.optimize import linear_sum_assignment
from random import shuffle

# Global Policy - implementation of "Multi-robot collaborative dense scene reconstruction"

def kmeans(np_obstacle_map, np_frontier_map, k):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
        :param boxes: numpy array of shape (r, 2), where r is the number of rows
        :param k: number of clusters
        :return: numpy array of shape (k, 2)
    @ note: when the indices returned by numpy, they always return in [y, x] order
            while the goal and robot positions are in a natural [x, y] order
    """

    is_frontier = np_frontier_map == 1 # it's [y,x] order
    frontier_idx = np.where(is_frontier) # [[y1, y2...], [x1, x2...]]
    rows = frontier_idx[0].shape[0] # number of frontier points in the discretized map (exclude duplicates)
    # execption: no frontiers or fewer frontiers than clusters
    if rows == 0:
        return None, None, None, None
    if rows < k:
        k = rows

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    # randomly select k frontier points as initial cluster centers
    cluster_idx = np.arange(rows)
    shuffle(cluster_idx)
    cluster_idx = cluster_idx[:k]

    # loop over 5 iterations for convergence
    for count in range(5):

        # For each cluster center, calculates geodesic distances to all frontier points using the Fast Marching Method (FMM).
        for k_i, ci in enumerate(cluster_idx):
            cy, cx = frontier_idx[0][ci], frontier_idx[1][ci] # [y, x] order

            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[cy, cx] = 1 # indexing with [y, x] order
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)
            distances[:, k_i] = np_obstacle_map_distance[frontier_idx[0], frontier_idx[1]] # indexing with [y, x] order

        # find the nearest cluster for each frontier point
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for k_i in range(k):
            # Assigns each frontier point to the nearest cluster.
            nearest_clusters_k_i_idx = np.where(nearest_clusters == k_i)
            if not nearest_clusters_k_i_idx:
                continue
            
            nearest_clusters_k_i_idx = nearest_clusters_k_i_idx[0]

            if frontier_idx[0][nearest_clusters_k_i_idx].size == 0 or frontier_idx[1][nearest_clusters_k_i_idx].size == 0:
                continue

            # Updates cluster centers to the point with minimum average distance to all points in the cluster.
            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            # indexing with [y, x] order
            np_obstacle_map_frontierK[int(frontier_idx[0][nearest_clusters_k_i_idx].mean()), int(frontier_idx[1][nearest_clusters_k_i_idx].mean())] = 1
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)
            
            # is_frontier is already in [y, x] order
            temp_distance_k_i = np_obstacle_map_distance[is_frontier][nearest_clusters_k_i_idx]
            cluster_idx[k_i] = nearest_clusters_k_i_idx[np.argmin(temp_distance_k_i)]

        last_clusters = nearest_clusters

    return (frontier_idx[0][cluster_idx], frontier_idx[1][cluster_idx]), nearest_clusters, is_frontier, frontier_idx

class CoScanPlanner:
    def __init__(self):
        self.global_goals = None
        
    def get_long_term_goals(self, np_obstacle_map, np_frontier_map, agent_pos):
        '''
        @ note: agent_pos is in [x, y] order, so need to change the order when indexing to numpy array
        '''
        num_agent = agent_pos.shape[0]
        np_frontier_map = np.copy(np_frontier_map)
        np_obstacle_map_distance = []

        dd_mask = np.ones(np_obstacle_map.shape, dtype=bool)
        for i in range(num_agent):
            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1) # mask out obstacles (where value == 1) 
            np_obstacle_map_frontierK[agent_pos[i, 1], agent_pos[i, 0]] = 1 # mark agent position as 1 (now it's the only 1 value)
            dd = skfmm.distance(1 - np_obstacle_map_frontierK)
            np_obstacle_map_distance.append(dd) # store distance map for each agent
            # mask out unreachable areas
            if type(dd) is np.ndarray:
                dd_mask[:] = False
            else:
                dd_mask &= dd.mask
        np_frontier_map[dd_mask] = 0 # remove unreachable frontiers

        # cluster_center is in [y, x] order
        cluster_center, nearest_clusters, is_frontier, frontier_idx = kmeans(np_obstacle_map, np_frontier_map, num_agent)
        # nearest_clusters: (n_frontier,)
        if cluster_center is None:
            self.global_goals = [None] * num_agent
        else:
            nc = cluster_center[0].shape[0] # number of clusters
            cost = np.zeros((num_agent, nc))
            # n_agent x n_cluster
            for i in range(num_agent):
                # distance from agent i to cluster center j
                cost[i, :] = np_obstacle_map_distance[i][cluster_center[0], cluster_center[1]]

            # If there are more agents than clusters, it repeats the cost matrix
            cost = np.hstack([cost] * ((num_agent - 1) // nc + 1))

            self.global_goals = np.zeros((num_agent, 2), dtype=np.int32)
            row_ind, col_ind = linear_sum_assignment(cost) # use Hungarian algorithm to solve the assignment problem
            for i in range(num_agent):
                cluster_idx = col_ind[i] % nc
                agent_idx = row_ind[i]
                frontier_dist = np_obstacle_map_distance[agent_idx][is_frontier]
                frontier_dist[nearest_clusters != cluster_idx] = np.inf # only consider frontiers in the assigned cluster
                select = np.argmin(frontier_dist)
                # return goal in [x, y] order
                self.global_goals[agent_idx] = [frontier_idx[1][select], frontier_idx[0][select]]

        return self.global_goals

    def replan(self, np_obstacle_map, np_frontier_map, goal, planning_window):
        # goal: [x,y]
        gx1, gx2, gy1, gy2 = planning_window
        np_obstacle_map = np.ma.masked_values(np_obstacle_map, 1)
        np_obstacle_map[goal[1], goal[0]] = 1
        dd = skfmm.distance(1 - np_obstacle_map)[gy1:gy2, gx1:gx2]
        dd[np_frontier_map[gy1:gy2, gx1:gx2] == 0] = np.inf
        goal = np.unravel_index(np.argmin(dd), dd.shape)
        return goal[0], goal[1]

    def check_finish(self, np_frontier_map, stop):
        if len([1 for goal in self.global_goals if goal is not None]) == 1:
            self.global_goals = None
            return True
        for goal, stopped in zip(self.global_goals, stop):
            if goal is None:
                continue
            if stopped:
                stopped = False
                self.global_goals = None
                return True
            elif np_frontier_map[goal[1], goal[0]] == 0:
                self.global_goals = None
                return True
        return False