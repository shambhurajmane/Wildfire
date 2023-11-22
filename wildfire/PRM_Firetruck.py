import math
import sys
import os
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
from scipy import spatial as sp
import random
import localplanner as lp

# from autocar_nav import reeds_shepp as rsCurve





def obstaclesMap(obstacleX, obstacleY, xyResolution):

    # Compute Grid Index for obstacles
    obstacleX = [round(x / xyResolution) for x in obstacleX]
    obstacleY = [round(y / xyResolution) for y in obstacleY]

    # Set all Grid locations to No Obstacle
    obstacles =[[False for i in range(max(obstacleY))]for i in range(max(obstacleX))]

    # Set Grid Locations with obstacles to True

    for i, j in zip(obstacleX, obstacleY): 
        obstacles[i][j] = True
        break

    return obstacles


def nodeIsValid(neighbourNode, obstacles, mapParameters):

    # Check if Node is out of map bounds
    if neighbourNode[0]<= mapParameters.mapMinX or \
       neighbourNode[0]>= mapParameters.mapMaxX or \
       neighbourNode[1]<= mapParameters.mapMinY or \
       neighbourNode[1]>= mapParameters.mapMaxY:
        return False

    # Check if Node on obstacle
    if obstacles[neighbourNode[0]][neighbourNode[1]]:
        return False

    return True

def PRM_path(start, goal, graph):
    openSet = []
    closedSet = set()

    heapq.heappush(openSet,(0, start, None) )
    
    counter=0
    while openSet:
        current_cost, current_node, parent = heapq.heappop(openSet)
        

        if abs(current_node[0]-goal[0])<6 and abs(current_node[1]-goal[1])<6:
            # print("found")
            path=[]
            while parent is not None:
                path.append(current_node)
                current_cost, current_node, parent = parent
            path.append(current_node)    
            # print(path)
            x=[]
            y=[]
            for i in range(len(path)-1):
                x.append(path[i][0])
                y.append(path[i][1])
            smoothed_x, smoothed_y = lp.smooth_path(x, y)
            max_curvature = 1.0  # You can adjust this threshold based on your robot's turning capabilities
            filtered_x, filtered_y = lp.filter_by_curvature(smoothed_x, smoothed_y, max_curvature)
            return filtered_x[::-1], filtered_y[::-1]  # Reverse the path to get it from start to goal
        
        if current_node in closedSet:
            continue
        
        closedSet.add(current_node)
        for neighbor in graph.get(current_node, []):
            
            if neighbor not in closedSet:
                neighbor_cost = current_cost + math.dist(current_node, neighbor) + math.dist(neighbor, goal)
                heapq.heappush(openSet, (neighbor_cost, neighbor, (neighbor_cost, neighbor, (current_cost, current_node, parent))))


           
    return None

def check_collision(p1, p2, obstacles):
    """Check if the path between two points collide with obstacles
    arguments:
        p1 - point 1, [row, col]
        p2 - point 2, [row, col]

    return:
        True if there are obstacles between two points
    """
    # ### YOUR CODE HERE ###
    # All points in between
    points = zip(np.linspace(p1[0], p2[0], dtype=int), np.linspace(p1[1], p2[1], dtype=int))

    # Check for obstacles
    for point in points:
        x, y = point[0], point[1]

        # Avoid getting too close to an obstacle as x and y are approximated to integers
        try:
            if (
                (obstacles[x][y + 1] == 0)
                or (obstacles[x][y - 1] == 0)
                or (obstacles[x + 1][y] == 0)
                or (obstacles[x - 1][y] == 0)
            ):
                return True
        except IndexError:
            # Handle IndexError when checking array boundaries
            continue

    return False

def uniform_sample(mapParameters, n_pts):
    """Use uniform sampling and store valid points
    arguments:
        n_pts - number of points try to sample,
                not the number of final sampled points

    check collision and append valid points to self.samples
    as [(row1, col1), (row2, col2), (row3, col3) ...]
    """
    # Initialize graph
    nodes_list=[]
    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)
    # Generate random points within the map boundaries
    random_rows = np.random.uniform(0, len(obstacles), n_pts)
    random_cols = np.random.uniform(0, len(obstacles), n_pts)

    for i in range(n_pts):
        sampleNode = (int(random_rows[i]),int(random_cols[i]))
        if nodeIsValid(sampleNode, obstacles, mapParameters):
            nodes_list.append(sampleNode)   
    
            
    return nodes_list

def build_graph(nodes_list,mapParameters):
    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)
    graph={}
    kd_tree = sp.KDTree(nodes_list)
    for i in range(len(nodes_list)):
        node = (nodes_list[i][0],nodes_list[i][1] )
        neighbors = kd_tree.query_ball_point(node, 5)
        for neighbor_index in neighbors:
            if i != neighbor_index and check_collision(node, (nodes_list[neighbor_index][0],nodes_list[neighbor_index][0]),obstacles):
                neighbor = nodes_list[neighbor_index]
                graph.setdefault(node, []).append(neighbor)
                graph.setdefault(neighbor, []).append(node)
    # print(graph)
    return graph



def drawCar(x, y, color='black'):
    angle = np.linspace( 0 , 2 * np.pi , 150 ) 
 
    radius = 0.4
    
    cx = radius * np.cos( angle ) 
    cy = radius * np.sin( angle ) 
    car = np.array([[-2, -2, 2, 2, -2],
                    [2 / 2, -2 / 2, -2 / 2, 2 / 2, 2 / 2]])

    # rotationZ = np.array([[math.cos(yaw), -math.sin(yaw)],
    #                  [math.sin(yaw), math.cos(yaw)]])
    # car = np.dot(rotationZ, car)
    car += np.array([[x], [y]])
    plt.plot(car[0, :], car[1, :], color)

def main():
    total_time=0
    burntBushes=[]
    intactBushes=[]
    extinuguishedBushes=[]
    # Set Start, Goal x, y, theta
    s = [10, 10, np.deg2rad(90)]
    
    # s = [10, 35, np.deg2rad(0)]
    # g = [30, 25, np.deg2rad(0)]

    # Get Obstacle Map
    percentage= 10
    map_size= 250
    obstacleX, obstacleY, bushX, bushY= map(map_size, percentage)

    # Calculate map Paramaters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 1, np.deg2rad(15.0))
    
    intactBushes=list(bushX.keys())
    # Compute Grid Index for start and Goal node
    s_x = round(s[0] / mapParameters.xyResolution)
    s_y = round(s[1] / mapParameters.xyResolution)

    

    # Create start and end Node ( index, cost, parent)
    
    
    nodes_list = uniform_sample(mapParameters,10000)
    s =(s_x,s_y)
    if s not in nodes_list:
        nodes_list.append(s)

    graph = build_graph(nodes_list,mapParameters)
    # Find Holonomic Heuristric

    
    
    while total_time<3600:
        # Run Hybrid A*
        bushIndex=random.choice(intactBushes)
        g = (bushX[bushIndex][1], bushY[bushIndex][1])
        
        burntBushes.append(bushIndex)
        
        path = PRM_path(s, g, graph)
        x=[]
        y=[]
        for i in range(len(path)-1):
            x.append(path[i][0])
            y.append(path[i][1])

        # Draw Animated Car
        for k in range(len(path)-1): 
            plt.cla()
            fig = plt.gcf()  # Get the current figure
            fig.set_facecolor('blue')  # Set the background color of the current figure
            fig.set_size_inches(10, 10)
            
            plt.xlim(min(obstacleX), max(obstacleX)) 
            plt.ylim(min(obstacleY), max(obstacleY))
            plt.plot(obstacleX, obstacleY, "go")
            plt.plot(g[0],g[1], "ro")
            for bushes in burntBushes:
                plt.plot(bushX[bushes],bushY[bushes],"ro")
            for bushes in extinuguishedBushes:
                plt.plot(bushX[bushes],bushY[bushes],"bo")
            plt.plot(x,y, linewidth=1.5, color='r', zorder=0)
            drawCar(x[k], y[k])
            # plt.plot(x[k], y[k], linewidth=0.3, color='g')
            # plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
            plt.title("Hybrid A*")
            plt.pause(0.001)
        # plt.show()
        # print(total_time)
        total_time+=len(path)*4
        extinuguishedBushes.append(bushIndex)
        intactBushes.remove(bushIndex)
        burntBushes.remove(bushIndex)
        s=(path[len(path)-1][0], path[len(path)-1][1])
    plt.show()    
        
    

if __name__ == '__main__':
    main()