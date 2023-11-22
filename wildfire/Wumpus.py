import math
import sys
import os
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd
import random
import rclpy
from rclpy.node import Node

from std_msgs.msg import String






class HolonomicNode:
    def __init__(self, x,y, parent):
        self.x=x
        self.y=y
        self.dist = 0 
        self.heuristic=0
        self.parent = parent
        
    def __lt__(self, other):
        return (self.dist + self.heuristic) < (other.dist + other.heuristic)

class MapParameters:
    def __init__(self, mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY):
        self.mapMinX = mapMinX               # map min x coordinate(0)
        self.mapMinY = mapMinY               # map min y coordinate(0)
        self.mapMaxX = mapMaxX               # map max x coordinate
        self.mapMaxY = mapMaxY               # map max y coordinate
        self.xyResolution = xyResolution     # grid block length
        self.yawResolution = yawResolution   # grid block possible yaws
        self.ObstacleKDTree = ObstacleKDTree # KDTree representating obstacles
        self.obstacleX = obstacleX           # Obstacle x coordinate list
        self.obstacleY = obstacleY           # Obstacle y coordinate list

def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):
        
        # calculate min max map grid index based on obstacles in map
        mapMinX = round(min(obstacleX) / xyResolution)
        mapMinY = round(min(obstacleY) / xyResolution)
        mapMaxX = round(max(obstacleX) / xyResolution)
        mapMaxY = round(max(obstacleY) / xyResolution)

        # create a KDTree to represent obstacles
        ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])

        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  


def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta



def holonomicMotionCommands():

    # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
    holonomicMotionCommand = [(0, -1), (0, 1), (-1, 0), (1, 0),(1, 1),(-1, 1),(1, -1),(-1, -1)]
    return holonomicMotionCommand


def eucledianCost(holonomicMotionCommand):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

def heuristicCost(node,goal):
    # Compute Eucledian Distance between two nodes
   
    return abs(node.x - goal.x) + abs(node.y - goal.y)



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

def holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):

    # Check if Node is out of map bounds
    if neighbourNode.x<= mapParameters.mapMinX or \
       neighbourNode.x>= mapParameters.mapMaxX or \
       neighbourNode.y<= mapParameters.mapMinY or \
       neighbourNode.y>= mapParameters.mapMaxY:
        return False

    # Check if Node on obstacle
    if obstacles[neighbourNode.x][neighbourNode.y]:
        return False

    return True

def holonomicCostsWithObstacles(startNode, goalNode, mapParameters):
    openSet = []
    closedSet = set()

    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)
    holonomicMotionCommand = holonomicMotionCommands()

    
    heapq.heappush(openSet,startNode )

    while openSet:
        currentNode = heapq.heappop(openSet)

        if currentNode.x == goalNode.x and currentNode.y == goalNode.y:
            print(currentNode.x,currentNode.y)
            path = []
            while currentNode:
                path.append((currentNode.x, currentNode.y))
                currentNode = currentNode.parent
                
            print(path)
            return path[::-1]  # Reverse the path to get it from start to goal

        closedSet.add((currentNode.x, currentNode.y))
        print(len(closedSet))
        
        for i,j in holonomicMotionCommand:
            neighbourNode = HolonomicNode(currentNode.x + i,\
                                      currentNode.y + j,\
                                      currentNode)

            if not holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):
                continue
            
            

            neighbourNodeIndex = (neighbourNode.x, neighbourNode.y)
            dist_cost = currentNode.dist + 1
            if neighbourNodeIndex in closedSet:
                continue
                        
            if neighbourNode not in openSet or neighbourNode.dist > dist_cost:
                neighbourNode.dist = dist_cost
                neighbourNode.heuristic = heuristicCost(neighbourNode,goalNode)
                neighbourNode.parentIndex = currentNode
                if neighbourNode not in openSet:
                    heapq.heappush(openSet , neighbourNode )
           
    return None

def map(map_size, percentage):
    # Build Map
    obstacleX, obstacleY = [], []
    bushX={}
    bushY={}
    

    for i in range(map_size):
        obstacleX.append(i)
        obstacleY.append(0)

    for i in range(map_size):
        obstacleX.append(0)
        obstacleY.append(i)

    for i in range(map_size):
        obstacleX.append(i)
        obstacleY.append(map_size-1)

    for i in range(map_size):
        obstacleX.append(map_size-1)
        obstacleY.append(i)
    
    # for i in range(10,20):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(30,51):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(0,31):
    #     obstacleX.append(20)
    #     obstacleY.append(i) 

    # for i in range(0,31):
    #     obstacleX.append(30)
    #     obstacleY.append(i) 

    # for i in range(40,50):
    #     obstacleX.append(15)
    #     obstacleY.append(i)

    # for i in range(25,40):
    #     obstacleX.append(i)
    #     obstacleY.append(35)
    total_tenderion= int(map_size * percentage *0.01)
    
    for i in range(total_tenderion):
        j=random.randint(1, 6)
        k=random.randint(0,(map_size))  #row
        l=random.randint(0,(map_size))  #column
        #print(j,k)
        

        if j==1 and k<(map_size-2) and l<(map_size-3):
            obstacleX.append(k)
            obstacleY.append(l)
            obstacleX.append(k)
            obstacleY.append(l+1)
            obstacleX.append(k+1)
            obstacleY.append(l+1)
            obstacleX.append(k+1)
            obstacleY.append(l+2)
            bushX[i]=[k,k,k+1,k+1]
            bushY[i]=[l,l+1,l+1,l+2]
            

        elif j==2 and k<(map_size-3) and l<(map_size-2):
            obstacleX.append(k)
            obstacleY.append(l)
            obstacleX.append(k+1)
            obstacleY.append(l)
            obstacleX.append(k+1)
            obstacleY.append(l+1)
            obstacleX.append(k+2)
            obstacleY.append(l+1)
            bushX[i]=[k,k+1,k+1,k+2]
            bushY[i]=[l,l,l+1,l+1]
            
        elif j==3 and k<(map_size-2) and l<(map_size-3):
            obstacleX.append(k)
            obstacleY.append(l)
            obstacleX.append(k+1)
            obstacleY.append(l)
            obstacleX.append(k+1)
            obstacleY.append(l+1)
            obstacleX.append(k+1)
            obstacleY.append(l+2)
            bushX[i]=[k,k+1,k+1,k+1]
            bushY[i]=[l,l,l+1,l+2]
        
        elif j==4 and k<(map_size-2) and l<(map_size-3):
            obstacleX.append(k)
            obstacleY.append(l+2)
            obstacleX.append(k+1)
            obstacleY.append(l)
            obstacleX.append(k+1)
            obstacleY.append(l+1)
            obstacleX.append(k+1)
            obstacleY.append(l+2)
            bushX[i]=[k,k+1,k+1,k+1]
            bushY[i]=[l+2,l,l+1,l+2]
            
        elif j==5 and k<(map_size-3) and l<(map_size-2):
            obstacleX.append(k)
            obstacleY.append(l)
            obstacleX.append(k)
            obstacleY.append(l+1)
            obstacleX.append(k+1)
            obstacleY.append(l+1)
            obstacleX.append(k+2)
            obstacleY.append(l+1)
            bushX[i]=[k,k,k+1,k+2]
            bushY[i]=[l,l+1,l+1,l+1]
            
        elif j==6 and k<(map_size-2) and l<(map_size-3):
            obstacleX.append(k)
            obstacleY.append(l)
            obstacleX.append(k+1)
            obstacleY.append(l)
            obstacleX.append(k+2)
            obstacleY.append(l)
            obstacleX.append(k+2)
            obstacleY.append(l+1)
            bushX[i]=[k,k+1,k+2,k+2]
            bushY[i]=[l,l,l,l+1]
        
        else:
            i=i-1

    return obstacleX, obstacleY, bushX, bushY


def run(s, g, mapParameters, plt):

    # Compute Grid Index for start and Goal node
    s_x = round(s[0] / mapParameters.xyResolution)
    s_y = round(s[1] / mapParameters.xyResolution)
    g_x = round(g[0] / mapParameters.xyResolution)
    g_y = round(g[1] / mapParameters.xyResolution)
    

    # Create start and end Node ( index, cost, parent)
    startNode = HolonomicNode(s_x,s_y , None)
    goalNode = HolonomicNode(g_x, g_y, None)

    # Find Holonomic Heuristric
    path = holonomicCostsWithObstacles(startNode,goalNode, mapParameters)

    # Backtrack
    # print((startNode, goalNode, closedSet, plt))
    
    

    return path

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

    # Get Obstacle Map
    percentage= 10
    map_size= 250
    obstacleX, obstacleY, bushX, bushY= map(map_size, percentage)

    # Calculate map Paramaters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 1, np.deg2rad(15.0))
    
    intactBushes=list(bushX.keys())

    
    while total_time<3600:
        # Run Hybrid A*
        bushIndex=random.choice(intactBushes)
        g = [bushX[bushIndex][1], bushY[bushIndex][1], np.deg2rad(180)]
        burntBushes.append(bushIndex)
        
        path = run(s, g, mapParameters, plt)
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
        print(total_time)
        total_time+=len(path)*4
        extinuguishedBushes.append(bushIndex)
        intactBushes.remove(bushIndex)
        burntBushes.remove(bushIndex)
        s=[path[len(path)-1][0], path[len(path)-1][1]]
    plt.show()    
        
    



if __name__ == '__main__':
    main()