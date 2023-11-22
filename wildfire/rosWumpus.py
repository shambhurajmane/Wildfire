import math
import sys
import os
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd



class HolonomicNode:
    def __init__(self, x,y, parent):
        self.x=x
        self.y=y
        self.dist = 0 
        self.heuristic=0
        self.parent = parent
        
    def __lt__(self, other):
        return (self.dist + self.heuristic) < (other.dist + other.heuristic)



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
    for i in range(-10,10,1):
        try: 
            if obstacles[(neighbourNode.x) +i][(neighbourNode.y)+i] :
                return False
        except IndexError: 
            continue

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
            # print(currentNode.x,currentNode.y)
            path = []
            while currentNode:
                path.append((currentNode.x, currentNode.y))
                currentNode = currentNode.parent
                
            # print(path)
            return path[::-1]  # Reverse the path to get it from start to goal

        closedSet.add((currentNode.x, currentNode.y))
        # print(len(closedSet))
        
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


def run(s, g, mapParameters):

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
