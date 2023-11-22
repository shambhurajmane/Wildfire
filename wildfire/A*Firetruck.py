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
# from autocar_nav import reeds_shepp as rsCurve


class Car:
    maxSteerAngle = 0.6
    steerPresion = 5
    wheelBase = 2.58
    axleToFront = 3.0
    axleToBack = 0.4
    width = 2.0

class Cost:
    reverse = 10    
    directionChange = 150   #forward to reverse or revesre to forward
    steerAngle = 1
    steerAngleChange = 5
    hybridCost = 200

class Node:
    def __init__(self, gridIndex, traj, steeringAngle, direction, cost, parentIndex):
        self.gridIndex = gridIndex         # grid block x, y, yaw index
        self.traj = traj                   # trajectory x, y  of a simulated node
        self.steeringAngle = steeringAngle # steering angle throughout the trajectory
        self.direction = direction         # direction throughout the trajectory
        self.cost = cost                   # node cost
        self.parentIndex = parentIndex     # parent node index

class HolonomicNode:
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex
        self.cost = cost
        self.parentIndex = parentIndex

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


def index(Node):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([Node.gridIndex[0], Node.gridIndex[1], Node.gridIndex[2]])

def motionCommands():

    # Motion commands for a Non-Holonomic Robot like a Car or Bicycle (Trajectories using Steer Angle and Direction)
    direction = 1
    motionCommand = []
    for i in np.arange(Car.maxSteerAngle, -(Car.maxSteerAngle), -Car.maxSteerAngle/Car.steerPresion):
        motionCommand.append([i, direction])
        motionCommand.append([i, -direction])
    return motionCommand

def holonomicMotionCommands():

    # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
    holonomicMotionCommand = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return holonomicMotionCommand


def kinematicSimulationNode(currentNode, motionCommand, mapParameters, simulationLength=7, step = 1 ):

    # Simulate node using given current Node and Motion Commands
    traj = []
    angle = pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))  # to convert larger angles to [-2pi,2pi] by subtracting or adding 2pi iteratively
    traj.append([currentNode.traj[-1][0] + motionCommand[1] * step * math.cos(angle),
                currentNode.traj[-1][1] + motionCommand[1] * step * math.sin(angle),
                pi_2_pi(angle + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))])
    for i in range(int((simulationLength/step))-1):
        traj.append([traj[i][0] + motionCommand[1] * step * math.cos(traj[i][2]),
                    traj[i][1] + motionCommand[1] * step * math.sin(traj[i][2]),
                    pi_2_pi(traj[i][2] + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))])

    # Find grid index
    gridIndex = [round(traj[-1][0]/mapParameters.xyResolution), \
                 round(traj[-1][1]/mapParameters.xyResolution), \
                 round(traj[-1][2]/mapParameters.yawResolution)]
    # print(traj, gridIndex, mapParameters)
    # Check if node is valid
    if not isValid(traj, gridIndex, mapParameters):
        
        return None

    # Calculate Cost of the node
    cost = simulatedPathCost(currentNode, motionCommand, simulationLength)

    return Node(gridIndex, traj, motionCommand[0], motionCommand[1], cost, index(currentNode))



def isValid(traj, gridIndex, mapParameters):

    # Check if Node is out of map bounds
    if gridIndex[0]<=mapParameters.mapMinX or gridIndex[0]>=mapParameters.mapMaxX or \
       gridIndex[1]<=mapParameters.mapMinY or gridIndex[1]>=mapParameters.mapMaxY:
        return False

    # Check if Node is colliding with an obstacle
    if collision(traj, mapParameters):
        # print("not valid")
        return False
    return True

def collision(traj, mapParameters):

    carRadius = (Car.axleToFront + Car.axleToBack)/2 + 1
    dl = (Car.axleToFront - Car.axleToBack)/2
    for i in traj:
        cx = i[0] + dl * math.cos(i[2])
        cy = i[1] + dl * math.sin(i[2])
        pointsInObstacle = mapParameters.ObstacleKDTree.query_ball_point([cx, cy], carRadius)

        if not pointsInObstacle:
            continue

        for p in pointsInObstacle:
            xo = mapParameters.obstacleX[p] - cx
            yo = mapParameters.obstacleY[p] - cy
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])

            if abs(dx) < carRadius and abs(dy) < Car.width / 2 + 1:
                return True

    return False



def simulatedPathCost(currentNode, motionCommand, simulationLength):

    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    if motionCommand[1] == 1:
        cost += simulationLength 
    else:
        cost += simulationLength * Cost.reverse

    # Direction change cost
    if currentNode.direction != motionCommand[1]:
        cost += Cost.directionChange

    # Steering Angle Cost
    cost += motionCommand[0] * Cost.steerAngle

    # Steering Angle change cost
    cost += abs(motionCommand[0] - currentNode.steeringAngle) * Cost.steerAngleChange

    return cost

def eucledianCost(holonomicMotionCommand):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

def holonomicNodeIndex(HolonomicNode):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([HolonomicNode.gridIndex[0], HolonomicNode.gridIndex[1]])

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
    if neighbourNode.gridIndex[0]<= mapParameters.mapMinX or \
       neighbourNode.gridIndex[0]>= mapParameters.mapMaxX or \
       neighbourNode.gridIndex[1]<= mapParameters.mapMinY or \
       neighbourNode.gridIndex[1]>= mapParameters.mapMaxY:
        return False

    # Check if Node on obstacle
    if obstacles[neighbourNode.gridIndex[0]][neighbourNode.gridIndex[1]]:
        return False

    return True

def holonomicCostsWithObstacles(goalNode, mapParameters):

    gridIndex = [round(goalNode.traj[-1][0]/mapParameters.xyResolution), round(goalNode.traj[-1][1]/mapParameters.xyResolution)]
    gNode =HolonomicNode(gridIndex, 0, tuple(gridIndex))

    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)

    holonomicMotionCommand = holonomicMotionCommands()

    openSet = {holonomicNodeIndex(gNode): gNode}
    closedSet = {}

    priorityQueue =[]
    heapq.heappush(priorityQueue, (gNode.cost, holonomicNodeIndex(gNode)))

    while True:
        if not openSet:
            break

        _, currentNodeIndex = heapq.heappop(priorityQueue)
        currentNode = openSet[currentNodeIndex]
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        for i in range(len(holonomicMotionCommand)):
            neighbourNode = HolonomicNode([currentNode.gridIndex[0] + holonomicMotionCommand[i][0],\
                                      currentNode.gridIndex[1] + holonomicMotionCommand[i][1]],\
                                      currentNode.cost + eucledianCost(holonomicMotionCommand[i]), currentNodeIndex)

            if not holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):
                continue

            neighbourNodeIndex = holonomicNodeIndex(neighbourNode)

            if neighbourNodeIndex not in closedSet:            
                if neighbourNodeIndex in openSet:
                    if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                        openSet[neighbourNodeIndex].cost = neighbourNode.cost
                        openSet[neighbourNodeIndex].parentIndex = neighbourNode.parentIndex
                        # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                else:
                    openSet[neighbourNodeIndex] = neighbourNode
                    heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))

    holonomicCost = [[np.inf for i in range(max(mapParameters.obstacleY))]for i in range(max(mapParameters.obstacleX))]

    for nodes in closedSet.values():
        holonomicCost[nodes.gridIndex[0]][nodes.gridIndex[1]]=nodes.cost

    return holonomicCost

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

def backtrack(startNode, goalNode, closedSet, plt):

    # Goal Node data
    # print(closedSet[-1])
    startNodeIndex= index(startNode)
    currentNodeIndex = list(closedSet)[-1]#goalNode.parentIndex
    currentNode = closedSet[currentNodeIndex]
    x=[]
    y=[]
    yaw=[]
    # steering_angle = []
    # velocity =[]

    # Iterate till we reach start node from goal node
    while currentNodeIndex != startNodeIndex:
        a, b, c = zip(*currentNode.traj)
        x += a[::-1] 
        y += b[::-1] 
        yaw += c[::-1]
        
        # steering_angle.append(currentNode.steeringAngle)
        # velocity.append(currentNode.direction)
        
        
        currentNodeIndex = currentNode.parentIndex
        currentNode = closedSet[currentNodeIndex]
    return x[::-1], y[::-1], yaw[::-1] 

def run(s, g, mapParameters, plt):

    # Compute Grid Index for start and Goal node
    sGridIndex = [round(s[0] / mapParameters.xyResolution), \
                  round(s[1] / mapParameters.xyResolution), \
                  round(s[2]/mapParameters.yawResolution)]
    gGridIndex = [round(g[0] / mapParameters.xyResolution), \
                  round(g[1] / mapParameters.xyResolution), \
                  round(g[2]/mapParameters.yawResolution)]

    # Generate all Possible motion commands to car
    motionCommand = motionCommands()

    # Create start and end Node
    startNode = Node(sGridIndex, [s], 0, 1, 0 , tuple(sGridIndex))
    goalNode = Node(gGridIndex, [g], 0, 1, 0, tuple(gGridIndex))

    # Find Holonomic Heuristric
    holonomicHeuristics = holonomicCostsWithObstacles(goalNode, mapParameters)

    # Add start node to open Set
    openSet = {index(startNode):startNode}
    
    closedSet = {}

    # Create a priority queue for acquiring nodes based on their cost's
    costQueue = heapdict()

    # Add start mode into priority queue
    costQueue[index(startNode)] = max(startNode.cost , Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]])
    counter = 0
    
    # Run loop while path is found or open set is empty
    while True:
        counter +=1
        # Check if openSet is empty, if empty no solution available
        
        if not openSet:
            print("path not found")
            break

        # Get first node in the priority queue
        currentNodeIndex = costQueue.popitem()[0]
        currentNode = openSet[currentNodeIndex]

        # Revove currentNode from openSet and add it to closedSet
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        # print(currentNodeIndex[0])
        if abs(currentNodeIndex[0] - index(goalNode)[0])<11 and abs(currentNodeIndex[1] - index(goalNode)[1])<11 :
            print("Path Found")
            # print(currentNode.traj[-1])
            break
        
        # Get all simulated Nodes from current node
        for i in range(len(motionCommand)):
            simulatedNode = kinematicSimulationNode(currentNode, motionCommand[i], mapParameters)
          
            # Check if path is within map bounds and is collision free
            if not simulatedNode:
                # print("ss")
                continue

            # Draw Simulated Node
            x,y,z =zip(*simulatedNode.traj)
            plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            simulatedNodeIndex = index(simulatedNode)
            
            if simulatedNodeIndex not in closedSet: 
                
                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulatedNodeIndex not in openSet:
                    openSet[simulatedNodeIndex] = simulatedNode
                    costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
                else:
                    if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
                        openSet[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
     
    # Backtrack
    # print((startNode, goalNode, closedSet, plt))
    x, y, yaw = backtrack(startNode, goalNode, closedSet, plt)

    return x, y, yaw

def drawCar(x, y, yaw, color='black'):
    car = np.array([[-Car.axleToBack, -Car.axleToBack, Car.axleToFront, Car.axleToFront, -Car.axleToBack],
                    [Car.width / 2, -Car.width / 2, -Car.width / 2, Car.width / 2, Car.width / 2]])

    rotationZ = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
    car = np.dot(rotationZ, car)
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
    # g = [22, 28, np.deg2rad(0)]

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
        
        x, y, yaw = run(s, g, mapParameters, plt)
        # print(x)

        # Draw Animated Car
        for k in range(len(x)): 
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
            plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
            drawCar(x[k], y[k], yaw[k])
            plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
            plt.title("Hybrid A*")
            plt.pause(0.001)
        # plt.show()
        print(total_time)
        total_time+=len(x)*4
        extinuguishedBushes.append(bushIndex)
        intactBushes.remove(bushIndex)
        burntBushes.remove(bushIndex)
        s=[x[len(x)-1], y[len(x)-1], yaw[len(x)-1]]
    plt.show()    
        
    

if __name__ == '__main__':
    main()