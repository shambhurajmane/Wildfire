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
import ROSPRM_Firetruck as ft 
import rosWumpus as wp
# from autocar_nav import reeds_shepp as rsCurve


class Car:
    maxSteerAngle = 0.6
    steerPresion = 5
    wheelBase = 3.0
    axleToFront = 4.0
    axleToBack = 0.9
    width = 2.2

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


def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):
        
        # calculate min max map grid index based on obstacles in map
        mapMinX = round(min(obstacleX) / xyResolution)
        mapMinY = round(min(obstacleY) / xyResolution)
        mapMaxX = round(max(obstacleX) / xyResolution)
        mapMaxY = round(max(obstacleY) / xyResolution)

        # create a KDTree to represent obstacles
        ObstacleKDTree = sp.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])

        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  

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



# def wumpus(x, y, color='black'):
#     angle = np.linspace( 0 , 2 * np.pi , 150 ) 
 
#     radius = 0.4
    
#     cx = radius * np.cos( angle ) 
#     cy = radius * np.sin( angle ) 
#     wumpus = np.array([[-2, -2, 2, 2, -2],
#                     [2 / 2, -2 / 2, -2 / 2, 2 / 2, 2 / 2]])

#     angle = np.linspace( 0 , 2 * np.pi , 150 ) 
 
#     wumpus += np.array([[cx], [cy]])
#     plt.plot(wumpus[0, :], wumpus[1, :], color)
    
    
def fireTruck(x, y, color='black'):
    angle = np.linspace( 0 , 2 * np.pi , 150 ) 
 
    radius = 0.4
    
    cx = radius * np.cos( angle ) 
    cy = radius * np.sin( angle ) 
    fireTruck = np.array([[-Car.axleToBack, -Car.axleToBack, Car.axleToFront, Car.axleToFront, -Car.axleToBack],
                    [Car.width / 2, -Car.width / 2, -Car.width / 2, Car.width / 2, Car.width / 2]])

    # rotationZ = np.array([[math.cos(yaw), -math.sin(yaw)],
    #                  [math.sin(yaw), math.cos(yaw)]])
    # fireTruck = np.dot(rotationZ, fireTruck)
    fireTruck += np.array([[x], [y]])
    plt.plot(fireTruck[0, :], fireTruck[1, :], color)




def ratio_graph(intact,burnt,extinguished):
    
    # set width of bar 
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8)) 
    
    
    # Set position of bar on X axis 
    br1 = np.arange(len(intact)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    
    # Make the plot
    plt.bar(br1, intact, color ='r', width = barWidth, 
            edgecolor ='grey', label ='intact') 
    plt.bar(br2, burnt, color ='g', width = barWidth, 
            edgecolor ='grey', label ='burnt') 
    plt.bar(br3, extinguished, color ='b', width = barWidth, 
            edgecolor ='grey', label ='extinguished') 
    
    # Adding Xticks 
    plt.xlabel('Iteration', fontweight ='bold', fontsize = 15) 
    plt.ylabel('ratio with total bushes', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(5)], 
            ['iteration 1', 'iteration 2', 'iteration 3', 'iteration 4', 'iteration 5'])
    
    plt.legend()
    plt.show() 
    
def com_graph(prm,astar):
    
    # set width of bar 
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8)) 
    
    # Set position of bar on X axis 
    br1 = np.arange(len(prm)) 
    br2 = [x + barWidth for x in br1] 
    
    # Make the plot
    plt.bar(br1, prm, color ='r', width = barWidth, 
            edgecolor ='grey', label ='prm computational time') 
    plt.bar(br2, astar, color ='g', width = barWidth, 
            edgecolor ='grey', label ='astar computational time') 

    
    # Adding Xticks 
    plt.xlabel('Iteration', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Algorithm computational time', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(5)], 
            ['iteration 1', 'iteration 2', 'iteration 3', 'iteration 4', 'iteration 5'])
    
    plt.legend()
    plt.show()


def main():
    intact=[]
    burnt=[]
    ext=[]
    
    prm_com=[]
    astr_com=[]
    for i in range(5):
        # initialization
        total_time=0
        burntBushes=[]
        intactBushes=[]
        extinuguishedBushes=[]
        prm_time=0
        astar_time=0

        #  map setup
        percentage= 10
        map_size= 250
        obstacleX, obstacleY, bushX, bushY= map(map_size, percentage)

        mapParameters = calculateMapParameters(obstacleX, obstacleY, 1, np.deg2rad(15.0))
        
        intactBushes=list(bushX.keys())
        
        
        
        
        # Initial wumpus and firetruck location
        iw_x=[10]
        iw_y=[10]
        if_x=[240]
        if_y=[240]
        
        # s_x = round(s[0] / mapParameters.xyResolution)
        # s_y = round(s[1] / mapParameters.xyResolution)
        
        #PRM sampling of free nodes and building graph
        start = time.time()


        nodes_list = ft.uniform_sample(mapParameters,10000)
        wumpus_s =(iw_x[0],iw_y[0])
        firetruck_s =(if_x[0],if_y[0])
        if firetruck_s not in nodes_list:
            nodes_list.append(firetruck_s)
        
        for i in intactBushes:
            nodes_list.append((bushX[i][1], bushY[i][1]))
            
        graph = ft.build_graph(nodes_list,mapParameters)
        end = time.time()
        print(end - start)
        astar_counter=0
      
        #simulation
        while total_time<3000:
            bushIndex=random.choice(intactBushes)
            wumpus_g = (bushX[bushIndex][1], bushY[bushIndex][1])
            start = time.time()               
            w_path = wp.run(wumpus_s, wumpus_g, mapParameters)
            end = time.time()
            astar_counter+=1
            astar_time+=(end - start)
            x=[]
            y=[]
            for i in range(len(w_path)-1):
                x.append(w_path[i][0])
                y.append(w_path[i][1])

            # Draw Animated Car
            for k in range(len(w_path)-1): 
                plt.cla()
                
                fig = plt.gcf()  # Get the current figure
                fig.set_facecolor('blue')  # Set the background color of the current figure
                fig.set_size_inches(10, 10)
                
                plt.xlim(min(obstacleX), max(obstacleX)) 
                plt.ylim(min(obstacleY), max(obstacleY))
                plt.plot(obstacleX, obstacleY, "go")
                plt.plot(wumpus_g[0],wumpus_g[1], "ro")
                for bushes in burntBushes:
                    plt.plot(bushX[bushes],bushY[bushes],"ro")
                for bushes in extinuguishedBushes:
                    plt.plot(bushX[bushes],bushY[bushes],"bo")
                plt.plot(x,y, linewidth=1.5, color='r', zorder=0)
                plt.plot(x[k], y[k], "ko")
                fireTruck(if_x[0],if_y[0])
                # plt.plot(x[k], y[k], linewidth=0.3, color='g')
                # plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
                plt.title("Hybrid A*")
                plt.pause(0.01)
            # plt.show()
            print(total_time)
            total_time+=len(w_path)*3
            burntBushes.append(bushIndex)
            intactBushes.remove(bushIndex)
            wumpus_s=[w_path[len(w_path)-1][0], w_path[len(w_path)-1][1]]

        w_reached=True
        f_reached=True
        total_time =0 
        prm_counter=0
        
        while total_time<1000:
            # Run Hybrid A*
            try: 
                if w_reached:
                    print("wumpus changing target")
                    wbushIndex=random.choice(intactBushes)
                    wumpus_g = (bushX[wbushIndex][1], bushY[wbushIndex][1])
                    start = time.time()
                    w_path = wp.run(wumpus_s, wumpus_g, mapParameters)

                    end = time.time()
                    astar_time+=(end - start)
                    astar_counter+=1
    
                    
                    w_x=[]
                    w_y=[]
                    for i in range(len(w_path)-1):
                        w_x.append(w_path[i][0])
                        w_y.append(w_path[i][1])
                        
                    print(len(w_path))
                    
                    w_reached = False
                    
                if f_reached:
                    print("firetruck changing target")
                    fbushIndex = random.choice(burntBushes)
                    firetruck_g = (bushX[fbushIndex][1], bushY[fbushIndex][1])
                    # print(len(graph))
                    start = time.time()
                    f_path = ft.PRM_path(firetruck_s, firetruck_g, graph)
                    end = time.time()
                    prm_time+=(end - start)
                    prm_counter+=1
                    
                    # print( "path is ",f_path)
                    f_x=[]
                    f_y=[]
                    for i in range(len(f_path)-1):
                        f_x.append(f_path[i][0])
                        f_y.append(f_path[i][1])
                    print(len(f_path))
                    
                    f_reached = False
            except IndexError:
                break 
            
    
        
            # Draw Animated Car
            for k in range(1500): 
                plt.cla()
                
                fig = plt.gcf()  # Get the current figure
                fig.set_facecolor('blue')  # Set the background color of the current figure
                fig.set_size_inches(10, 10)
                
                plt.xlim(min(obstacleX), max(obstacleX)) 
                plt.ylim(min(obstacleY), max(obstacleY))
                plt.plot(obstacleX, obstacleY, "go")
                plt.plot(firetruck_g[0],firetruck_g[1], "ro")
                for bushes in burntBushes:
                    plt.plot(bushX[bushes],bushY[bushes],"ro")
                for bushes in extinuguishedBushes:
                    plt.plot(bushX[bushes],bushY[bushes],"bo")
                plt.plot(f_x,f_y, linewidth=1.5, color='b', zorder=0)
                plt.plot(w_x,w_y, linewidth=1.5, color='r', zorder=0)
                
                
                if k == (len(f_x)-1):
                    print("firetruck reached")
                    f_reached = True
                    total_time+=len(f_x)*3
                    extinuguishedBushes.append(fbushIndex)
                    burntBushes.remove(fbushIndex)
                    firetruck_s=(f_path[len(f_path)-1][0], f_path[len(f_path)-1][1])
                    w_x = w_x[k:]
                    w_y = w_y[k:]
                    break
            
                if k == (len(w_x)-1):
                    print("wumpus reached")
                    w_reached = True
                
                    burntBushes.append(wbushIndex)
                    intactBushes.remove(wbushIndex)
                    wumpus_s=[w_path[len(w_path)-1][0], w_path[len(w_path)-1][1]]
                    f_x = f_x[k:]
                    f_y = f_y[k:]
                    break
                
                plt.plot(w_x[k], w_y[k], "ko")
                fireTruck(f_x[k], f_y[k])
                    
                # plt.plot(x[k], y[k], linewidth=0.3, color='g')
                plt.title("Hybrid A*")
                plt.pause(0.01)
            # plt.show()
        total= len(intactBushes) + len(burntBushes) + len(extinuguishedBushes)
        
        intact.append(len(intactBushes)/total)
        burnt.append(len(burntBushes)/total)
        ext.append(len(extinuguishedBushes)/ total)
        
        prm_com.append(prm_time/prm_counter)
        astr_com.append(astar_time/astar_counter)
        
        
        
        plt.show()   
        ratio_graph(intact,burnt, ext)
        com_graph(prm_com,astr_com)
        
     
        
    

if __name__ == '__main__':
    main()