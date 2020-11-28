import os
import sys
import time
import serial
import numpy as np
from numpy import linalg as LNG 
import math
import matplotlib.pyplot as plt
from matplotlib import colors

#class used for path planning
#for each map where pathplanning should be done an object of this class needs to be created
class pathPlaning: 
   #method that returns the path generated as a numpy arra with [xxxxx]newline[yyyyyyy]. 
   #(only returns a proper path after setGoal and setStart where set once, because otherwise start and goal are 0,0)
    def getPath(self):#returns  the path with grid box "accuracy" so a point for every gridbox
        if(len(self.__path)):
            path=[tuple([coordinate* float(self.__CMPerPixel) for coordinate in point]) for point in self.__path]
            path=np.array(path)
            path = np.array(path).reshape(-1, 2).transpose()
            path=np.flip(path,0)
        return path
    #method that returns a reduced path, where only the points, where the path changes direction, are returned.
    #(only returns a proper path when setGoal and setStart where set atleast once, because otherwise start and goal are 0,0)
    def getOptimizedPath(self):     #returns only the  points where the calculated path changes direction
        if(len(self.__path)):
            originalPath=np.array(self.__path)
            originalPath = np.array(originalPath).reshape(-1, 2).transpose()
            edgePoints = [tuple(originalPath[:,0])]
            movmentOld=np.array([0, 0])
            movmentNew=np.array([0, 0])

            for pointNumber in range(1,originalPath.shape[1]):              #go through the path and only retain the points after where a different movment is needed
                movmentNew=originalPath[:,pointNumber]-originalPath[:,pointNumber-1]
                if(abs(movmentNew[0]-movmentOld[0])>self.__tolerance or abs(movmentNew[1]-movmentOld[1])>self.__tolerance):
                    if(tuple(originalPath[:,pointNumber-1]) not in edgePoints):
                        edgePoints.append(tuple(originalPath[:,pointNumber-1]))
                    
                movmentOld=movmentNew
            edgePoints.append(tuple(originalPath[:,-1]))
            optimalPath =[tuple([coordinate* float(self.__CMPerPixel) for coordinate in point]) for point in edgePoints]    #transform pixels into cms
            optimalPath = np.array(optimalPath)                             #convert list of tuples into np array with [yyyyyyyyyyy] newline [xxxxxxxxxxx]
            optimalPath = np.array(optimalPath).reshape(-1, 2).transpose()
            optimalPath = np.flip(optimalPath,0)                                    #make sure [xxxxxxxxxxx]newline [yyyyyyyyyy]
        return optimalPath
    #method to set the starting point, calls the internal path generation method to generate apath that then can be gotten byGetPath or getOptimizedPath
    #path generation fails when setGoal was not called once before
    def setStart(self,start):
        start=np.flip(start)                                            #flip xy for more easy use inside the different functions.
        self.__startP=np.around(np.divide(start,self.__CMPerPixel))
        self.__startP=self.__startP.astype(int)
        if(self.__startP[0]>=self.__mapDimensions[0] or self.__startP[1]>=self.__mapDimensions[1] or self.__startP[0]<0 or self.__startP[1]<0):
            raise Exception('Start outside Map')
        if(self.__map[self.__startP[0], self.__startP[1]]==self.__occupancyMarkerMap):
            raise Exception('Start node is not traversable')
        self.__generatePath()
        
    #method to set the goal in a map. When this method is called it calls the internal generateGradient method. This will take some time depending on the map size. 
    #Recommended to not change the goal often.
    def setGoal(self,goal):
        goal=np.flip(goal)                                              #flip x and y to y and x
        if(LNG.norm(np.subtract(self.__goalP,np.around(np.divide(goal,self.__CMPerPixel))))>self.__tolerance):      #only replan if goal has changed
            self.__goalP=np.around(np.divide(goal,self.__CMPerPixel))
            self.__goalP=self.__goalP.astype(int)
            if(self.__goalP[0]>=self.__mapDimensions[0] or self.__goalP[1]>=self.__mapDimensions[1] or self.__goalP[0]<0 or self.__goalP[1]<0):
                raise Exception('Goal outside Map')
            if(self.__map[self.__goalP[0], self.__goalP[1]]==self.__occupancyMarkerMap):
                raise Exception('goal node is not traversable')
            self.__generateGradient()
        else:
            self.__goalP=self.__goalP.astype(int)
			
    def getDistanceMap(self):                   #returns the so called gradientMap. The robot just"rolls" down the gradient to get to the goal
                                                #the gradient map is a np array containing in each cell that is not occupied the distance to the goal
        return self.__gradientMap               #the occupied cells are not marked in this map, they just stay at 0 distance!
        
        
        
    #initlializing the pathplanning process with a map and map related parameters, like whwat is used to mark the occupied cells and how many cm are per gridbox(pixel)
    def __init__(self, occupancyGrid,occupancyMarker,CMPerPixel):
        self.__map=occupancyGrid
        self.__occupancyMarkerMap=int(occupancyMarker)
        self.__gradientMap=np.zeros_like(occupancyGrid)     #empty gradient map
        self.__CMPerPixel=CMPerPixel
        self.__startP=np.zeros((2,1))
        self.__goalP=np.zeros((2,1))
        self.__tolerance=1e-6                   #tolerance used to determine if something stayed the same
        self.__path=[]
        mapShape=self.__map.shape
        self.__mapDimensions=tuple((mapShape[0],mapShape[1]))

    
    #internal method that generates a gradient map. it calculates for each unoccupied cell the distance to the goal.
    #It is not the euclidian distance, but the distance in unoccupied cells,which essentially means it calculates the shortest path from each unoccupied cell to the goal
    #It start doing this at the goal 
    def __generateGradient(self):
        self.__gradientMap=np.zeros_like(self.__map)#make sure the map is all 0 again
        #all visited notes
        openSet = [self.__goalP]
        closedSet = []                  #all investigated nodes
              
        movements=self.__get_movements_8n()
        while openSet:
            currentGridPoint=openSet.pop(0) 
            closedSet.append(tuple(currentGridPoint))
            for dx, dy, deltacost in movements:
            
                neighbor = (currentGridPoint[0]+dx, currentGridPoint[1]+dy)

                # if the node is not in the map, skip
                if(neighbor[0]>=self.__mapDimensions[0] or neighbor[1]>=self.__mapDimensions[1] or neighbor[0]<0 or neighbor[1]<0):
                    
                    continue
                
                # if the node is occupied or has already been visited, skip
                if (self.__map[neighbor[0], neighbor[1]] or (neighbor in closedSet)):
                    continue

                # compute the cost to reach the node through the given path

                tentative_Distance = self.__gradientMap[currentGridPoint[0],currentGridPoint[1]]+deltacost

                # Add the neighbor list of nodes who's neighbors need to be visited
                if(neighbor not in openSet):
                    openSet.append(neighbor)

                # If the computed cost if the best one for that node, then update the costs and 
                # node from which it came
                
                if (tentative_Distance < self.__gradientMap[neighbor[0],neighbor[1]] or self.__gradientMap[neighbor[0],neighbor[1]]==0):
                    # This path to neighbor is better than any previous one. Record it!
                    self.__gradientMap[neighbor[0],neighbor[1]]=tentative_Distance
        
    #uses the generated Gradient map to generate a path from the specified starting location, by simply searching for the next cell with a lower distance.(it "rolls" downhill)
    #    
    def __generatePath(self):
        self.__path=[]
        self.__path.append(tuple(self.__startP))
        movements=self.__get_movements_8n()
        
        while (tuple(self.__goalP) not in self.__path):
            current=self.__path[-1]
            closestNeigbor=current
            for dx, dy, deltacost in movements:
                
                neighbor = (current[0]+dx, current[1]+dy)
                if(neighbor[0]>=self.__mapDimensions[0] or neighbor[1]>=self.__mapDimensions[1] or neighbor[0]<0 or neighbor[1]<0):              
                    continue
                if(self.__map[neighbor[0],neighbor[1]]!=self.__occupancyMarkerMap):
                    if(self.__gradientMap[neighbor[0],neighbor[1]]<self.__gradientMap[closestNeigbor[0],closestNeigbor[1]]):
                        closestNeigbor=neighbor
            if(tuple(closestNeigbor)==tuple(current)):
                #print("Path:",self.__path)
                raise Exception('no path found')
            else:
                self.__path.append(tuple(closestNeigbor))

        
        #print("path to goal:",self.__path)
    
    
    #function given in previous exercices and used as is
    #author: unknown
    def __get_movements_8n(self):
        """
        Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
        (up, down, left, right and the 4 diagonals).
        :return: list of movements with cost [(dx, dy, movement_cost)]
        """
        s2 = math.sqrt(2)
        return [(1, 0, 1.0),
                (0, 1, 1.0),
                (-1, 0, 1.0),
                (0, -1, 1.0),
                (1, 1, s2),
                (-1, 1, s2),
                (-1, -1, s2),
                (1, -1, s2)]