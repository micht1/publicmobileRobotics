import os
import sys
import time
import serial
import numpy as np
from numpy import linalg as LNG 
import math
import matplotlib.pyplot as plt
from matplotlib import colors
#class that does pathplanning for a fixed map
class pathPlaning: 
    
    def getPath(self):#returns  the path with grid box "accuracy" so a point for every gridbox
        print("pixel to CM",self.__CMPerPixel)
        return [tuple([coordinate* float(self.__CMPerPixel) for coordinate in point]) for point in self.__path]
    def getOptimizedPath(self):     #returns only the  points where the calculated path changes direction
        originalPath=np.array(self.__path)
        originalPath = np.array(originalPath).reshape(-1, 2).transpose()
        edgePoints = [tuple(originalPath[:,0])]
        movmentOld=np.array([0, 0])
        movmentNew=np.array([0, 0])

        for pointNumber in range(1,originalPath.shape[1]):
            movmentNew=originalPath[:,pointNumber]-originalPath[:,pointNumber-1]
            if(abs(movmentNew[0]-movmentOld[0])>self.__tolerance or abs(movmentNew[1]-movmentOld[1])>self.__tolerance):
                if(tuple(originalPath[:,pointNumber-1]) not in edgePoints):
                    edgePoints.append(tuple(originalPath[:,pointNumber-1]))
                
            movmentOld=movmentNew
        edgePoints.append(tuple(originalPath[:,-1]))
        return [tuple([coordinate* float(self.__CMPerPixel) for coordinate in point]) for point in edgePoints]
        
    def setStart(self,start):
        self.__startP=np.around(np.divide(start,self.__CMPerPixel))
        self.__startP=self.__startP.astype(int)
        if(self.__startP[0]>=self.__mapDimensions[0] or self.__startP[1]>=self.__mapDimensions[1] or self.__startP[0]<0 or self.__startP[1]<0):
            raise Exception('Start outside Map')
        if(self.__map[self.__startP[0], self.__startP[1]]==self.__occupancyMarkerMap):
            raise Exception('Start node is not traversable')
        self.__generatePath()
        
        
    def setGoal(self,goal):
    
        if(LNG.norm(np.subtract(self.__goalP,np.around(np.divide(goal,self.__CMPerPixel))))>self.__tolerance):
            self.__goalP=np.around(np.divide(goal,self.__CMPerPixel))
            self.__goalP=self.__goalP.astype(int)
            if(self.__goalP[0]>=self.__mapDimensions[0] or self.__goalP[1]>=self.__mapDimensions[1] or self.__goalP[0]<0 or self.__goalP[1]<0):
                raise Exception('Goal outside Map')
            if(self.__map[self.__goalP[0], self.__goalP[1]]==self.__occupancyMarkerMap):
                raise Exception('goal node is not traversable')
            self.__generateGradient()
        else:
            self.__goalP=self.__goalP.astype(int)
			
    def getDistanceMap(self):
        
        return self.__gradientMap
        
        
        
    #initlializing the pathplanning process with a map and map related parameters
    def __init__(self, occupancyGrid,occupancyMarker,CMPerPixel):
        self.__map=occupancyGrid
        self.__occupancyMarkerMap=int(occupancyMarker)
        self.__gradientMap=np.zeros_like(occupancyGrid)
        self.__CMPerPixel=CMPerPixel
        self.__startP=np.zeros((2,1))
        self.__goalP=np.zeros((2,1))
        self.__tolerance=1e-6
        self.__path=[]
        
        self.__mapDimensions=self.__map.shape
        #method to set the goal
    
    
    def __generateGradient(self):
        #all visited notes
        openSet = [self.__goalP]
        closedSet = []
              
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
        
        
    def __generatePath(self):
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