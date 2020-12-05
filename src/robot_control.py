
import os
import sys
import time
import math
import serial
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from Thymio import Thymio
from tqdm import tqdm

#const definition
SPEED = 200
MAX_SPEED = 500

def odometry(p, sigma_p, t, MAX_SPEED, B = 9.5, CALIB = 0.0315, Z = np.zeros((3,2))):

    #get elapsed time and wheels speed
    t[1] = time.time()

    if t[0] == 0:
        T = 1e-6
    else:
        T = np.float32(t[1]-t[0])

    t[0] = time.time()

    speed_l = th["motor.left.speed"]
    speed_r = th["motor.right.speed"]

    #convert for negative speed
    if speed_l > MAX_SPEED:
        speed_l = speed_l - 2**16
    if speed_r > MAX_SPEED:
        speed_r = speed_r - 2**16
    
    # compute wheel displacement
    ds_l = T*speed_l*CALIB
    ds_r = T*speed_r*CALIB
    #compute displacement
    ds = 0.5*(ds_r + ds_l)
    d_theta = (ds_r - ds_l)/B
    dx = ds*math.cos(p[2] + d_theta/2)
    dy = - ds*math.sin(p[2] + d_theta/2)
    dp = np.array([dx, dy, d_theta])
    p = p + dp
    # bound the robot orientation between [-pi , pi]
    if p[2] > math.pi:
        p[2] = p[2] - 2*math.pi
    if p[2] < - math.pi:
        p[2] = p[2] + 2*math.pi

    ### standard deviation covarance matrix, see slide 21, lesson 6 of BMR
    k = 2e-2
    sigma_delta = np.array([[k*math.fabs(ds_r), 0                ], 
                            [0                , k*math.fabs(ds_l)]])
    c = math.cos(p[2] + d_theta/2)
    s = math.sin(p[2] + d_theta/2)
    J = np.array([[(1 + ds_l)/2*c + (ds_r + ds_l)*(1 - ds_l)/4*s, (ds_r + 1)/2*c + (ds_r + ds_l)*(ds_r - 1)/4*s, 1, 0, 0], 
                  [(1 + ds_l)/2*s - (ds_r + ds_l)*(1 - ds_l)/4*c, (ds_r + 1)/2*s - (ds_r + ds_l)*(ds_r - 1)/4*c, 0, 1, 0], 
                  [(1 - ds_l)/2                                 , (ds_r - 1)/2                             , 0, 0, 1]])
    Sigma = np.asarray(np.bmat([[sigma_p, Z], [np.transpose(Z), sigma_delta]]))
    D = np.matmul(J, Sigma)
    Sigma_prim = np.matmul(D, np.transpose(J))

    #######################################################################
    
    return p, Sigma_prim, t

def path_following(p, path, THREASHOLD = 0.5):
    
    waypoint = path[:,0]

    ############### WAYPOINT METRICS ##################

    waypoint_dir = waypoint-[p[0],p[1]]
    #robot-->waypoint distance
    waypoint_dist = math.sqrt(sum(waypoint_dir**2))
    #robot-->waypoint angle
    waypoint_ang = math.atan2(- waypoint_dir[1],waypoint_dir[0])
    #relative error with robot orientation
    err_angle = waypoint_ang - p[2]
    #if the error angle if above 180° turn the other way
    if err_angle > math.pi:
        err_angle = err_angle - 2*math.pi
    if err_angle < - math.pi:
        err_angle = 2*math.pi + err_angle


    ###################################################

    #if the waypoint is reached, returns popped path and next waypoint
    if waypoint_dist < THREASHOLD:
        path, waypoint = popcol(path, 0)
        #if the robot has reached the goal, exit
        if np.size(path) == 0:
            print('GOAL REACHED')
            return p, path  

    speed_regulation(waypoint_dist, err_angle)

    return p, path

def popcol(my_array,pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    print('---------------------------------------------------------WAYPOINT REACHED')
    i = pc
    pop = my_array[:,i]
    new_array = np.hstack((my_array[:,:i],my_array[:,i+1:]))
    return [new_array,pop]

def speed_regulation(waypoint_dist, err_angle, K = MAX_SPEED, FORWARD_THREASHOLD = 0.1):

    ################### regulation #####################
    
    ### Proportional control
    forward_speed = K/3/(5*math.fabs(err_angle)+1)
    rotation_speed = err_angle*K/4

    #compute wheel speed speed
    left_wheel_speed = forward_speed - rotation_speed
    right_wheel_speed = forward_speed + rotation_speed
    #convert to integer
    left_wheel_speed = np.int16(left_wheel_speed)
    right_wheel_speed = np.int16(right_wheel_speed)

    #Saturate speed
    if left_wheel_speed > MAX_SPEED:
        left_wheel_speed = MAX_SPEED
    if left_wheel_speed < - MAX_SPEED:
        left_wheel_speed = - MAX_SPEED
    if right_wheel_speed > MAX_SPEED:
        right_wheel_speed = MAX_SPEED
    if right_wheel_speed < - MAX_SPEED:
        right_wheel_speed = - MAX_SPEED

    #assign speed to wheel and manage negative values
    if left_wheel_speed < 0:
        th.set_var("motor.left.target", 2**16 + left_wheel_speed)
    else:
        th.set_var("motor.left.target", left_wheel_speed)
    if right_wheel_speed < 0:
        th.set_var("motor.right.target", 2**16 + right_wheel_speed)
    else:
        th.set_var("motor.right.target", right_wheel_speed)