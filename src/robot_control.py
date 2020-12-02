
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

#variables definition
Ts = 0.1
Tw = 0.00
forward = 0
p = np.array([0,0,0])
trace_x = []
trace_y = []
plot = False
######################## TEST PATHS #############################
#sine
x_axis = np.arange(0,50,5)
amplitude = 5*np.sin(x_axis/5)
path = np.array([x_axis,amplitude])
#10cm square
#path = np.array([[0,10,10,0,0],[0,0,-10,-10,0]])
# straight line
#path = np.array([[0,60],[0,0]])
plt.plot(x_axis, amplitude)
plt.show()
#const definition
SPEED = 150
MAX_SPEED = 500

t = np.array([0,0], dtype = 'float64')

def odometry(p, t, MAX_SPEED, B = 9.5, CALIB = 0.0315):

    #get elapsed time and wheels speed
    t[1] = time.time()
    if t[0] == 0:
        T = 0
    else:
        T = np.float32(t[1]-t[0])
    print(T)
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
    # set robot angle to 0 deg if it has made one full turn
    if p[2] >= 2*math.pi:
        p[2] = 0
    
    return p,t

def popcol(my_array,pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    print('---------------------------------------------------------WAYPOINT REACHED')
    i = pc
    pop = my_array[:,i]
    new_array = np.hstack((my_array[:,:i],my_array[:,i+1:]))
    return [new_array,pop]

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

    #print some variables
    print('position:         ', p)
    print('waypoint:         ', waypoint)
    print('waypoint dir:     ', waypoint_dir)
    print('waypoint dist:    ', waypoint_dist)
    print('waypoint angle:   ', waypoint_ang)
    print('error angle:      ', err_angle)
    print("\n")
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

def speed_regulation(waypoint_dist, err_angle, K = MAX_SPEED, FORWARD_THREASHOLD = 0.1):

    ############## regulation #################
    ############## Proportional control
    forward_speed = K/4/(10*math.fabs(err_angle)+1)
    rotation_speed = err_angle*K/2

    ############## separated turn and forward displacement
    # if math.fabs(err_angle) <= FORWARD_THREASHOLD:
    #     th.set_var("motor.left.target", SPEED)
    #     th.set_var("motor.right.target", SPEED)
    # if err_angle > FORWARD_THREASHOLD:
    #     th.set_var("motor.left.target", 2**16 - SPEED)
    #     th.set_var("motor.right.target", SPEED)
    # if err_angle < (-1*FORWARD_THREASHOLD):
    #     th.set_var("motor.left.target", SPEED)
    #     th.set_var("motor.right.target", 2**16 - SPEED)
    #######################################################

    left_wheel_speed = forward_speed - rotation_speed
    right_wheel_speed = forward_speed + rotation_speed
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

    print('err_angle ', err_angle)
    print('forward speed ', forward_speed)
    print('rotation_speed', rotation_speed)
    print('left speed ', left_wheel_speed)
    if left_wheel_speed < 0:
        th.set_var("motor.left.target", 2**16 + left_wheel_speed)
    else:
        th.set_var("motor.left.target", left_wheel_speed)
    if right_wheel_speed < 0:
        th.set_var("motor.right.target", 2**16 + right_wheel_speed)
    else:
        th.set_var("motor.right.target", right_wheel_speed)



th = Thymio.serial(port="COM6", refreshing_rate=Ts)

# wait until connected
while len(th.variable_description()) == 0:
	time.sleep(0.5)
	print("wating for connection...")

print("connected")
time.sleep(3)

if plot:
    plt.ion()
    figure, ax = plt.subplots(figsize=(8,6))
    line1, = ax.plot(p[0],p[1])
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
""
# control
while np.size(path):
    time.sleep(Tw)
    p,t = odometry(p, t, MAX_SPEED)
    p, path = path_following(p, path)

    if plot:
        trace_x.append(p[0])
        trace_y.append(p[1])
        line1.set_xdata(trace_x)
        line1.set_ydata(trace_y)
        figure.canvas.draw()
        figure.canvas.flush_events()

    ############### manual control ##################
    # if th["button.forward"] == 1:
    #     th.set_var("motor.left.target", SPEED)
    #     th.set_var("motor.right.target", SPEED)
    # elif th["button.right"] == 1:
    #     th.set_var("motor.left.target", SPEED)
    #     th.set_var("motor.right.target", 2**16-SPEED)
    # elif th["button.left"] == 1:
    #     th.set_var("motor.left.target", 2**16-SPEED)
    #     th.set_var("motor.right.target", SPEED)
    # else:
    #     th.set_var("motor.left.target", 0)
    #     th.set_var("motor.right.target", 0)
    #################################################

    if th["button.center"] == 1:
        break
th.set_var("motor.left.target", 0)
th.set_var("motor.right.target", 0)
time.sleep(1)
quit()
""