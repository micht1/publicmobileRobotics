#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:00:40 2020

@author: barmak
"""

import os
import sys
import time
import serial
import numpy as np

# Adding the src folder in the current directory
#sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from Thymio import Thymio

#th = Thymio.serial(port="COM5", refreshing_rate=0.1)
#time.sleep(1) # To make sure the Thymio has had time to connect

def move(thymio,l_speed=500, r_speed=500, verbose=False):
    """
    Sets the motor speeds of the Thymio 
    param l_speed: left motor speed
    param r_speed: right motor speed
    param verbose: whether to print status messages or not
    """
    # Printing the speeds if requested
    if verbose:
        print("\t\t Setting speed : ", l_speed, r_speed)

    # Changing negative values to the expected ones with the bitwise complement
    l_speed = l_speed if l_speed >= 0 else 2 ** 16 + l_speed
    r_speed = r_speed if r_speed >= 0 else 2 ** 16 + r_speed

    # Setting the motor speeds
    thymio.set_var("motor.left.target", l_speed)
    thymio.set_var("motor.right.target", r_speed)


def stop(thymio,verbose=False):
    """
    param verbose: whether to print status messages or not
    """
    # Printing the speeds if requested
    if verbose:
        print("\t\t Stopping")

    # Setting the motor speeds
    thymio.set_var("motor.left.target", 0)
    thymio.set_var("motor.right.target", 0)


def run_ann_without_memory(thymio):

    # Weights of neuron inputs
    w_l = np.array([40,  20, -20, -20, -40,  30, -10])
    w_r = np.array([-40, -20, -20,  20,  40, -10,  30])

    # Scale factors for sensors and constant factor
    sensor_scale = 200
    constant_scale = 20

    # State for start and stop
    state = 1

    x = np.zeros(shape=(7,))
    y = np.zeros(shape=(2,))

    j = 0
    while True:
        j += 1

        if state != 0:
            # Get and scale inputs
            x = np.array(thymio["prox.horizontal"]) / sensor_scale

            # Compute outputs of neurons and set motor powers
            y[0] = np.sum(x * w_l) + 100
            y[1] = np.sum(x * w_r) + 100

            #print(j, int(y[0]), int(y[1]), thymio["prox.horizontal"])
            move(thymio,int(y[0]), int(y[1]))
        if(all(sensorValues==0 for sensorValues in robotStatus.thymio["prox.horizontal"])):
         return
  #run_ann_without_memory(th)
