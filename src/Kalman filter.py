#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install filterpy')

from __future__ import division, print_function
from numpy.random import randn
from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance_ellipse
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

import copy
import matplotlib.pyplot as plt
import numpy as np


# In[7]:


class Camera(object):
    
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        
    # Each call to read() updates the position by one time step and returns the new measurement    
    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]


# In[8]:


from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance_ellipse
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

def tracker1():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    tracker.u = 0.
    
    # Q matrix: process noise
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05) # we assume that the noise is a discrete time Wiener process
    tracker.Q = block_diag(q, q)

    # H matrix: measurement function: z = H*x
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    # R matrix: measurement noise
    tracker.R = np.array([[5, 0], [0, 5]]) # we assume x and y are independant -> off diagonals = 0


    # P matrix: initial position & covariance
    tracker.x = np.array([[3, 0, 4, 0]]).T # initial position and velocity
    tracker.P = np.eye(4) * 500. # if we know it precisely, put a low number
    
    return tracker

# simulate thymio movement
N = 30
sensor = Camera([0, 0], (2, 1), 1.)
zs = np.array([np.array([sensor.read()]).T for _ in range(N)])

# run filter
thymio_tracker = tracker1()
mu, cov, _, _ = thymio_tracker.batch_filter(zs)

# plot results
plt.plot(zs[:, 0], zs[:, 1], label="Measured")
plt.plot(mu[:, 0], mu[:, 2], label="Filtered")
plt.legend()


# In[ ]:




