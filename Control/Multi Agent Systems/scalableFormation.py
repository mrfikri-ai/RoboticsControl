# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 08:17:54 2022

@author: Muhamad Rausyan Fikri

This code elaborates the scalable formation control in continuous systems
-- not finished yet --
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import lsim

# Initialization
# untuk mendapatkan nilai phi = L + R
adjA = np.array([[0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 1, 0, 0]])
 
v = np.ones(4, dtype=np.int16)
D = np.diag(v)
L = D - adjA

# Define xdx and xdy
xdx = np.array([[1], [3], [1], [3]])
xdy = np.array([[1], [1], [3], [3]])

# N = 4
sumA = np.zeros((4, 4))
sumB = np.zeros((4, 4))
rx = np.zeros(4)
ry = np.zeros(4)

# iterate through rows
for i in range(4):
   # iterate through columns
   for j in range(4):
       sumA[i, j] = adjA[i, j] * (xdx[i] - xdx[j])
       sumB[i, j] = adjA[i, j] * (xdy[i] - xdy[j])
   rx[i] = -(1/xdx[i]*sum(sumA[i,:]))
   ry[i] = -(1/xdy[i]*sum(sumB[i,:]))
   
   
# Define diagonal matrix
Rx = np.array([[rx[0], 0, 0, 0],
               [0, rx[1], 0, 0],
               [0, 0, rx[2], 0],
               [0, 0, 0, rx[3]]])

Ry = np.array([[ry[0], 0, 0, 0],
               [0, ry[1], 0, 0],
               [0, 0, ry[2], 0],
               [0, 0, 0, ry[3]]])

phix = L + Rx 
phiy = L + Ry

np.linalg.eig(phix)
np.linalg.eig(phiy)

Ax = np.concatenate(([-L, -Rx],[np.zeros((4,4)), -Rx]))
Ay = np.concatenate(([-L, -Ry],[np.zeros((4,4)), -Ry]))

A = np.random.rand(4,4)
B = np.zeros((8,1))
C = np.eye(8)
D = np.array([[0]])
t = np.linspace(0, 50, num = 5000)
u = np.zeros(len(t))

f = len(t)
# f = f[1,2]

sysx = signal.StateSpace(Ax, B, C, D)
sysy = signal.StateSpace(Ay, B, C, D)

# scipy.signal.lsim(system, U, T, X0=None, interp=True)
x = lsim(sysx, u, t, np.array([[2],[4],[8],[2],[4],[9],[8],[xdx[4]]]))
y = lsim(sysy, u, t, np.array([[0],[0],[1],[8],[8],[2],[9],[xdy[4]]]))

x1 = lsim(sysx, u, t, np.array([[x[f,1]], [x[f,2]], [x[f,3]], [x[f,4]],
                                [x[f,5]], [x[f,6]], [x[f,7], [xdx[4]*2])) 
y1 = lsim(sysy, u, t, np.array([[y[f,1]], [y[f,2]], [y[f,3]], [y[f,4]],
                                [y[f,5]], [y[f,6]], [y[f,7], [xdy[4]*2]))


'''
x = lsim(sysx,u,t,[2;4;8;2;4;9;8;xdx(4)]);
y = lsim(sysy,u,t,[0;0;1;8;8;2;9;xdy(4)]);

x1 = lsim(sysx,u,t,[x(f,1);x(f,2);x(f,3);x(f,4);x(f,5);x(f,6);x(f,7);xdx(4)*2]);
y1 = lsim(sysy,u,t,[y(f,1);y(f,2);y(f,3);y(f,4);y(f,5);y(f,6);y(f,7);xdy(4)*2]);
'''
