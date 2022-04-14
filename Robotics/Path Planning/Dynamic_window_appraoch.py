# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:39:44 2022

@author: Muhamad Rausyan Fikri
"""

import numpy as np
from scipy import linalg
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}, threshold=sys.maxsize)

dt = 0.1

def DynamicWindowApproach(x, model, goal, evalParam, ob, R):
    # Dynamic Window [vmin,vmax,wmin,wmax]
    Vr = CalcDynamicWindow(x, model) # OK
    
    # Numerical calculation
    evalDB, trajDB = Evaluation(x, Vr, goal, ob, R, model, evalParam)
    
    if evalDB.size == 0:
        print('no path to goal!!');
        u = np.array([[0], [0]])
        quit()
     
    # Regularization of each function
    evalDB = NormalizeEval(evalDB);
    
    # Calculation of the final evaluation function
    feval = np.empty((0,1), float);
    for id in range(0, len(evalDB[:,0])):
        
        evalParam_temp = np.reshape(evalParam[0:3], (1, 3))
        feval = np.append(feval, np.dot(evalParam_temp, np.reshape(evalDB[id,2:5], (3,1))), axis=0);
                 
    evalDB = np.append(evalDB, feval, axis=1)
    
    # optimal evaluation function
    maxv, ind = np.where(feval == np.amax(feval))         
    ind = np.reshape(maxv, (1, 1))
    maxv = feval[maxv]
    
    u = np.reshape(evalDB[ind, 0:2], (2, 1)) 
    return u, trajDB
    
def CalcDynamicWindow(x, model):
    # Maximum and minimum range of car speed
    Vs = np.array([[0, model[0], -model[1], model[1]]])
    
    # Dynamic window calculated based on current speed and acceleration limits
    Vd = np.array([x[3]-np.dot(model[2],dt), x[3]+np.dot(model[2],dt), x[4]-np.dot(model[3],dt), x[4]+np.dot(model[3],dt)])
    Vd = [Vd.flatten()]
    
    Vtmp = np.append(Vs, Vd, axis=0)
    Vr = np.array([max(Vtmp[:,0]), min(Vtmp[:,1]), max(Vtmp[:,2]), min(Vtmp[:,3])])
    
    return Vr

def Evaluation(x, Vr, goal, ob, R, model, evalParam):
    evalDB = np.empty((0, 5), float);
    trajDB = np.empty((0, 31), float);
    
    Vr[0] = round(Vr[0], 4)
    Vr[1] = round(Vr[1], 4)
    model[4] = round(model[4], 4)
    Vr[2] = round(Vr[2], 4)
    Vr[3] = round(Vr[3], 4)
    model[5] = round(model[5], 4)
    
    for vt in np.arange(Vr[0], Vr[1] + model[4], model[4]):
        for ot in np.arange(Vr[2], Vr[3] + model[5], model[5]):
            
            # Trajectory estimation; get xt: the predicted pose after 
            # the robot moves forward; traj: the trajectory between the current moment and the predicted moment
            ot = round(ot, 4)
           
            # evalParam(4), forward simulation time;
            xt, traj = GenerateTrajectory(x, vt, ot, evalParam[3], model);  

            # Calculation of evaluation
            heading = CalcHeadingEval(xt, goal);
            dist = CalcDistEval(xt, ob, R)
            vel = abs(vt);
            
            # braking system
            stopDist = CalcBreakingDist(vel, model);
            
            if dist > stopDist: 
                evalDB = np.append(evalDB, np.array([[vt, ot, heading, dist, vel]]), axis=0)
                trajDB = np.append(trajDB, traj, axis=0)
        
    return evalDB, trajDB

def GenerateTrajectory(x,vt,ot,evaldt,model):
    # Trajectory generation function
    # evaldt: forward simulation time; vt, ot current velocity and angular velocity;
    
    time = 0;
    u = np.array([[vt],[ot]])             # input value
    traj = x                    # robot trajectory
    
    while time <= evaldt:
        time = time + dt        # time update
        x = f(x,u)              # motion update
        
        traj = np.append(traj, x, axis=1)
        
    return x, traj

def f(x, u):
    # Motion Model
    # u = [vt; wt];Velocity and angular velocity at the current moment
     
    F = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])

    B = np.array([[np.dot(dt,math.cos(x[2])), 0],
                  [np.dot(dt,math.sin(x[2])), 0],
                  [0, dt],
                  [1, 0],
                  [0, 1]])

    x = np.dot(F, x) + np.dot(B, u)  #this is correct has been tested
    return x

def CalcHeadingEval(x, goal):
    # evaluation of heading robot
    theta = math.degrees(x[2]);             # Robot orientation
    
    # Orientation of target point
    goalTheta = math.degrees(math.atan2(goal[1]-x[1][0], goal[0]-x[0][0])); 
    
    if goalTheta > theta:
        targetTheta = goalTheta - theta         # [deg]
    else:
        targetTheta = theta - goalTheta         # [deg]
    
    heading = 180 - targetTheta;
    return heading

def CalcDistEval(x, ob, R):
    # Obstacle distance evaluation
    dist = 100;
    for io in range(0, len(ob[:, 0])):
        disttmp = linalg.norm((ob[io,:] - np.reshape(x[0:2], (2,1))), 2) - R; # NORM function with type-of "2-norm"
        if dist > disttmp:           # Minimum distance to obstacle
            dist = disttmp

    # The obstacle distance evaluation limits a maximum value. 
    # If it is not set, once a trajectory has no obstacles, it will take too much weight
    if dist >= 2*R:
        dist = 2*R
    
    return dist

def CalcBreakingDist(vel, model):
    # Calculate the braking distance according to kinematic model
    
    stopDist = 0;
    while vel > 0:
        stopDist = stopDist + vel*dt      # Stop or braking distance calculation
        vel = vel - model[2]*dt 
    return stopDist

def NormalizeEval(EvalDB):
    # Regularization of the function
    if not sum(EvalDB[:,2]) == 0:
        EvalDB[:,2] = EvalDB[:,2] / sum(EvalDB[:,2]);
        
    if not sum(EvalDB[:,3]) == 0:
        EvalDB[:,3] = EvalDB[:,3] / sum(EvalDB[:,3]);
        
    if not sum(EvalDB[:,4]) == 0:
        EvalDB[:,4] = EvalDB[:,4] / sum(EvalDB[:,4]);
        
    return EvalDB


print('Dynamic Window Approach sample program start!!')

# Initial state of the robot [x (m), y (m), yaw (Rad), v (m / s), w (rad / s)]
x = np.array([[0], [0], [math.pi/2], [0], [0]])

# Target point position [x (m), y (m)]
goal = np.array([7, 8]);

# Obstacle position list [x (m) y (m)]
obstacle = np.array([[0, 2],
                     [4, 2],
                     [4, 4],
                     [5, 4],
                     [5, 5],
                     [5, 6],
                     [5, 9],
                     [8, 8],
                     [8, 9],
                     [7, 9],
                     [6, 5],
                     [6, 3],
                     [6, 8],
                     [6, 7],
                     [7, 4],
                     [9, 8],
                     [9, 11],
                     [9, 6]])

# Obstacle radius for collision detection
obstacleR = 0.6;

# Time [s] 
dt = 0.1;

# Kinematics model
# Maximum speed m / s], maximum rotation speed [rad / s], acceleration [m / ss], rotation acceleration [rad / ss],
# Speed resolution rate [m / s], rotation speed resolution rate [rad / s]]
Kinematic = np.array([1.0, math.radians(20.0), 0.2, math.radians(50.0), 0.01, math.radians(1)]);

# Parameter number reference [heading, dist, velocity, predictDT]
evalParam = np.array([0.05, 0.2, 0.1, 3.0]);

# area of the environment # WINDOW OF THE ANIMATION x1 = -1; x2 = 11
area = np.array([-1, 11, -1, 11]);

# Imitation experiment
result = {
    'x': np.empty((0, 5), float)
}

for i in range(0,5001):
    # DWA 
    u, traj = DynamicWindowApproach(x, Kinematic, goal, evalParam, obstacle, obstacleR);
    
    x = f(x,u)  # The robot moves to the next moment
    
    # save the simulation results
    result['x'] = np.append(result['x'], np.reshape(x, (1,5)), axis=0);
    
    # when reach the destination
    if linalg.norm(x[0:2] - np.reshape(goal, (2,1))) < 0.5:
        print('Arrive Goal!!')
        break
    
    # ====Animation====
    # hold off;
    ArrowLength = 0.5 
    
    # initializing a figure in 
    # which the graph will be plotted
    fig, ax = plt.subplots(figsize=(12,7)) 
      
    # marking the x-axis and y-axis
    ax.axis(area)
    
    # Robot
    ax.quiver(x[0], x[1], ArrowLength*math.cos(x[2]), ArrowLength*math.sin(x[2]), headwidth=0.5, minshaft=0.5, hatch='O');
    # hold on;
    ax.plot(result['x'][:,0], result['x'][:,1]);# hold on;
    ax.plot(goal[0], goal[1],'*r'); # hold on;
    ax.plot(obstacle[:,0], obstacle[:,1],'*k');# hold on;
    
    # Explore the track
    if not traj.size == 0:
        #print("len(traj[:,0])/5         : \n{}".format(len(traj[:,0])/5))
        for it in np.arange(0, len(traj[:,0])/5):
            ind = int((1 + (it - 1) * 5)-1)
            #print("ind          : \n{} ; type: {}".format(ind, type(ind)))
            ax.plot(traj[ind,:],traj[ind+1,:]);# hold on;
