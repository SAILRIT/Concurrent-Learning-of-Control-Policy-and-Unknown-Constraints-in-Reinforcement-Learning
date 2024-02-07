#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:14:42 2023

@author: lay0005
"""

from cmath import inf
from math import exp
from random import seed
import GPyOpt
import numpy as np
from scipy.stats import norm
import pandas as pd
import math
import rtamt

def robustness(dataset, spec):
    rob = spec.evaluate(['agent_x',dataset[0]],['agent_y',dataset[1]])
    return [i[1] for i in rob]

def specification(spec_str):
    spec = rtamt.StlDenseTimeSpecification()
    spec.name = 'STL discrete-time online Python monitor'
    spec.declare_var('agent_x', 'float')
    spec.declare_var('agent_y', 'float')
    spec.spec = spec_str
    spec.parse()
    return spec


def signal(x_tr,y_tr,i):
    x = []
    y = []
    for j in range(len(x_tr)): 
        x.append([j,x_tr[i][j]])
        y.append([j,y_tr[i][j]])
    return (x, y)

def size(formula):
    ops = '<>FG&|Un'
    size = 0
    for i in formula:
        if i in ops:
            size += 1
    return size
 
  
def GP_opt(episodes, spec_obs_template,operation, x_reg,y_reg,x_anom,y_anom,cost_limit,rng):

    def pars(x):
        t1 = 0
        t2 = inf
        # x = [[-0.75, 1.0,1.0, 0.2,-1.4,  0.7 ,-0.5,-0.3,0.25 ,0.9 ,0.0, -1.5 ,-1.9,1,1.0,-1.0]]
        # x = np.array(x)
        # print(x)
        # choose variables in the same way as the search "space" below was designed
        h1x = x[:, 0][0]
        h1y = x[:, 1][0]
        h2x = x[:, 2][0]
        h2y = x[:, 3][0]
        h3x = x[:, 4][0]
        h3y = x[:, 5][0]
        h4x = x[:, 6][0]
        h4y = x[:, 7][0]
        h5x = x[:, 8][0]
        h5y = x[:, 9][0]
        h6x = x[:, 10][0]
        h6y = x[:, 11][0]
        h7x = x[:, 12][0]
        h7y = x[:, 13][0]
        h8x = x[:, 14][0]
        h8y = x[:, 15][0]


            
        # Decide formatting based on template    
        spec_obs = spec_obs_template.format(h1x,h1y,h2x,h2y,h3x,h3y,h4x,h4y,h5x,h5y,h6x,h6y,h7x,h7y,h8x,h8y)
        phi = f'G(not {spec_obs})'
        
        # print('Phi is : ', phi)
        rob_reg = np.empty(0)
        rob_anom = np.empty(0)

        if t1 < t2:
            for i in range(len(x_reg.columns)):
                 if operation == '>' or operation == '<' or operation == 'not':
                     rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
                 elif operation == '&':
                     first_part = phi[:phi.find('&')]
                     second_part = phi[phi.find('&')+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_reg,y_reg,i), specification(first_part))
                         rob2 = robustness(signal(x_reg,y_reg,i), specification(second_part))
                         rob_r = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))

                 elif operation == '|' :
                     first_part = phi[:phi.find('|', phi.find('|') + 1)]
                     second_part = phi[phi.find('|', phi.find('|') + 1)+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_reg,y_reg,i), specification(first_part))
                         rob2 = robustness(signal(x_reg,y_reg,i), specification(second_part))
                         rob_r = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))
                 elif operation == 'G' or operation == 'F' or operation == 'U':            
                     rob_r =robustness(signal(x_reg,y_reg,i), specification(phi))[0]
                 rob_reg = np.append(rob_reg, rob_r)
                 
            for i in range(len(x_anom.columns)):
                 if operation == '>' or operation == '<' or operation == 'not':
                     rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
                 elif operation == '&':
                     first_part = phi[:phi.find('&')]
                     second_part = phi[phi.find('&')+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_anom,y_anom,i), specification(first_part))
                         rob2 = robustness(signal(x_anom,y_anom,i), specification(second_part))
                         rob_a = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))
                         # print(rob1,rob2)
                         # print('--------------------------------',rob_a)
                 elif operation == '|' :
                     first_part = phi[:phi.find('|', phi.find('|') + 1)]
                     second_part = phi[phi.find('|', phi.find('|') + 1)+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_anom,y_anom,i), specification(first_part))
                         rob2 = robustness(signal(x_anom,y_anom,i), specification(second_part))
                         rob_a = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))

                 elif operation == 'G' or operation == 'F' or operation == 'U':            
                     rob_a =robustness(signal(x_anom,y_anom,i), specification(phi))[0]

                 # Adjusting for under 25 vilation being labled as safe    
                 reference_points = [(-0.75, 1.0), (1.0, 0.20), (-1.40, 0.70), (-0.50, -0.30), (0.25, 0.90), (0.0, -1.50),(-1.9, 1.00),(1.00, -1.00)]        
                 # Sample trace data as lists of (x, y) values
                 
                 x_values = x_anom[i] # Replace with your actual x values
                 y_values = y_anom[i]  # Replace with your actual y values

                 violated_timesteps = []
                 
                 # Iterate through the trace
                 for timestep, (x_an, y_an) in enumerate(zip(x_values, y_values)):
                     # Check the STL formula for each reference point
                     violated = any(math.sqrt((x_an - x_ref)**2 + (y_an - y_ref)**2) < 0.4 for x_ref, y_ref in reference_points)
                 
                     if violated:
                         violated_timesteps.append(timestep)
                 # print('-----------VTS',len(violated_timesteps),'at',i)        
                 # Print all time steps where the STL formula was violated
                 if violated_timesteps:
                     if len(violated_timesteps) <= cost_limit:
                        rob_a = abs(rob_a)
  
                 rob_anom = np.append(rob_anom, rob_a)
            # print(rob_anom)            
            TP = 0
            FN = 0
            TN = 0
            FP = 0
            for i in rob_reg:
                if i>= 0: TP +=1 
                if i<0: FN +=1
            for i in rob_anom:
                 if i< 0:TN +=1
                 if i >=0: FP +=1
            # rob_reg_av = np.average(rob_reg)
            # rob_anom_av = np.average(rob_anom)    

            print('--------',TP,TN,'-----',FP,FN)
            # try:            
            #     MCR = (abs((len(x_reg.columns)-pos)/len(x_reg.columns)) + abs((len(x_anom.columns)-neg)/len(x_anom.columns)) )/2
            # except:
            #     A = pos + neg + 10*abs(rob_reg_av-rob_anom_av) #Fix later
            
            # if pos + neg > 0.95*(len(x_reg.columns) + len(x_anom.columns)):
            #     A += 100   
            MCR = ((abs((len(x_reg.columns)-TP)/len(x_reg.columns)) + abs((len(x_anom.columns)-TN)/len(x_anom.columns)) )/2)*100
            # MCR = (FP + FN)/(TP+TN +FN+FP)
        elif t2<=t1:
            MCR = 100
        print('MCR is:',MCR)
        return MCR
    #Design space based on template of the STL
    domain = tuple(rng) # search space for parameters (happens to be the same for all parameters in the grid world)
    space = [{'name': 'h1x', 'type': 'continuous', 'domain': domain},
            {'name': 'h1y', 'type': 'continuous', 'domain': domain},
            {'name': 'h2x', 'type': 'continuous','domain': domain},
            {'name': 'h2y', 'type': 'continuous', 'domain': domain},
            {'name': 'h3x', 'type': 'continuous','domain': domain},
            {'name': 'h3y', 'type': 'continuous', 'domain': domain},
            {'name': 'h4x', 'type': 'continuous','domain': domain},            
            {'name': 'h4y', 'type': 'continuous', 'domain': domain},
            {'name': 'h5x', 'type': 'continuous','domain': domain},
            {'name': 'h5y', 'type': 'continuous', 'domain': domain},
            {'name': 'h6x', 'type': 'continuous','domain': domain},
            {'name': 'h6y', 'type': 'continuous', 'domain': domain},
            {'name': 'h7x', 'type': 'continuous','domain': domain},
            {'name': 'h7y', 'type': 'continuous', 'domain': domain},
            {'name': 'h8x', 'type': 'continuous','domain': domain},
            {'name': 'h8y', 'type': 'continuous', 'domain': domain}]

    feasible_region = GPyOpt.Design_space(space=space)
    # bounds = feasible_region.get_bounds()
    # print(bounds)

    initial_design = GPyOpt.experiment_design.initial_design(
        'random', feasible_region, 50)

    # print(initial_design)
    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(pars)

    # --- CHOOSE the model type
    model = GPyOpt.models.GPModel(
        exact_feval=True, optimize_restarts=10, verbose=False)

    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(
        feasible_region)

    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI(
        model, feasible_region, optimizer=aquisition_optimizer)

    # --- CHOOSE a collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # BO object
    bo = GPyOpt.methods.ModularBayesianOptimization(
        model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    max_time = None
    max_iter = episodes
    tolerance = 1e-8     # distance between two consecutive observations

    # Run the optimization
    bo.run_optimization(max_iter=max_iter, max_time=max_time,
                        eps=tolerance, verbosity=False)

    # bo.plot_acquisition()
    best_Y = bo.plot_convergence()
    
    vals = []
    for i in range(len(bo.x_opt)):
        vals.append(bo.x_opt[i])
    # Final spec with parameters
    spec_obs = spec_obs_template.format(vals[0],vals[1],vals[2],vals[3],vals[4],vals[5], vals[6],vals[7],vals[8],vals[9],vals[10], vals[11],vals[12], vals[13],vals[14], vals[15]) #h1x,h1y,h2x,h2y,h3x,h3y,h4x,h4y,h5x,h5y,h6x,h6y
    return spec_obs, best_Y
#
#phi = GP_opt('G[{1},{2}](y > {0:.2f})','G', input1=x_reg, input2=y_reg,input3=x_anom, input4=y_anom, rng_mod=rng_mod)
#print(phi)
