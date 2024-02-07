#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:50:40 2023

@author: lay0005
"""
import pandas as pd
from utils import train_alg, percentage, full_spec, organize_rollout
from BOSNC import robustness,specification,signal,size, GP_opt
from HumanLabelingAutomation import human_labeling

# Hyperparams
op = 'G'
env_id = 'SafetyPointCircle1-v0'
total_steps = 1000000
steps_per_epoch = 500
num_rollout = 50 # Number of rollouts per iteration
cost_limit = 0 #cost limit for Lagrange
delta = 0.90# safe traces percentage threshold

# Design True specification used for human labeling automation


wall_x_pos = 1.0
wall_x_neg = -1.20

spec_obs_template = '(agent_x > {:.2f}) or (agent_x < {:.2f})'
spec_obs = spec_obs_template.format(wall_x_pos, wall_x_neg)
# SPC1 ********

spec_obs_true = f'G(not({spec_obs}))'


# Manually designed rollout with guaranteed positive traces
df_x1, df_y1 = organize_rollout('initial_dataset_100ep_SNC_unsafe.pkl')
df_x2, df_y2 = organize_rollout('initial_dataset_100ep_SNC_safe.pkl')
df_x = pd.concat([df_x1.iloc[:, : 10], df_x2.iloc[:, : 10]], axis=1, ignore_index=True)
df_y = pd.concat([df_y1.iloc[:, : 10], df_y2.iloc[:, : 10]], axis=1, ignore_index=True)

print('Human Labeling...')
x_reg,y_reg, x_anom , y_anom = human_labeling(spec_obs_true,op, df_x, df_y,cost_limit) # labeling traces
num_safe = len(x_reg.columns) #number of safe rollout traces
num_unsafe = len(x_anom.columns) #number of safe rollout traces
percentage_safe = percentage(len(x_reg.columns),len(x_anom.columns))
safe_p = [percentage_safe]
size_xreg = [len(x_reg.columns)]
print('Initial Percentage of safe traces: ',percentage_safe,'%')

# Running our algorithm
percentage_safe = 0
print('Begining bi-level optimization...(our Alg)')
it = 1 #iteration number
specs = []
best_Ys = []
while percentage_safe < 90.0:
    # pop_df = initialize(x_reg, y_reg, x_anom, y_anom, rng)
    # infered_STL = GA(pop_df,x_reg, y_reg, x_anom, y_anom, rng) #infer full STL from traces (template+parameters)
    rng = [[0,2],[-2,0]]
    print('Optimizing parameters...')
    infered_STL_obs, best_Y = GP_opt(50,spec_obs_template,op, x_reg,y_reg, x_anom , y_anom,cost_limit,rng) #Infer parameters given STL template
    # full_spec_new = full_spec(spec_goal, infered_STL_obs)
    print('----Infered STL_Obs is----', infered_STL_obs)
    specs.append(infered_STL_obs)
    best_Ys.append(best_Y)
    
    # Train agorithm and get rollout dataset
    df_x, df_y = train_alg('TD3Lag', env_id,total_steps, steps_per_epoch, infered_STL_obs, num_rollout) # Running TD3 to get results for BL1

    # Human labeling of rollout dataset
    x_r, y_r, x_a, y_a = human_labeling(spec_obs_true,op, df_x, df_y, cost_limit)  # Label rollout traces from RL
    
    # computing safe percentage in rollout dataset
    percentage_safe = percentage(len(x_r.columns),len(x_a.columns))
    safe_p.append(percentage_safe)
    
    # Appending new data to original dataset
    x_reg = pd.concat([x_reg, x_r], axis=1, ignore_index=True)
    y_reg = pd.concat([y_reg, y_r], axis=1, ignore_index=True) 
    x_anom = pd.concat([x_anom, x_a], axis=1, ignore_index=True)
    y_anom= pd.concat([y_anom, y_a], axis=1, ignore_index=True)

    size_xreg.append(len(x_reg.columns))
    num_safe = len(x_r.columns) #number of safe rollout traces
    num_unsafe = len(x_a.columns)
    print('Iteration number: ',it,'---',percentage_safe,'% safe traces')
    it += 1

# Run Baseline1 - cost not considered

# print('Running BL1')
df_x, df_y = train_alg('TD3', env_id, total_steps, steps_per_epoch, spec_obs, num_rollout) # Running TD3 to get results for BL1
x_reg,y_reg, x_anom , y_anom = human_labeling(spec_obs_true,op, df_x, df_y,cost_limit) # labeling traces
num_safe = len(x_reg.columns) #number of safe rollout traces
num_unsafe = len(x_anom.columns) #number of safe rollout traces
percentage_safe = percentage(len(x_reg.columns),len(x_anom.columns))
safe_p = [percentage_safe]
size_xreg = [len(x_reg.columns)]
print('Initial Percentage of safe traces: ',percentage_safe,'%')

# Run Baseline2 - True STL constraint used

print('Running BL2')
df_x, df_y = train_alg('TD3Lag', env_id,total_steps, steps_per_epoch, spec_obs, num_rollout) # Running TD3 to get results for BL1
x_reg,y_reg, x_anom , y_anom = human_labeling(spec_obs_true,op, df_x, df_y,cost_limit) # labeling traces
num_safe = len(x_reg.columns) #number of safe rollout traces
num_unsafe = len(x_anom.columns) #number of safe rollout traces
percentage_safe_BL2 = percentage(len(x_reg.columns),len(x_anom.columns))
size_xreg_BL2 = [len(x_reg.columns)]
print('BL2 Percentage of safe traces: ',percentage_safe_BL2,'%')

