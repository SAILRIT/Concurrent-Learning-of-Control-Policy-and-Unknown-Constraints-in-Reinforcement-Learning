#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:50:40 2023

@author: lay0005
"""
import pandas as pd
from utils import train_alg, percentage, full_spec, organize_rollout
from BOSVHC import robustness,specification,signal,size, GP_opt
from HumanLabelingAutomation import human_labeling

# Hyperparams
op = 'G'
env_id = 'SafetyHalfCheetahVelocity-v1'
total_steps = 1000000
steps_per_epoch = 1000
num_rollout = 50 # Number of rollouts per iteration
cost_limit = 0 #cost limit for Lagrange
delta = 0.90 # safe traces percentage threshold

# Design True specification used for human labeling automation
vx_threshold = 3.20960
spec_obs_template = 'v_x > {:.5f}'

spec_obs = spec_obs_template.format(vx_threshold)
spec_obs_true = f'G(not({spec_obs}))'


# Manually designed initial labled traces
df_x1, df_v1 = organize_rollout('initial_dataset_100ep_SVHC_unsafe.pkl')
df_x2, df_v2 = organize_rollout('initial_dataset_100ep_SVHC_safe.pkl')
# df_x2, df_v2 = organize_rollout('initial_dataset_300ep_SVHC_safe.pkl')
df_p_x = pd.concat([df_x1.iloc[:, : 10], df_x2.iloc[:, : 10]], axis=1, ignore_index=True)
df_v_x = pd.concat([df_v1.iloc[:, : 10], df_v2.iloc[:, : 10]], axis=1, ignore_index=True)

# df_p_x, df_v_x = df_x2, df_v2

print('Human Labeling...')

p_x_reg,v_x_reg, p_x_anom , v_x_anom  = human_labeling(spec_obs_true,op, df_p_x, df_v_x , cost_limit) # labeling traces
num_safe = len(v_x_reg.columns) #number of safe rollout traces
num_unsafe = len(v_x_anom.columns) #number of safe rollout traces
percentage_safe = percentage(len(v_x_reg.columns),len(v_x_anom.columns))
safe_p = [percentage_safe]
size_xreg = [len(v_x_reg.columns)]
print('Initial Percentage of safe traces (without alteration): ',percentage_safe,'%')


# Running our algorithm
percentage_safe = 0
print('Begining bi-level optimization...(our algorithm)')
it = 1 #iteration number
specs = []
best_Ys = []
final_spec = []
while percentage_safe < delta:
    # pop_df = initialize(x_reg, y_reg, x_anom, y_anom, rng)
    # infered_STL = GA(pop_df,x_reg, y_reg, x_anom, y_anom, rng) #infer full STL from traces (template+parameters)
    rng = [0,5]
    print('Optimizing parameters...')
    infered_STL_obs, best_Y = GP_opt(50,spec_obs_template,op, p_x_reg,v_x_reg, p_x_anom , v_x_anom,cost_limit, rng) #Infer parameters given STL template
    # full_spec_new = full_spec(spec_goal, infered_STL_obs)
    print('----Infered STL_Obs is----', infered_STL_obs)
    specs.append(infered_STL_obs)
    best_Ys.append(best_Y)
    # Train agorithm and get rollout dataset
    df_p_x, df_v_x = train_alg('TD3Lag', env_id,total_steps,steps_per_epoch, spec_obs, num_rollout) # Running TD3 to get results for BL1
    
    # Human labeling of rollout dataset
    p_x_r,v_x_r, p_x_a , v_x_a  = human_labeling(spec_obs_true,op, df_p_x, df_v_x , cost_limit)  # Label rollout traces from RL
    
    # computing percentage of safe traces in rollout dataset
    percentage_safe = percentage(len(v_x_r.columns),len(v_x_a.columns))
    safe_p.append(percentage_safe)
    
    # Appending new data for regukar and anomalous traces
    p_x_reg = pd.concat([p_x_reg, p_x_r], axis=1, ignore_index=True)
    v_x_reg = pd.concat([v_x_reg, v_x_r], axis=1, ignore_index=True) 
    p_x_anom = pd.concat([p_x_anom, p_x_a], axis=1, ignore_index=True)
    v_x_anom= pd.concat([v_x_anom, v_x_a], axis=1, ignore_index=True)

    size_xreg.append(len(v_x_reg.columns))
    num_safe = len(v_x_r.columns) #number of safe rollout traces
    num_unsafe = len(v_x_a.columns)
    print('Iteration number: ',it,'---',percentage_safe,'% safe traces')
    it += 1
    
final_spec.append(infered_STL_obs)

# Run Baseline1 - cost not considered
print('Running BL1')
df_p_x, df_v_x = train_alg('TD3', env_id, total_steps, steps_per_epoch, spec_obs, num_rollout) # Running TD3 to get results for BL1
p_x_reg,v_x_reg, p_x_anom , v_x_anom = human_labeling(spec_obs_true,op, df_p_x, df_v_x , cost_limit) # labeling traces
num_safe = len(v_x_reg.columns) #number of safe rollout traces
num_unsafe = len(v_x_anom.columns) #number of safe rollout traces
percentage_safe = percentage(len(v_x_reg.columns),len(v_x_anom.columns))
safe_p = [percentage_safe]
size_xreg = [len(v_x_reg.columns)]
print('BL1 percentage of safe traces (without alteration): ',percentage_safe,'%')  

# Run Baseline2 - True STL constraint used  
print('Running BL2')
df_p_x, df_v_x = train_alg('TD3Lag', env_id,total_steps, steps_per_epoch, spec_obs, num_rollout) # Running TD3 to get results for BL1
p_x_reg,v_x_reg, p_x_anom , v_x_anom  = human_labeling(spec_obs_true,op, df_p_x, df_v_x , cost_limit) # labeling traces
num_safe = len(v_x_reg.columns) #number of safe rollout traces
num_unsafe = len(v_x_anom.columns) #number of safe rollout traces
percentage_safe_BL2 = percentage(len(v_x_reg.columns),len(v_x_anom.columns))
safe_p = [percentage_safe]
size_xreg_BL2 = [len(v_x_reg.columns)]
print('BL2 Percentage of safe traces: ',percentage_safe_BL2,'%')
