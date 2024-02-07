#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:59:56 2023

@author: lay0005
"""
import omnisafe
import pandas as pd
import argparse
from omnisafe.utils.plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt

def train_alg(alg, env_id, total_steps, steps_per_epoch, spec_obs, num_rollout):
    steps_per_epoch = steps_per_epoch
    if alg.lower() == 'td3':
        custom_cfgs = {
            'seed': 0,
            'train_cfgs': {
                'total_steps': total_steps,
                'vector_env_nums': 1,
                'parallel': 1,
        
            },
            'algo_cfgs': {
                'steps_per_epoch': 1000,        
                'update_iters': 1,
            },
            'logger_cfgs': {
                'use_wandb': False,
            },
        }
        
    else: # to include cost limit
        custom_cfgs = {
            'seed': 0,
            'train_cfgs': {
                'total_steps': total_steps,
                'vector_env_nums': 1,
                'parallel': 1,
            },
            'algo_cfgs': {
                'steps_per_epoch': 1000,        
                'update_iters': 1,
            },
            'lagrange_cfgs': {
                'cost_limit': 0.0,        
            },
            'logger_cfgs': {
                'use_wandb': False,
            },
        }
        
    agent = omnisafe.Agent(alg, env_id,spec_obs,custom_cfgs=custom_cfgs)
    agent.learn()
    # agent.plot(smooth=1)
    # agent.render(num_episodes=1, width=256, height=256)
    LOG_DIR = agent.LOG_DIR()
    print('============',LOG_DIR)
    states = Evaluate(LOG_DIR,spec_obs,env_id,num_episodes=num_rollout)
    # states = agent.evaluate(spec_obs,num_episodes=num_rollout)
    
    x_trace = []
    y_trace = []
    for i in states:
        x = []
        y = []
        for j in i:
            x.append(j[0])
            y.append(j[1])
        x_trace.append(x)
        y_trace.append(y)
            
    max_length = steps_per_epoch
        
    for i in x_trace:
        while len(i) < max_length:
            i.append(i[-1])
            
    for i in y_trace:
        while len(i) < max_length:
            i.append(i[-1])    
            
    df_x = pd.DataFrame()
    df_y = pd.DataFrame()

    for i in range(len(x_trace)):
        df_x.loc[:, len(df_x.columns)] = x_trace[i]
        df_y.loc[:, len(df_y.columns)] = y_trace[i]
        
    return df_x, df_y

def percentage(reg, anom):
    return (reg/(reg+anom))*100

def full_spec(spec_goal, spec_obs):
    return 'F(' + spec_goal +') & G(not('+ spec_obs +'))'

def plot_runs(dirs):
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--logdir', nargs='*')
        parser.add_argument('--legend', '-l', nargs='*')
        parser.add_argument('--xaxis', '-x', default='Steps')
        parser.add_argument('--value', '-y', default='Rewards', nargs='*')
        parser.add_argument('--count', action='store_true')
        parser.add_argument('--smooth', '-s', type=int, default=1)
        parser.add_argument('--select', nargs='*')
        parser.add_argument('--exclude', nargs='*')
        parser.add_argument('--cost_limit', nargs='*')
        parser.add_argument('--estimator', default='mean')
        args = parser.parse_args()
        
        args.logdir = dirs
        args.cost_limit = 25
        
        plotter = Plotter()
        plotter.make_plots(
            args.logdir,
            args.legend,
            args.xaxis,
            args.value,
            args.count,
            args.cost_limit,
            smooth=args.smooth,
            select=args.select,
            exclude=args.exclude,
            estimator=args.estimator,
        )


# def under_25(x_reg,y_reg, x_anom , y_anom):
#     j = 0
#     for i in range(len(x_anom.columns)):
#         import math
    
#         # Define the reference points
#         reference_points = [(-0.75, 1.0), (1.0, 0.20), (-1.40, 0.70), (-0.50, -0.30), (0.25, 0.90), (0.0, -1.50),(-1.9, 1.00),(1.00, -1.00)]        
#         # Sample trace data as lists of (x, y) values
#         x_values = x_anom[i] # Replace with your actual x values
#         y_values = y_anom[i]  # Replace with your actual y values
        
#         violated_timesteps = []
        
#         # Iterate through the trace
#         for timestep, (x, y) in enumerate(zip(x_values, y_values)):
#             # Check the STL formula for each reference point
#             violated = any(math.sqrt((x - x_ref)**2 + (y - y_ref)**2) < 0.4 for x_ref, y_ref in reference_points)
        
#             if violated:
#                 violated_timesteps.append(timestep)
    
#         # Print all time steps where the STL formula was violated
#         if violated_timesteps:
#             if len(violated_timesteps) >= 25:
#                 print(len(violated_timesteps))
#                 j+=1
#             else:
#                 print(f"STL formula violated for {len(violated_timesteps)} timesteps and over 25 steps violated is {j} of {len(df_xanom.columns)}")
#                 x_reg = pd.concat([x_reg, x_r], axis=1, ignore_index=True)
#                 y_reg = pd.concat([y_reg, y_r], axis=1, ignore_index=True) 
#                 x_anom = pd.concat([x_anom, x_a], axis=1, ignore_index=True)
#                 y_anom= pd.concat([y_anom, y_a], axis=1, ignore_index=True)
#     else:
#         print("STL formula was not violated in the trace.")
        
#     return add_to_regx,add_to_regy
def organize_rollout(file_name):
    import pickle
    
    open_file = open(file_name, "rb")
    
    states = pickle.load(open_file)
    
    open_file.close()
    print(len(states))
    x_trace = []
    y_trace = []
    for i in states:
        x = []
        y = []
        for j in i:
            x.append(j[0])
            y.append(j[1])
        x_trace.append(x)
        y_trace.append(y)
     
    df_x = pd.DataFrame()
    df_y = pd.DataFrame()

    for i in range(len(x_trace)):
        df_x.loc[:, len(df_x.columns)] = x_trace[i]
        df_y.loc[:, len(df_y.columns)] = y_trace[i]
        
    return df_x, df_y


def Evaluate(LOG_DIR,spec_obs,name,num_episodes):
    import os

    import omnisafe
    it = []
    # Just fill your experiment's log directory in here.
    # Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
    evaluator = omnisafe.Evaluator(spec_obs,render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    for item in scan_dir:
        it.append(item)
    it.sort(key=lambda item: int(item.name.split('-')[1].split('.')[0]))
    item = it[-1]

    # print(item)
    if item.is_file() and item.name.split('.')[-1] == 'pt':
        evaluator.load_saved(
            save_dir=LOG_DIR,
            model_name=item.name,
            camera_name='track',
            width=256,
            height=256,
        )
        (episode_rewards, episode_costs,states) = evaluator.evaluate(spec_obs,num_episodes=num_episodes)
    scan_dir.close()
    # ********** saving states ***********
    import pickle
    file_name = "states_"+name+".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(states, open_file)

    open_file.close()

    open_file = open(file_name, "rb")
    open_file.close()

    return states
    
def plot_BO_Convergence(best_Y, expname):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
  
    ######################################################
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # set width of bar
    barWidth = 0
    # fig, ax = plt.subplots(figsize =(12, 8))
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)
    #########################################################
    
    plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=12)
    plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=12)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True,linestyle=':', color='gray')
    ax.set_axisbelow(True)
    
    # Make the plot
    # exp=['exp1', 'exp2' , 'exp3','exp4','exp5','exp6','exp7','exp8','exp9','exp10']
    plt.errorbar(list(range(len(best_Y))),best_Y, linewidth=2 ,fmt = '--o',color = 'tab:blue', ms=7, label ='SPG')

    # Adding Xticks
    plt.xlabel('Iteration', family='serif', fontsize=14,  labelpad=5)
    plt.ylabel('Best f', family='serif', fontsize=14,  labelpad=10)
    #weight='bold',
    leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
    ltext  = leg.get_texts()
    plt.setp(ltext, family='serif', fontsize=12)
    # plt.xlim(100,10)
    # plt.ylim(0.6,0.9)
    plt.title('Convergence of BO')
    plt.tight_layout(pad=0.5)
    #plt.legend()
    fig.set_size_inches(6, 4) # Resize the figure for better display in the notebook
    # plt.xticks([r + 1.5*barWidth for r in range(10)],list(range(n)))
    #ax.set_title('Comparison between methods over 20 BO iterations',fontsize=14)
    # ax.yaxis.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.5,alpha = 0.3)
    plt.savefig(expname + ' BO convergence.png')
    plt.show()
    return best_Y
 
        
# SHV Dirs
#dirs = ['/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-25-14-28-08 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-26-01-05-31 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-26-01-07-08 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-26-12-58-20 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-26-13-00-07 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-27-12-09-31 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-27-12-09-39 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-27-12-10-02 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/SHV final/seed-000-2023-10-28-18-36-38 (copy)']

# SPC1 Dirs
#dirs = ['/home/WVU-AD/lay0005/Downloads/runs/TD3Lag-{SafetyPointCircle1-v0}/seed-000-2023-10-31-23-54-54 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/TD3Lag-{SafetyPointCircle1-v0}/seed-000-2023-10-31-23-55-30 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/spc1 final/seed-000-2023-10-30-22-49-43 (copy)','/home/WVU-AD/lay0005/Downloads/runs/spc1 final/seed-000-2023-10-30-22-50-21 (copy)','/home/WVU-AD/lay0005/Downloads/runs/spc1 final/seed-000-2023-10-30-22-51-56 (copy)','/home/WVU-AD/lay0005/Downloads/runs/spc1 final/seed-000-2023-10-31-23-54-12 (copy)']

#SHCV1 Dirs
# dirs = ['/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-04-12-58-28 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-05-11-43-02 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-04-12-56-08 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-05-11-43-31 (copy)','/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-06-11-34-51 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-04-12-56-28 (copy)', '/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-05-11-43-17 (copy)','/home/WVU-AD/lay0005/Downloads/runs/shcv final/seed-000-2023-11-06-11-34-37 (copy)']

