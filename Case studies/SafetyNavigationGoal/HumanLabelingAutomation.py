#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:29:41 2023

@author: lay0005
"""

import rtamt
import math
import pandas as pd
from BOSNG import robustness,specification,signal,size
#from InitialDataset import initial_trace
#x_trace, y_trace = initial_trace()


def human_labeling(spec,operation, x , y, cost_limit):
    x_reg = []
    y_reg = []    
    x_anom = []
    y_anom = []
    
    
    for i in range(len(x.columns)):
        if operation == '>' or operation == '<' or operation == 'not':
            rob = min(robustness(signal(x, y,i), specification(spec)))
        elif operation == '&':
            first_part = spec[:spec.find('&')]
            second_part = spec[spec.find('&')+1:]

            if size(first_part) == 1 or size(second_part) == 1:
                rob = min(robustness(signal(x, y,i), specification(spec)))
            else:
                rob1 = robustness(signal(x, y,i), specification(first_part))
                rob2 = robustness(signal(x, y,i), specification(second_part))
                rob = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))
        elif operation == '|' :
            first_part = spec[:spec.find('&', spec.find('&') + 1)]
            second_part = spec[spec.find('&', spec.find('&') + 1)+1:]
            if size(first_part) == 1 or size(second_part) == 1:
                rob = min(robustness(signal(x, y,i), specification(spec)))
            else:
                rob = robustness(signal(x, y,i), specification(spec))[0]
        elif operation == 'G' or operation == 'F' or operation == 'U':  
            rob =robustness(signal(x, y,i), specification(spec))[0]
        if rob >=0:
            x_reg.append([j for j in x[i]])
            y_reg.append([j for j in y[i]])
            # print('Label : safe')
        if rob <0:
            x_anom.append([j for j in x[i]])
            y_anom.append([j for j in y[i]])
            # print('Label : unsafe')
    
    #If cost_limit != 0 
      
    if cost_limit > 0:
        j = 0
        i = 0 
        while i < len(x_anom): #moving traces with number of violated timesteps under "cost_limit" to regular df
        # Define the reference points
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
            # print(len(violated_timesteps))
            # Print all time steps where the STL formula was violated
            if violated_timesteps:
                if len(violated_timesteps) <= cost_limit:
                    print(len(violated_timesteps),'changed')
                    x_reg.append(x_values)
                    y_reg.append(y_values)  
                    x_anom.pop(i)
                    y_anom.pop(i)  
                    i+= 1
                    j+=1
            i+=1
        print('Final sizes : x_reg = ',len(x_reg),', x_anom = ', len(x_anom), 'with', j, 'change(s)')

    print('Sizes:  x_reg = ',len(x_reg),', x_anom = ', len(x_anom))     
    x_reg = pd.DataFrame(x_reg).T
    y_reg = pd.DataFrame(y_reg).T
    x_anom = pd.DataFrame(x_anom).T
    y_anom = pd.DataFrame(y_anom).T

    return x_reg,y_reg, x_anom , y_anom
         
#x_reg,y_reg,x_anom,y_anom = Human_labling('F(x>=4 & y>=4)','F',x_trace,y_trace)
