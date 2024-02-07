#!/usr/bin/env pv_xthon3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:29:41 2023

@author: lav_x0005
"""

import rtamt
import math
import pandas as pd
from BOSVHC import robustness,specification,signal,size
#from InitialDataset import initial_trace
#p_x_trace, v_x_trace = initial_trace()


def human_labeling(spec,operation, p_x , v_x, cost_limit):
    p_x_reg = []
    v_x_reg = []    
    p_x_anom = []
    v_x_anom = []
       
    for i in range(len(p_x.columns)):
        if operation == '>' or operation == '<' or operation == 'not':
            rob = min(robustness(signal(p_x, v_x,i), specification(spec)))
        elif operation == '&':
            first_part = spec[:spec.find('&')]
            second_part = spec[spec.find('&')+1:]

            if size(first_part) == 1 or size(second_part) == 1:
                rob = min(robustness(signal(p_x, v_x,i), specification(spec)))
            else:
                rob1 = robustness(signal(p_x, v_x,i), specification(first_part))
                rob2 = robustness(signal(p_x, v_x,i), specification(second_part))
                rob = min(math.copv_xsign(1,rob1[0]), math.copv_xsign(1,rob2[0]))
        elif operation == '|' :
            first_part = spec[:spec.find('&', spec.find('&') + 1)]
            second_part = spec[spec.find('&', spec.find('&') + 1)+1:]
            if size(first_part) == 1 or size(second_part) == 1:
                rob = min(robustness(signal(p_x, v_x,i), specification(spec)))
            else:
                rob = robustness(signal(p_x, v_x,i), specification(spec))[0]
        elif operation == 'G' or operation == 'F' or operation == 'U':  
            rob =robustness(signal(p_x, v_x,i), specification(spec))[0]
        if rob >=0:
            p_x_reg.append([j for j in p_x[i]])
            v_x_reg.append([j for j in v_x[i]])

        if rob <0:
            p_x_anom.append([j for j in p_x[i]])
            v_x_anom.append([j for j in v_x[i]])

            
    if cost_limit > 0:
        j = 0
        i = 0 
        # alter labels based on cost limit
        while i < len(v_x_anom): # moving under cost_limit traces to regular df
            # Define the reference points
            reference_points = [(3.20960)]
            p_x_values = p_x_anom[i] # Replace with v_xour actual p_x values
            v_x_values = v_x_anom[i]  # Replace with v_xour actual v_x values
        
            violated_timesteps = []
            
            # Iterate through the trace
            for timestep, (p_x_an, v_x_an) in enumerate(zip(p_x_values, v_x_values)):
                # Check the STL formula for each reference point
                violated = any (v_x_an> v_x_ref for v_x_ref in reference_points)
            
                if violated:
                    violated_timesteps.append(timestep)
        
            # Print all time steps where the STL formula was violated
            if violated_timesteps:
                if len(violated_timesteps) <= cost_limit:
                    print(len(violated_timesteps))
                    p_x_reg.append(p_x_values)
                    v_x_reg.append(v_x_values)  
                    p_x_anom.pop(i)
                    v_x_anom.pop(i)  
                    i+= 1
                    j+=1
            i+=1
        print('Final sizes : x_reg = ',len(p_x_reg),', x_anom = ', len(p_x_anom), 'with', j, 'change(s)')

    print('Sizes:  x_reg = ',len(p_x_reg),', x_anom = ', len(p_x_anom))  
     
    p_x_reg = pd.DataFrame(p_x_reg).T
    v_x_reg = pd.DataFrame(v_x_reg).T
    p_x_anom = pd.DataFrame(p_x_anom).T
    v_x_anom = pd.DataFrame(v_x_anom).T

    
    
    return p_x_reg,v_x_reg, p_x_anom , v_x_anom
         
#p_x_reg,v_x_reg,p_x_anom,v_x_anom = Human_labling('F(p_x>=4 & v_x>=4)','F',p_x_trace,v_x_trace)
