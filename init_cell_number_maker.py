#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:25:17 2024

@author: hossein
"""





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import subprocess
import os


N_runs = 5
WT_init_frac_interval = np.array([1.0, 1.0])

eps = 0.0001

Init_numbers_array = np.zeros((2, N_runs), dtype=int)

if np.mean(WT_init_frac_interval) > (1.0 - eps):
    # pure WT
    min_size = 10
    max_size = 70
    init_num_cells = np.random.randint(min_size,max_size+1,N_runs)
    
    Init_numbers_array[0,:] = init_num_cells.copy()
    Init_numbers_array[1,:] = 0 * init_num_cells.copy()
    
elif np.mean(WT_init_frac_interval) < eps:
    # pure Cancer
    min_size = 10
    max_size = 100
    init_num_cells = np.random.randint(min_size,max_size+1,N_runs)
    
    Init_numbers_array[0,:] = 0 * init_num_cells.copy()
    Init_numbers_array[1,:] = init_num_cells.copy()
    
    
else:
    # mixed
    
    min_size_WT = 20
    max_size_WT = 100
    
    min_size_C = 10
    max_size_C = 70
    
    # min_size_tot = 
    # max_size_tot = 
    
    for runC in range(N_runs):
        init_num_cells_WT = np.random.randint(min_size_WT,max_size_WT+1,1)
        init_num_cells_C  = np.random.randint(min_size_C,max_size_C+1,1)
        
        WT_frac = init_num_cells_WT/(init_num_cells_WT+init_num_cells_C)
        
        if ( WT_frac > WT_init_frac_interval[0] and WT_frac < WT_init_frac_interval[1]):
            
            Init_numbers_array[0, runC] = init_num_cells_WT
            Init_numbers_array[1, runC] = init_num_cells_C
    

np.savetxt("Init_numbers_array.csv", Init_numbers_array, delimiter=',', fmt='%d')

for runC in range(N_runs):
    folderName = "run_"+str(runC+1)
    
    Init_numbers_file = Init_numbers_array[:,runC]
    
    np.savetxt(folderName +"/"+"Init_numbers.csv", Init_numbers_file, delimiter=',', fmt='%d')
