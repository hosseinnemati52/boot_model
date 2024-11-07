#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:22:25 2024

@author: hossein
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:34:39 2024

@author: Nemat002
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import subprocess
import os
import seaborn as sns

CYCLING_STATE =   (1)
G1_ARR_STATE =    (-1)
G0_STATE =        (-2)
DIFF_STATE =      (-3)
APOP_STATE =      (-4)
CA_CELL_TYPE =    (1)
WT_CELL_TYPE =    (0)
NULL_CELL_TYPE =  (-1)


def parse_array(value):
    # Remove surrounding brackets and extra spaces
    value = value.strip('[]')
    # Handle multi-line arrays (convert ';' to '],[')
    if ';' in value:
        value = value.replace(';', ',')
    # Add surrounding brackets to make it a proper list format
    value = f'[{value}]'
    
    try:
        # Convert to a NumPy array
        return np.array(eval(value))
    except (SyntaxError, NameError) as e:
        print(f"Error parsing array: {e}")
        return None

def read_custom_csv(filename):
    # Initialize dictionary to hold variables
    variables = {}
    
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            # Skip comments or empty lines
            if line.strip() == '' or line.strip().startswith('##'):
                continue
            
            # Split the line into key and value
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Process based on the type of the value
            if value.startswith('[') and value.endswith(']'):
                # This is a list or array
                variables[key] = parse_array(value)
                
            elif re.match(r'^[\d.]+$', value):
                # This is a number (int or float)
                variables[key] = float(value) if '.' in value else int(value)
            
            elif re.match(r'^[\w]+$', value):
                # This is a string or keyword
                variables[key] = value
            
    # Extract variables from the dictionary
    N_UpperLim = variables.get('N_UpperLim', None)
    NTypes = variables.get('NTypes', None)
    typeR0 = variables.get('typeR0', None)
    typeR2PI = variables.get('typeR2PI', None)
    typeTypeEpsilon = variables.get('typeTypeEpsilon', None)
    
    typeGamma = variables.get('typeGamma', None)
    typeTypeGammaCC = variables.get('typeTypeGammaCC', None)
    typeTypeF_rep_max = variables.get('typeTypeF_rep_max', None)
    typeTypeF_abs_max = variables.get('typeTypeF_abs_max', None)
    R_eq_coef = variables.get('R_eq_coef', None)
    R_cut_coef_force = variables.get('R_cut_coef_force', None)
    
    typeFm = variables.get('typeFm', None)
    typeDr = variables.get('typeDr', None)
    
    G1Border = variables.get('G1Border', None)
    
    typeOmega = variables.get('typeOmega', None)
    typeBarrierW = variables.get('typeBarrierW', None)
    typeSigmaPhi = variables.get('typeSigmaPhi', None)
    typeSigmaFit = variables.get('typeSigmaFit', None)
    barrierPeakCoef = variables.get('barrierPeakCoef', None)
    typeFit0 = variables.get('typeFit0', None)
    Fit_Th_Wall = variables.get('Fit_Th_Wall', None)
    Fit_Th_G0 = variables.get('Fit_Th_G0', None)
    Fit_Th_Diff = variables.get('Fit_Th_Diff', None)
    Fit_Th_Apop = variables.get('Fit_Th_Apop', None)
    
    maxTime = variables.get('maxTime', None)
    dt = variables.get('dt', None)
    dt_sample = variables.get('dt_sample', None)
    samplesPerWrite = variables.get('samplesPerWrite', None)
    printingTimeInterval = variables.get('printingTimeInterval', None)
    
    R_cut_coef_game = variables.get('R_cut_coef_game', None)
    tau = variables.get('tau', None)
    typeTypePayOff_mat_real_C = variables.get('typeTypePayOff_mat_real_C', None)
    typeTypePayOff_mat_real_F1 = variables.get('typeTypePayOff_mat_real_F1', None)
    typeTypePayOff_mat_real_F2 = variables.get('typeTypePayOff_mat_real_F2', None)
    typeTypePayOff_mat_imag_C = variables.get('typeTypePayOff_mat_imag_C', None)
    typeTypePayOff_mat_imag_F1 = variables.get('typeTypePayOff_mat_imag_F1', None)
    typeTypePayOff_mat_imag_F2 = variables.get('typeTypePayOff_mat_imag_F2', None)
    
    # typeOmega0 = variables.get('typeOmega0', None)
    # typeOmegaLim = variables.get('typeOmegaLim', None)
    newBornFitKey = variables.get('newBornFitKey', None)
    
    initConfig = variables.get('initConfig', None)
    
    return {
        'N_UpperLim': N_UpperLim,
        'NTypes': NTypes,
        'typeR0': typeR0,
        'typeR2PI': typeR2PI,
        'typeTypeEpsilon': typeTypeEpsilon,
        
        'typeGamma': typeGamma,
        'typeTypeGammaCC': typeTypeGammaCC,
        'typeTypeF_rep_max': typeTypeF_rep_max,
        'typeTypeF_abs_max': typeTypeF_abs_max,
        'R_eq_coef': R_eq_coef,
        'R_cut_coef_force': R_cut_coef_force,
        
        'typeFm': typeFm,
        'typeDr': typeDr,
        
        'G1Border' : G1Border,
        
        'typeOmega' : typeOmega,
        'typeBarrierW' : typeBarrierW,
        'typeSigmaPhi' : typeSigmaPhi,
        'typeSigmaFit' : typeSigmaFit,
        'barrierPeakCoef' : barrierPeakCoef,
        'typeFit0' : typeFit0,
        'Fit_Th_Wall' : Fit_Th_Wall,
        'Fit_Th_G0' : Fit_Th_G0,
        'Fit_Th_Diff' : Fit_Th_Diff,
        'Fit_Th_Apop' : Fit_Th_Apop,
        
        'maxTime': maxTime,
        'dt': dt,
        'dt_sample': dt_sample,
        'samplesPerWrite': samplesPerWrite,
        'printingTimeInterval': printingTimeInterval,
                
        'R_cut_coef_game' : R_cut_coef_game,
        'tau' :tau,
        'typeTypePayOff_mat_real_C' :typeTypePayOff_mat_real_C,
        'typeTypePayOff_mat_real_F1' :typeTypePayOff_mat_real_F1,
        'typeTypePayOff_mat_real_F2' :typeTypePayOff_mat_real_F2,
        'typeTypePayOff_mat_imag_C' :typeTypePayOff_mat_imag_C,
        'typeTypePayOff_mat_imag_F1' :typeTypePayOff_mat_imag_F1,
        'typeTypePayOff_mat_imag_F2' :typeTypePayOff_mat_imag_F2,
        
        'newBornFitKey' : newBornFitKey,
        
        'initConfig': initConfig
    }

def read_custom_csv_pp(filename):
    # Initialize dictionary to hold variables
    variables = {}
    
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            # Skip comments or empty lines
            if line.strip() == '' or line.strip().startswith('##'):
                continue
            
            # Split the line into key and value
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Process based on the type of the value
            if value.startswith('[') and value.endswith(']'):
                # This is a list or array
                variables[key] = parse_array(value)
                
            elif re.match(r'^[\d.]+$', value):
                # This is a number (int or float)
                variables[key] = float(value) if '.' in value else int(value)
            
            elif re.match(r'^[\w]+$', value):
                # This is a string or keyword
                variables[key] = value
            
    # Extract variables from the dictionary
    frame_plot_switch = variables.get('frame_plot_switch', None)
    
    
    return {
        'frame_plot_switch': frame_plot_switch
            }

def stats_plotter(fileName):
    
    plt.figure()
    
    plt.plot(time, alive_stat, label='tot')
    plt.plot(time, C_alive_stat, label='Ca', color='g')
    plt.plot(time, C_apop_stat, label='Ca_apop', linestyle='dashed')
    plt.plot(time, WT_alive_stat, label='WT', color='m')
    plt.plot(time, WT_cyc_stat, label='WT_cyc', linestyle='dashed')
    plt.plot(time, WT_g1_cyc_stat, label='WT_g1_cyc', linestyle='dashed')
    plt.plot(time, WT_g1_arr_stat, label='WT_g1_arr', linestyle='dashed')
    plt.plot(time, WT_g1_tot_stat, label='WT_g1_tot', linestyle='dashed')
    plt.plot(time, WT_g0_stat, label='WT_g0', linestyle='dashed')
    plt.plot(time, WT_diff_stat, label='WT_diff', linestyle='dashed')
    plt.plot(time, WT_diff_stat+WT_g0_stat, label='WT_g0_diff', linestyle='dashed')
    plt.plot(time, WT_apop_stat, label='WT_apop')
    
    plt.xlabel("time (h)")
    plt.ylabel("Number")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fileName+".PNG", dpi=200)
    # plt.close()
    
    plt.figure()
    
    plt.plot(time, alive_stat/alive_stat[0], label='tot')
    plt.plot(time, C_alive_stat/C_alive_stat[0], label='Ca', color='g')
    plt.plot(time, WT_alive_stat/WT_alive_stat[0], label='WT', color='m')
    # plt.plot(time, WT_cyc_stat/WT_cyc_stat[0], label='WT_cyc', linestyle='dashed')
    # plt.plot(time, WT_g1_cyc_stat/WT_g1_cyc_stat[0], label='WT_g1_cyc', linestyle='dashed')
    # plt.plot(time, WT_g1_arr_stat/WT_g1_arr_stat[0], label='WT_g1_arr', linestyle='dashed')
    # plt.plot(time, WT_g1_tot_stat, label='WT_g1_tot', linestyle='dashed')
    # plt.plot(time, WT_g0_stat, label='WT_g0', linestyle='dashed')
    # plt.plot(time, WT_diff_stat, label='WT_diff', linestyle='dashed')
    # plt.plot(time, WT_apop_stat, label='WT_apop', linestyle='dashed')
    
    plt.xlabel("time (h)")
    plt.ylabel("Normalized Number")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fileName+"_norm.PNG", dpi=200)
    # plt.close()
    
    plt.figure()
    plt.plot(time, WT_g1_tot_stat/WT_alive_stat, label='g1_frac')
    # plt.plot(time, WT_g0_stat/WT_alive_stat, label='g0_frac')
    # plt.plot(time, WT_diff_stat/WT_alive_stat, label='diff_frac')
    plt.plot(time, (WT_g0_stat+WT_diff_stat)/WT_alive_stat, label='g0_diff_frac')
    
    plt.xlabel("time (h)")
    plt.ylabel("fractions")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fileName+"_fracs.PNG", dpi=200)
    # plt.close()
    
    
    return 0

def plotter(t, snapshotInd):
    # size_vec = np.zeros(N_sph_tot)
    # scale = 4

    # norm = mcolors.Normalize(vmin=cell_sph.min(), vmax=cell_sph.max())
    # cmap = cm.viridis
    
    # fig, ax = plt.subplots()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    WT_indices = (cellType==WT_CELL_TYPE) & (cellState!=APOP_STATE)
    C_indices = (cellType==CA_CELL_TYPE) & (cellState!=APOP_STATE)
    
    thresh = 1e-8
    
    if len(WT_indices[WT_indices==1])>0:
        WT_fitness_max = np.max(cellFitness[WT_indices, 0])
        WT_fitness_min = np.min(cellFitness[WT_indices, 0])
        
        if abs(WT_fitness_max-WT_fitness_min)<thresh:
            WT_fitness_max = WT_fitness_min + thresh
    else:
        WT_fitness_max = 0
        WT_fitness_min = 0
    
    
    
    if len(C_indices[C_indices==1])>0:
        C_fitness_max = np.max(cellFitness[C_indices, 0])
        C_fitness_min = np.min(cellFitness[C_indices, 0])
        
        if abs(C_fitness_max-C_fitness_min)<thresh:
            C_fitness_max = C_fitness_min + thresh
    else:
        C_fitness_max = 0
        C_fitness_min = 0
    
    
    normWT = mcolors.Normalize(vmin = WT_fitness_min , vmax = WT_fitness_max)
    normC  = mcolors.Normalize(vmin =  C_fitness_min , vmax =  C_fitness_max)

    for i in range(NCells):
        # # size_vec[i] = scale * r_spheres[type_sph[i]]
        # circle = patches.Circle((cellX[i], cellY[i]), radius=cellR[i], edgecolor='k', facecolor='g'*(cellType[i]) + 'violet'*(1-cellType[i]), alpha=0.8)
        # ax1.add_patch(circle)
        
        if cellState[i]==APOP_STATE:
            continue
        
        
        if cellType[i] == 1:
            # normalized_fitness = normC(cellFitness[i][0])
            normalized_fitness = normC(0.5* ( cellFitness[i][0] + C_fitness_max))
            color = cm.Greens(normalized_fitness) 
        else:
            # normalized_fitness = normWT(cellFitness[i][0])
            normalized_fitness = normWT(0.5* ( cellFitness[i][0] +  WT_fitness_max))
            color = cm.Purples(normalized_fitness) 
        
        if cellState[i]==DIFF_STATE or cellState[i]==G0_STATE:
            polgon = patches.RegularPolygon((cellX[i], cellY[i]),numVertices=5, radius=cellR[i], edgecolor='k', facecolor=color, alpha=0.8)
            ax1.add_patch(polgon)
        else:
            circle = patches.Circle((cellX[i], cellY[i]), radius=cellR[i], edgecolor='k', facecolor=color, alpha=0.8)
            ax1.add_patch(circle)
    # plt.scatter(x_sph, y_sph, s=size_vec, c=cell_sph, alpha=1, cmap='viridis')
    ax1.axis("equal")
    # plt.xlim((0,Lx))
    # plt.ylim((0,Ly))
    ax1.set_aspect('equal', adjustable='box')
    
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array(cell_sph)
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Cell Value')

    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    title = 't = {:.3f}'.format(t)
    ax1.set_title(title)
    # ax1.set_title('Colored Scatter Plot with Circles')
    # plt.colorbar()  # Show color scale
    
    file_name = 'frames/frame_'+str(int(snapshotInd))+'.PNG'
    # ax1.title(title)
    # ax1.grid()
    
    ax2.plot(time, WT_alive_stat[:len(time)]/WT_alive_stat[0], label='WT(t)/WT(0)', color='violet')
    ax2.plot(time, C_alive_stat[:len(time)]/C_alive_stat[0], label='C(t)/C(0)', color='g')
    ax2.plot(time, C_alive_stat[:len(time)]/alive_stat[:len(time)], label='C(t)/tot(t)', color='g', linestyle='dashed')
    ax2.plot(time, WT_alive_stat[:len(time)]/alive_stat[:len(time)], label='WT(t)/tot(t)', color='violet', linestyle='dashed')
    ax2.set_yscale("log")
    
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel('time')
    
    
    # ax1.set_ylabel('Y-axis')
    
    plt.savefig(file_name, dpi=150)
    # plt.show()
    plt.close()
    
    return 0

def overal_plotter(exp_data_load_switch, save_switch):
    
    try:
        directory = "overal_pp"
        os.makedirs(directory, exist_ok=True)
    except:
        pass
    
    save_array = np.zeros((3, len(time)))
    
    ### absolute number
    list_of_data = [alive_stat_overal, 
                    C_alive_stat_overal, 
                    C_apop_stat_overal, 
                    WT_alive_stat_overal, 
                    WT_apop_stat_overal,
                    WT_cyc_stat_overal, 
                    WT_g0_diff_stat_overal,
                    WT_g1_arr_stat_overal,
                    WT_g1_cyc_stat_overal,
                    WT_g1_tot_stat_overal,
                    WT_sg2m_stat_overal]
    
    labels_of_data = ['tot', 
                      'C', 
                      'C_apop',
                      'WT',
                      'WT_apop',
                      'WT_cyc',
                      'WT_g0_diff',
                      'WT_g1_arr',
                      'WT_g1_cyc',
                      'WT_g1_tot',
                      'WT_sg2m']
    
    plt.figure()
    
    
    for dataC in range(len(list_of_data)):
        data = list_of_data[dataC].copy()
        data_label = labels_of_data[dataC]
        
        x = time
        y = np.mean(data,axis=0)
        y_err = np.std(data,axis=0)
        plt.plot(x, y, label=data_label)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.3)
        
        save_array = 0.0 * save_array
        save_array[0,:] = x
        save_array[1,:] = y
        save_array[2,:] = y_err
        np.savetxt("overal_pp"+"/"+data_label+"_ov_pl"+".txt", save_array, fmt='%.4f', delimiter=',')
        
        
    
    # if exp_data_load_switch:
    #     # x, y, y_err = exp_data_loader()
    #     exp_array = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
        
    #     x = exp_array[0,:]
    #     y = exp_array[1,:]
    #     y_err = exp_array[2,:]
        
    #     plt.scatter(x, y, label='exp')
    #     plt.errorbar(x, y, yerr=y_err)
    
    
    plt.xlabel('time(h)')
    plt.ylabel('Number')
    plt.grid()
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_switch:
        plt.savefig('overal_stat_absolute.PNG', dpi=300)
    ### absolute number
    
    
    
    ### normalized
    list_of_data.clear()
    list_of_data = [alive_stat_overal, 
                    C_alive_stat_overal, 
                    WT_alive_stat_overal]
    
    labels_of_data.clear()
    labels_of_data = ['norm_tot', 
                      'norm_C', 
                      'norm_WT']
    
    list_of_colors = ['b', 'g', 'm']
    
    plt.figure()
    
    for dataC in range(len(list_of_data)):
        data = list_of_data[dataC].copy()
        data = data / data[:,[0]]
        data_label = labels_of_data[dataC]
        
        color = list_of_colors[dataC]
        
        x = time
        y = np.mean(data,axis=0)
        y_err = np.std(data,axis=0)
        plt.plot(x, y, label=data_label, color=color)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.3, color=color)
        
        save_array = 0.0 * save_array
        save_array[0,:] = x
        save_array[1,:] = y
        save_array[2,:] = y_err
        np.savetxt("overal_pp"+"/"+data_label+"_ov_pl"+".txt", save_array, fmt='%.4f', delimiter=',')
    
    if exp_data_load_switch:
        #C_mix
        exp_array = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_C_mix')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        #WT_mix
        exp_array = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_WT_mix')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        #C_pure
        exp_array = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_C_pure')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        #WT_pure
        exp_array = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_WT_pure')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        
    plt.xlabel('time(h)')
    plt.ylabel('Normalized Number')
    plt.grid()
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_switch:
        plt.savefig('overal_stat_norm.PNG', dpi=300)
    ### normalized
    
    
    ### fractions
    list_of_data.clear()
    list_of_data = [WT_g1_tot_stat_overal, 
                    WT_g0_diff_stat_overal,
                    WT_sg2m_stat_overal]
    
    labels_of_data.clear()
    labels_of_data = ['frac_WT_g1', 
                      'frac_WT_g0_diff',
                      'frac_WT_sg2m']
    
    plt.figure()
    
    for dataC in range(len(list_of_data)):
        data = list_of_data[dataC].copy()
        data = data / WT_alive_stat_overal
        data_label = labels_of_data[dataC]
        
        x = time
        y = np.mean(data,axis=0)
        y_err = np.std(data,axis=0)
        plt.plot(x, y, label=data_label)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.3)
        
        save_array = 0.0 * save_array
        save_array[0,:] = x
        save_array[1,:] = y
        save_array[2,:] = y_err
        np.savetxt("overal_pp"+"/"+data_label+"_ov_pl"+".txt", save_array, fmt='%.4f', delimiter=',')
    
    plt.xlabel('time(h)')
    plt.ylabel('fractions')
    plt.grid()
    # plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_switch:
        plt.savefig('overal_stat_fracs.PNG', dpi=300)
    ### fractions
    
    
    
    
    
    return

def overal_saver():
    
    try:
        directory = "overal_pp"
        os.makedirs(directory, exist_ok=True)
    except:
        pass
        
    # C_alive_stat_overal[runC, :] = C_alive_stat.copy()
    # C_apop_stat_overal[runC, :] = C_apop_stat.copy()
    
    # WT_alive_stat_overal[runC, :] = WT_alive_stat.copy()
    # WT_apop_stat_overal[runC, :] = WT_apop_stat.copy()
    # WT_cyc_stat_overal[runC, :] = WT_cyc_stat.copy()
    # WT_diff_stat_overal[runC, :] = WT_diff_stat.copy()
    # WT_g0_stat_overal[runC, :] = WT_g0_stat.copy()
    # WT_g1_arr_stat_overal[runC, :] = WT_g1_arr_stat.copy()
    # WT_g1_cyc_stat_overal[runC, :] = WT_g1_cyc_stat.copy()
    # WT_g1_tot_stat_overal[runC, :] = WT_g1_tot_stat.copy()
    
    # WT_g0_diff_stat_overal[runC, :] = WT_g0_stat.copy() + WT_diff_stat.copy()
    
    
    # np.savetxt(directory+"/"+"time.txt", time, fmt='%.4f')
    # np.savetxt(directory+"/"+"alive_stat_overal.txt", alive_stat_overal, fmt='%d')
    # np.savetxt(directory+"/"+"C_alive_stat.txt", C_alive_stat, fmt='%d')
    # np.savetxt(directory+"/"+"C_apop_stat.txt", C_apop_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_alive_stat.txt", WT_alive_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_cyc_stat.txt", WT_cyc_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g1_cyc_stat.txt", WT_g1_cyc_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g1_arr_stat.txt", WT_g1_arr_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g1_tot_stat.txt", WT_g1_tot_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g0_stat.txt", WT_g0_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_diff_stat.txt", WT_diff_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_apop_stat.txt", WT_apop_stat, fmt='%d')
    
    return

def distributions_plotter():
    
    WT_enter_fit_mean = []
    WT_exit_fit_mean = []
    
    WT_enter_fit_err = []
    WT_exit_fit_err = []
    
    for time_Win_C in range(N_time_windows):
        WT_enter_fit_mean.append( np.mean(WT_enter_fit_all_runs[time_Win_C], axis=0) )
        WT_exit_fit_mean.append( np.mean(WT_exit_fit_all_runs[time_Win_C], axis=0) )
        
        WT_enter_fit_err.append( np.std(WT_enter_fit_all_runs[time_Win_C], axis=0)/np.sqrt(N_runs-1) )
        WT_exit_fit_err.append( np.std(WT_exit_fit_all_runs[time_Win_C], axis=0)/np.sqrt(N_runs-1) )
    
    for time_Win_C in range(N_time_windows):
        
        fig, ax1 = plt.subplots()

        # Calculate the bin centers from bin edges
        bin_edges = fit_hist_bins_focused
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plotting the histogram on the left y-axis
        ax1.bar(bin_centers, WT_enter_fit_mean[time_Win_C], width=np.diff(bin_edges), align='center', alpha=0.7, color='blue', label='Histogram')
        ax1.errorbar(bin_centers, WT_enter_fit_mean[time_Win_C], yerr=WT_enter_fit_err[time_Win_C], fmt='o', color='black', capsize=5, label='Error Bars')
        
        # Vertical line for threshold
        ax1.axvline(x=variables['Fit_Th_Wall'], color='gray', linestyle='--', label='Threshold')
        
        # Label for the left y-axis
        ax1.set_ylabel("PDF", color='blue')
        
        # Create a secondary y-axis for the barrier plot
        ax2 = ax1.twinx()
        
        # Generate and plot the barrier curve on the right y-axis
        x_barrier_plot = np.linspace(variables['Fit_Th_Wall'] + 0.01, bin_centers[-1], 1000)
        y_barrier_plot = variables['barrierPeakCoef'] / (x_barrier_plot - variables['Fit_Th_Wall'])
        ax2.plot(x_barrier_plot, y_barrier_plot, color='red', zorder=10, label='V(F)')
        
        # Label for the right y-axis
        ax2.set_ylabel("Barrier Values", color='red')
        
        # Label for the x-axis
        ax1.set_xlabel("Fitness")
        
        # Title of the plot
        title = "AVG WT entering fitness dist (" + str(time_Win_C * 0.25) + "* T_max)"
        plt.title(title)
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Save the plot with a specified file name
        fileName = "AVG_enter_fit_" + str(int(time_Win_C))
        plt.savefig(fileName, dpi=200)
    
        # Close the plot to avoid display
        plt.close()
        
    for time_Win_C in range(N_time_windows):
        
        # Create a new figure and axis
        fig, ax1 = plt.subplots()
        
        # Calculate the bin centers from bin edges
        bin_edges = fit_hist_bins_focused
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plotting the histogram on the left y-axis
        ax1.bar(bin_centers, WT_exit_fit_mean[time_Win_C], width=np.diff(bin_edges), align='center', alpha=0.7, color='red', label='Histogram')
        ax1.errorbar(bin_centers, WT_exit_fit_mean[time_Win_C], yerr=WT_exit_fit_err[time_Win_C], fmt='o', color='black', capsize=5, label='Error Bars')
        
        # Vertical line for threshold
        ax1.axvline(x=variables['Fit_Th_Wall'], color='gray', linestyle='--', label='Threshold')
        
        # Label for the left y-axis
        ax1.set_ylabel("PDF", color='red')
        
        # Create a secondary y-axis
        ax2 = ax1.twinx()
        
        # Generate and plot the barrier curve on the right y-axis
        x_barrier_plot = np.linspace(variables['Fit_Th_Wall'] + 0.01, bin_centers[-1], 1000)
        y_barrier_plot = variables['barrierPeakCoef'] / (x_barrier_plot - variables['Fit_Th_Wall'])
        ax2.plot(x_barrier_plot, y_barrier_plot, color='red', zorder=10, label='V(F)')
        
        # Label for the right y-axis (optional)
        ax2.set_ylabel("Barrier Values", color='red')
        
        # Label for the x-axis
        ax1.set_xlabel("Fitness")
        
        # Title of the plot
        title = "AVG WT G1-exiting fitness dist (" + str(time_Win_C * 0.25) + "* T_max)"
        plt.title(title)
        
        # Add legends
        ax1.legend(loc='upper left')
        # ax2.legend(loc='upper right')  # Uncomment if adding data to ax2 with a legend
        
        # Save the plot with a specified file name
        fileName = "AVG_exit_fit_" + str(int(time_Win_C))
        plt.savefig(fileName, dpi=200)
        
        # Close the plot to avoid display
        plt.close()
    
    return

def heatmap_dist_plotter():
    
    phi_heatmap_mean = np.mean(phi_dist_3d, axis=0)
    fit_heatmap_mean = np.mean(fit_dist_3d, axis=0)
    
    np.savetxt("overal_pp"+"/"+"phi_heatmap_mean.txt", phi_heatmap_mean, fmt='%1.6f', delimiter=',')
    np.savetxt("overal_pp"+"/"+"fit_heatmap_mean.txt", fit_heatmap_mean, fmt='%1.6f', delimiter=',')

    
    # fitness
    plt.figure()
    X, Y = np.meshgrid(time, fit_hist_bins[0:-1])
    plt.pcolormesh(X, Y, np.log(fit_heatmap_mean), shading='auto', cmap='viridis')
    plt.colorbar(label="log(Intensity)")
    plt.xlabel("time")
    plt.ylabel("fitness")
    title = 'AVG distribution of fitness (WT)'
    plt.title(title)
    file_name = 'AVG_fit_hist_WT.PNG'
    plt.savefig(file_name, dpi=150)
    plt.close()
    
    
    # phi
    plt.figure()
    X, Y = np.meshgrid(time, phi_hist_bins[0:-1])
    plt.pcolormesh(X, Y, np.log(phi_heatmap_mean), shading='auto', cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.xlabel("time")
    plt.ylabel("phi")
    title = 'AVG distribution of phi (WT)'
    plt.title(title)
    file_name = 'AVG_phi_hist_WT.PNG'
    plt.savefig(file_name, dpi=150)
    plt.close()
    
    return

def growth_factor_plotter():
    
    plt.figure()
    
    # Combine data and labels for box plot
    combined_data = [WT_growth_factors, C_growth_factors]
    sns.boxplot(data=combined_data, palette=["purple", "green"], width=0.5, showfliers=False)
    
    # Scatter data points over the box plot for WT and C
    x_positions = np.array([0, 1])
    jitter_amount = 0.08  # Spread for scattered points to avoid overlap
    
    if WT_alive_stat[0]>0:
        # Scatter WT data points in purple
        plt.scatter(np.random.normal(x_positions[0], jitter_amount, size=len(WT_growth_factors)), WT_growth_factors, 
                    color="purple", alpha=0.7, edgecolor="black", label="WT")
    
    if C_alive_stat[0]>0:
        # Scatter C data points in green
        plt.scatter(np.random.normal(x_positions[1], jitter_amount, size=len(C_growth_factors)), C_growth_factors, 
                    color="green", alpha=0.7, edgecolor="black", label="C")
    
    # Adding titles and labels
    title = "growth ratio at t = 60 h"
    plt.title(title)
    plt.xticks(ticks=[0, 1], labels=["WT", "C"])
    plt.xlabel("Type")
    plt.ylabel(r"$N/N_0$")
    plt.grid()
    plt.legend(fontsize=15)
    plt.savefig("growth_box.PNG", dpi=300)
    
    
    if WT_alive_stat[0]>0:
        
        # plt.figure()
        # plt.scatter(WT_init_percent, WT_growth_factors, color="purple")
        # plt.xlabel("starting ratio (% WT at t=0)", fontsize=12)
        # plt.ylabel(r"N/N_0", fontsize=15)
        # plt.title("WT cells in mixed organoids")
        # plt.savefig("growth_vs_perc_WT.PNG", dpi=300)
        plt.figure()
        plt.scatter(WT_init_percent, WT_growth_factors, color="purple")
        # Fit a linear regression line
        slope, intercept = np.polyfit(WT_init_percent, WT_growth_factors, 1)
        trendline = np.poly1d((slope, intercept))
        plt.plot(WT_init_percent, trendline(WT_init_percent), color="purple", linestyle="--", label="Fit Line")
        # Labels and title
        plt.xlabel("Starting ratio (% WT at t=0)", fontsize=20)
        plt.ylabel(r"$N/N_0$", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("WT cells in mixed organoids")
        # Optional: show legend for the fit line
        # plt.legend()
        # Save the figure
        plt.tight_layout()
        plt.savefig("growth_vs_perc_WT.PNG", dpi=300)
        plt.show()
    
    
    
    # plt.figure()
    # plt.scatter(1-WT_init_percent, C_growth_factors, color='green')
    # plt.xlabel("starting ratio (% C at t=0)", fontsize=12)
    # plt.ylabel(r"N/N_0", fontsize=15)
    # plt.title("C cells in mixed organoids")
    # plt.savefig("growth_vs_perc_C.PNG", dpi=300)
    
    if C_alive_stat[0]>0:
        
        plt.figure()
        plt.scatter(1 - WT_init_percent, C_growth_factors, color='green')
        # Fit a linear regression line
        slope, intercept = np.polyfit(1 - WT_init_percent, C_growth_factors, 1)
        trendline = np.poly1d((slope, intercept))
        plt.plot(1 - WT_init_percent, trendline(1 - WT_init_percent), color="g", linestyle="--", label="Fit Line")
        # Labels and title
        plt.xlabel("Starting ratio (% C at t=0)", fontsize=20)
        plt.ylabel(r"$N/N_0$", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("C cells in mixed organoids")
        # Optional: show legend for the fit line
        # plt.legend()
        # Save the figure
        plt.tight_layout()
        plt.savefig("growth_vs_perc_C.PNG", dpi=300)
        plt.show()
    
    return

def states_box_plotter():
    
    plt.figure()
    
    # Combine data and labels for box plot
    combined_data = [WT_G1_fractions, WT_SG2M_fractions, WT_G0_fractions]
    sns.boxplot(data=combined_data, palette=["purple", "yellow", "grey"], width=0.5, showfliers=False)
    
    # Scatter data points over the box plot for WT and C
    x_positions = np.array([0, 1, 2])
    jitter_amount = 0.08  # Spread for scattered points to avoid overlap
    
    # Scatter WT data points in purple
    plt.scatter(np.random.normal(x_positions[0], jitter_amount, size=len(WT_G1_fractions)), WT_G1_fractions, 
                color="purple", alpha=0.7, edgecolor="black", label="G1 percentage")
    
    # Scatter C data points in green
    plt.scatter(np.random.normal(x_positions[1], jitter_amount, size=len(WT_SG2M_fractions)), WT_SG2M_fractions, 
                color="yellow", alpha=0.7, edgecolor="black", label="S/G2/M percentage")
    
    plt.scatter(np.random.normal(x_positions[2], jitter_amount, size=len(WT_G0_fractions)), WT_G0_fractions, 
                color="grey", alpha=0.7, edgecolor="black", label="G0 percentage")
    
    # Adding titles and labels
    title = "WT phases distribution at t = 60 h"
    plt.title(title)
    plt.xticks(ticks=[0, 1, 2], labels=["G1", "S/G2/M", "G0"])
    plt.xlabel("Cell state", fontsize=15)
    plt.ylabel("percent", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("states_stats.PNG", dpi=300)
    
    return

N_runs = 30
time = np.loadtxt("run_1/pp_data/time.txt", delimiter=',', dtype=float)
N_samples = len(time)

## Reading params
filename = 'run_1/params.csv'
variables = read_custom_csv(filename)
## Reading params

alive_stat_overal = np.zeros((N_runs, N_samples), dtype=float)

C_alive_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
C_apop_stat_overal = np.zeros((N_runs, N_samples), dtype=float)

WT_alive_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_apop_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_cyc_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_diff_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g0_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g1_arr_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g1_cyc_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g1_tot_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_sg2m_stat_overal = np.zeros((N_runs, N_samples), dtype=float)

WT_g0_diff_stat_overal = np.zeros((N_runs, N_samples), dtype=float)


##### ent, ext, hist ############
fit_hist_bins = np.loadtxt("run_1/pp_data/fit_hist_bins.txt", delimiter=',')
fit_hist_bins_focused = np.loadtxt("run_1/pp_data/fit_hist_bins_focused.txt", delimiter=',')
phi_hist_bins = np.loadtxt("run_1/pp_data/phi_hist_bins.txt", delimiter=',')

N_time_windows = 5

WT_enter_fit_all_runs = [np.zeros((N_runs, len(fit_hist_bins)-1)) for _ in range(N_time_windows)]
WT_exit_fit_all_runs = [np.zeros((N_runs, len(fit_hist_bins)-1)) for _ in range(N_time_windows)]
##### ent, ext, hist ############


######## heatmap dists #######
phi_heatmap_run_1 = np.loadtxt("run_1/pp_data/WT_phi_hist_data.txt", delimiter=',')
n_rows , n_cols = np.shape(phi_heatmap_run_1)
phi_dist_3d = np.zeros((N_runs,n_rows, n_cols))

fit_heatmap_run_1 = np.loadtxt("run_1/pp_data/WT_fit_hist_data.txt", delimiter=',')
n_rows , n_cols = np.shape(fit_heatmap_run_1)
fit_dist_3d = np.zeros((N_runs,n_rows, n_cols))
######## heatmap dists #######

growth_factor_time = 60
for tC in range(len(time)):
    if np.abs(time[tC]-growth_factor_time)<(1e-6):
        growth_factor_tC = tC
WT_growth_factors = np.zeros(N_runs)
C_growth_factors = np.zeros(N_runs)
WT_init_percent = np.zeros(N_runs)

WT_G1_fractions =  np.zeros(N_runs)
WT_SG2M_fractions =  np.zeros(N_runs)
WT_G0_fractions =  np.zeros(N_runs)


for runC in range(N_runs):
    folderName = "run_"+str(runC+1)
    
    alive_stat = np.loadtxt(folderName+"/pp_data/alive_stat.txt", delimiter=',', dtype=int)
    
    C_alive_stat = np.loadtxt(folderName+"/pp_data/C_alive_stat.txt", delimiter=',', dtype=int)
    C_apop_stat = np.loadtxt(folderName+"/pp_data/C_apop_stat.txt", delimiter=',', dtype=int)
    
    WT_alive_stat = np.loadtxt(folderName+"/pp_data/WT_alive_stat.txt", delimiter=',', dtype=int)
    WT_apop_stat = np.loadtxt(folderName+"/pp_data/WT_apop_stat.txt", delimiter=',', dtype=int)
    WT_cyc_stat = np.loadtxt(folderName+"/pp_data/WT_cyc_stat.txt", delimiter=',', dtype=int)
    WT_diff_stat = np.loadtxt(folderName+"/pp_data/WT_diff_stat.txt", delimiter=',', dtype=int)
    WT_g0_stat = np.loadtxt(folderName+"/pp_data/WT_g0_stat.txt", delimiter=',', dtype=int)
    WT_g1_arr_stat = np.loadtxt(folderName+"/pp_data/WT_g1_arr_stat.txt", delimiter=',', dtype=int)
    WT_g1_cyc_stat = np.loadtxt(folderName+"/pp_data/WT_g1_cyc_stat.txt", delimiter=',', dtype=int)
    WT_g1_tot_stat = np.loadtxt(folderName+"/pp_data/WT_g1_tot_stat.txt", delimiter=',', dtype=int)
    WT_sg2m_stat = np.loadtxt(folderName+"/pp_data/WT_sg2m_stat.txt", delimiter=',', dtype=int)
    
    
    WT_growth_factors[runC] = WT_alive_stat[growth_factor_tC]/WT_alive_stat[0]
    C_growth_factors[runC] = C_alive_stat[growth_factor_tC]/C_alive_stat[0]
    WT_init_percent[runC] = WT_alive_stat[0] / alive_stat[0]
    
    WT_G1_fractions[runC] = WT_g1_tot_stat[growth_factor_tC] / WT_alive_stat[growth_factor_tC]
    WT_G0_fractions[runC] = WT_g0_stat[growth_factor_tC] / WT_alive_stat[growth_factor_tC]
    WT_SG2M_fractions[runC] = 1 - WT_G1_fractions[runC] - WT_G0_fractions[runC]
    
    alive_stat_overal[runC, :] = alive_stat.copy()
    
    C_alive_stat_overal[runC, :] = C_alive_stat.copy()
    C_apop_stat_overal[runC, :] = C_apop_stat.copy()
    
    WT_alive_stat_overal[runC, :] = WT_alive_stat.copy()
    WT_apop_stat_overal[runC, :] = WT_apop_stat.copy()
    WT_cyc_stat_overal[runC, :] = WT_cyc_stat.copy()
    WT_diff_stat_overal[runC, :] = WT_diff_stat.copy()
    WT_g0_stat_overal[runC, :] = WT_g0_stat.copy()
    WT_g1_arr_stat_overal[runC, :] = WT_g1_arr_stat.copy()
    WT_g1_cyc_stat_overal[runC, :] = WT_g1_cyc_stat.copy()
    WT_g1_tot_stat_overal[runC, :] = WT_g1_tot_stat.copy()
    WT_sg2m_stat_overal[runC, :] = WT_sg2m_stat.copy()
    
    WT_g0_diff_stat_overal[runC, :] = WT_g0_stat.copy() + WT_diff_stat.copy()
    
    
    ## distributions ##
    for time_Win_C in range(N_time_windows):
        WT_enter_fit_all_runs[time_Win_C][runC,:] = (np.loadtxt(folderName+"/pp_data/enter_fit_hist_plot_"+str(time_Win_C)+".txt", delimiter=',')).copy()
        WT_exit_fit_all_runs[time_Win_C][runC,:] = (np.loadtxt(folderName+"/pp_data/exit_fit_hist_plot_"+str(time_Win_C)+".txt", delimiter=',')).copy()
    ## distributions ##
    
    ## heatmap dists ##
    fit_dist_3d[runC, :,:] = np.loadtxt(folderName+"/pp_data/WT_fit_hist_data.txt", delimiter=',')
    phi_dist_3d[runC, :,:] = np.loadtxt(folderName+"/pp_data/WT_phi_hist_data.txt", delimiter=',')
    ## heatmap dists ##


np.savetxt("overal_pp"+"/"+"WT_growth_factors.txt", WT_growth_factors, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"C_growth_factors.txt", C_growth_factors, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"WT_init_percent.txt", WT_init_percent, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"growth_factor_time.txt", [growth_factor_time], fmt='%.4f', delimiter=',')

np.savetxt("overal_pp"+"/"+"WT_G1_fractions.txt", WT_G1_fractions, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"WT_G0_fractions.txt", WT_G0_fractions, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"WT_SG2M_fractions.txt", WT_SG2M_fractions, fmt='%.4f', delimiter=',')

exp_data_load_switch = 1
save_switch = 1

overal_plotter(exp_data_load_switch, save_switch)

growth_factor_plotter()

states_box_plotter()

distributions_plotter()
heatmap_dist_plotter()

# ## overal save
# overal_saver()
# ## overal save
