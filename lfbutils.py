#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def L2Norm (res, ref):
    ref_norm   = np.sqrt(np.sum(np.power(ref, 2)))
    error_norm = np.sqrt(np.sum(np.power((ref-res), 2)))
    return error_norm/ref_norm

def LinfNorm (res, ref):
    ref_norm   = np.max(np.abs(ref))
    error_norm = np.max(np.abs(ref-res))
    return error_norm/ref_norm
    
def setOverlaps(default_overlap, domain_corners, N_p_d, comm_freq, num_domains, base_overlap, overlap):
    sugested_base_vir_dy = []
    sugested_base_vir_dx = []

    for i in range(0, num_domains):
        orig_y = (domain_corners[i][1][0] - domain_corners[i][0][0])
        orig_x = (domain_corners[i][1][1] - domain_corners[i][0][1])
        sugested_base_vir_dy.append(int(np.lcm(orig_y, N_p_d[i][0]) / orig_y))
        sugested_base_vir_dx.append(int(np.lcm(orig_x, N_p_d[i][1]) / orig_x))

    base_vir_dy = np.lcm.reduce(sugested_base_vir_dy)
    base_vir_dx = np.lcm.reduce(sugested_base_vir_dx)

    vir_dy = []
    vir_dx = []
    for i in range(0, num_domains):
        orig_y = (domain_corners[i][1][0] - domain_corners[i][0][0])
        orig_x = (domain_corners[i][1][1] - domain_corners[i][0][1])
        vir_dy.append(int((orig_y*base_vir_dy) / N_p_d[i][0]))
        vir_dx.append(int((orig_x*base_vir_dx) / N_p_d[i][1]))

    max_orig_res_x = 0
    max_orig_res_y = 0
    y_sizes = []
    x_sizes = []

    # Recalculating overlps based on constrains and GCD of base and all other resolutions
    overlap_min = default_overlap
    overlap_max = (default_overlap[0]*2,default_overlap[1]*2)

    y_lcm = np.lcm.reduce(vir_dy)
    y_overlap_chunk = int(y_lcm / base_vir_dy)
    y_overlap       = y_overlap_chunk
    y_chunks             = 1

    while (y_overlap < overlap_min[0]):
        y_overlap = y_overlap + y_overlap_chunk
        y_chunks = y_chunks + 1

    X_lcm = np.lcm.reduce(vir_dx)
    x_overlap_chunk = int(X_lcm / base_vir_dx)
    x_overlap       = x_overlap_chunk
    x_chunks             = 1

    while (x_overlap < overlap_min[1]):
        x_overlap = x_overlap + x_overlap_chunk
        x_chunks = x_chunks + 1

    if ((x_overlap > overlap_max[1]) or (y_overlap > overlap_max[0])):
        raise Exception("Incompatible resolutions with given overlaps constrains")

    base_overlap.append(int(y_lcm / base_vir_dy)*y_chunks)
    base_overlap.append(int(X_lcm / base_vir_dx)*x_chunks)

    overlap_mult = 1
    # for i in range(0, num_domains):
    #     # Inflating overlaps for comm red
    #     if (comm_freq[i] != 1):
    #       print("Info: doubling the overlaps to alow communication reduction.")
    #       overlap_mult = 2

    
    for i in range(0, num_domains):
        curr_overlap_y = int(y_lcm/vir_dy[i]) * y_chunks
        curr_overlap_x = int(X_lcm/vir_dx[i]) * x_chunks

        overlap.append((curr_overlap_y*overlap_mult, curr_overlap_x*overlap_mult))

def show_load_balance(output, backend="matplotlib"):    
    if MPI.COMM_WORLD.Get_rank() == 0:
        ranks = MPI.COMM_WORLD.Get_size()
        # bs = 2**(base_pow+max_pow-1)
        # def_Ny_g = 2*bs
        # def_Nx_g = 2*bs
        # plt.imshow(std_output[0]['p_final'])
        # plt.colorbar()
        # plt.show()
        plt.figure()
        bar_data = []
        max_runtime = 0
        for i in range(len(output)):
            rank_data = output[i]["json_dict"]["RankData"]
            bar_data.append([])
            bar_data[i].append([])
            for data in rank_data:
                bar_data[i][0].append(data["VelocityStepTime"])
            bar_data[i].append([])
            for data in rank_data:
                bar_data[i][1].append(data["VelocityScatTime"])
            bar_data[i].append([])
            for data in rank_data:
                bar_data[i][2].append(data["VelocityGathTime"])
            bar_data[i].append([])
            for data in rank_data:
                bar_data[i][3].append(data["PressureStepTime"])
            bar_data[i].append([])
            for data in rank_data:
                bar_data[i][4].append(data["PressureScatTime"])
            bar_data[i].append([])
            for data in rank_data:
                bar_data[i][5].append(data["PressureGathTime"])
            

        for i in range(len(output)):                     
            left = np.zeros(ranks)
            for data in bar_data[i]:
                left += data
            if np.max(left) > max_runtime:
                max_runtime = np.max(left)
                
        if backend == "matplotlib":
            for i in range(len(output)):                     
                ax = plt.subplot(len(output), 1, i+1)
                left = np.zeros(ranks)
                width = 0.5
                for data in bar_data[i]:
                    p = ax.barh(range(ranks), data, width, left=left)
                    left += data
                plt.xlim((0,max_runtime))
        else:
            for i in range(len(output)):                     
                ax = plt.subplot(len(output), 1, i+1)
                left = np.zeros(ranks)
                unbalance = np.zeros(ranks)
                width = 0.5
                print("=======================================")
                for data in bar_data[i]:
                    # print(*(left+data))
                    print(*(data))
                    min_val = np.min(data)
                    print(*(data-min_val))
                    print("=======================================")
                    left += data
                    unbalance += data-min_val
                print()
                print(*(unbalance)) 
                print()
        

def create_domain_corners(orig_corners, decomposition, corners):
    N = (orig_corners[1][0] - orig_corners[0][0], orig_corners[1][1] - orig_corners[0][1])

    procs_y = decomposition[0]
    procs_x = decomposition[1]
    
    y_step  = int(N[0] / procs_y)
    x_step  = int(N[1] / procs_x)

    for i in range(procs_y):
        for j in range(procs_x):
            corners.append([(orig_corners[0][0] + i*y_step, orig_corners[0][1] + j*x_step), (orig_corners[0][0] + (i+1)*y_step, orig_corners[0][1] + (j+1)*x_step)])
    return corners
