#! /usr/bin/env python3

from pathlib import Path
import sys
import os

path_root = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(path_root))

from mpi4py import MPI
from scipy import special
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import numpy.fft as t
import matplotlib.pyplot as plt
import extractor as ext
import lfbutils as utl
import time
import json
import h5py

from mpl_toolkits.axes_grid1 import make_axes_locatable

import cProfile as profile
import pstats

use_kspace = False
equation_of_state = 'stokes'
nonlinear  = True
absorption = True
dispersion = True

np.seterr(divide='ignore', invalid='ignore')

def L2Norm (res, ref):
    ref_norm   = np.sqrt(np.sum(np.power(ref, 2)))
    error_norm = np.sqrt(np.sum(np.power((ref-res), 2)))
    return error_norm/ref_norm

def LinfNorm (res, ref):
    ref_norm   = np.max(np.abs(ref))
    error_norm = np.max(np.abs(ref-res))
    return error_norm/ref_norm


def db2neper(alpha, y=1):
    return 100 * alpha * (1e-6 / (2*np.pi))**y / (20 * np.log10(np.exp(1)));

#def absorbing_vars(sound_speed_compression, alpha_coeff_compression, alpha_power):


def nurts_lfb(grid, medium, options, source, output):

    world_comm = MPI.COMM_WORLD
    
    world_mpi_size = world_comm.Get_size()
    world_mpi_rank = world_comm.Get_rank()
    
    if world_mpi_rank == 0:
        print()
        print("Simulation Start")

    model_map           = options["model_map"]
    kelvin_voigt_model  = options["kelvin_voigt_model"]
    comm_freq           = options["comm_freq"]
    ploting             = options["ploting"]
    logging             = options["logging"]
    output_prefix       = options["output"]
    write_freqs         = options["write_freq"]
    use_single_rho      = options["use_single_rho"]
    interpolation       = options["interpolation"]

    profile = False
    if 'profile' in options.keys() and (str(options['profile']).lower == 'true' or str(options['profile']) == '1'):
        profile = True


    ds           = grid["ds"]
    dts          = grid["dt"]
    t_end        = grid["t_end"]
    N_p_d        = grid["N_per_domain"]
    d_c          = grid["domain_corners"]
    N            = grid["N"]
    base_overlap = grid["base_overlap"]

    sound_speed_compression_g = medium["sound_speed_compression"] 
    sound_speed_shear_g       = medium["sound_speed_shear"]       
    rho0_g                    = medium["rho"]                     
    p0_g                      = medium["p0"]
    alpha_coeff_compression   = medium["alpha_coeff_compression"]
    alpha_coeff_shear         = medium["alpha_coeff_shear"]
    alpha_power               = medium["alpha_power"]
    BonA                      = medium["BonA"]

    c_max = np.max(sound_speed_compression_g);
    c_ref = c_max;

    overlap       = grid["overlap"]
    
    num_domains = len(d_c)

    json_dict = {}

    json_dict["Grid"]          = N
    json_dict["Decomposition"] = grid["domain_corners"]
    json_dict["Mode"]          = options["mode"]
    json_dict["Overlap"]       = overlap

    p_final = None
    

    if world_mpi_size < num_domains:
        raise ValueError("MPI Processes should be at least equal to number of subdomains!")
        exit(-1)
    
    newGroup = world_comm.group.Excl(list(range(num_domains,world_mpi_size)))
    comm = world_comm.Create_group(newGroup)

    if world_mpi_rank < num_domains:


        mpi_size = comm.Get_size()
        mpi_rank = comm.Get_rank()
    
        write_freq = write_freqs[mpi_rank]

        N_orig = [0 , 0]

        offset_g  = d_c[mpi_rank][0]
        N_orig[0] = d_c[mpi_rank][1][0] - d_c[mpi_rank][0][0]
        N_orig[1] = d_c[mpi_rank][1][1] - d_c[mpi_rank][0][1]


        Ny_g = N[ext.AXIS_Y_2D]
        Nx_g = N[ext.AXIS_X_2D]

        x = np.arange(0, Nx_g, 1)


        dy = ds[mpi_rank][ext.AXIS_Y_2D]
        dx = ds[mpi_rank][ext.AXIS_X_2D]

        if(len(dts) != 1):
            dt = dts[mpi_rank]
        else:
            dt = dts[0]

        if(len(overlap) != 1):
            overlap_y = overlap[mpi_rank][ext.AXIS_Y_2D]
            overlap_x = overlap[mpi_rank][ext.AXIS_X_2D]
        else:
            overlap_y = overlap[0][ext.AXIS_Y_2D]
            overlap_x = overlap[0][ext.AXIS_X_2D]

        if(len(N_p_d) != 1):
            Ny = N_p_d[mpi_rank][ext.AXIS_Y_2D]
            Nx = N_p_d[mpi_rank][ext.AXIS_X_2D]
        else:
            Ny = N_p_d[0][ext.AXIS_Y_2D]
            Nx = N_p_d[0][ext.AXIS_X_2D]

        Ny_with_o = Ny + 2*overlap_y
        Nx_with_o = Nx + 2*overlap_x

        Nt = int(t_end / dt)
        rem =  t_end - (Nt * dt)
        final_dt = None
        if not np.isclose(0.0, rem):
            final_dt = rem
            Nt = Nt+1
        
        if profile:
            Nt = 5
            t_end = Nt * dt
        

        comm.Barrier()
        if mpi_rank == 0:
            print("Extractor Initiated")
        extractor = ext.Extractor((Ny_g, Nx_g), N_p_d, d_c, overlap, base_overlap, ds, dts, t_end, num_domains, model_map, mpi_rank, comm, comm_freq, interpolation)
        if mpi_rank == 0:
            print("Extractor created")

        res_coeff_y = Ny / float(N_orig[ext.AXIS_Y_2D])
        res_coeff_x = Nx / float(N_orig[ext.AXIS_X_2D])


        if use_kspace:
            kappa = extractor.get_kappa(c_ref)
        else:
            kappa = 1


        ddy_k = extractor.get_spectral_deriv(ext.AXIS_Y_2D)
        ddx_k = extractor.get_spectral_deriv(ext.AXIS_X_2D)

        shift_pos_x = extractor.get_spectral_shift( dx/2, ext.AXIS_X_2D)
        shift_neg_x = extractor.get_spectral_shift(-dx/2, ext.AXIS_X_2D)

        shift_pos_y = extractor.get_spectral_shift( dy/2, ext.AXIS_Y_2D)
        shift_neg_y = extractor.get_spectral_shift(-dy/2, ext.AXIS_Y_2D)

        bell_x = extractor.get_bell(Nx_with_o, overlap_x)
        bell_x = bell_x.reshape((1, Nx_with_o))

        bell_y = extractor.get_bell(Ny_with_o, overlap_y)
        bell_y = bell_y.reshape((Ny_with_o, 1))

        bell   = np.matmul(bell_y, bell_x)


        

        # # # Global Density
        # # #///////////////////////////////////////////////////////////////////////////////////////////////////////////
        # # rho0 = rho 
        # # #///////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Global Lame prameters
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////
        mu_g = sound_speed_shear_g**2 * rho0_g
        lambda_g = sound_speed_compression_g**2 * rho0_g - 2*mu_g
        if kelvin_voigt_model:
            eta_g = 2 * rho0_g * medium.sound_speed_shear_g ** 3      * db2neper(alpha_coeff_shear, 2)
            chi_g = 2 * rho0_g * medium.sound_speed_compression_g ** 3 * db2neper(alpha_coeff_compression, 2) - 2 * eta_g
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////

        for prefix in ['ux', 'uy', 'uz', 'sxx', 'syy', 'szz', 'p']:
            flag = prefix+'_flag'
            mask = prefix+'_mask'
            val  = prefix
            if flag in source.keys():
                source[mask], source[val] = extractor.prepare_local_sources(source[mask], source[val])
            else:
                source[flag] = 0

            if source[flag] != 0 and len(source[val]) == 0:
                source[flag] = 0


        data={}
        data["rho0"]                    = rho0_g
        data["rho0_sgx"]                = rho0_g
        data["rho0_sgy"]                = rho0_g
        data["rho0_sgx_inv"]            = rho0_g
        data["rho0_sgy_inv"]            = rho0_g
        data["mu"]                      = mu_g
        data["mu_sgxy_geom"]            = mu_g
        data["lambda"]                  = lambda_g
        data["sound_speed_compression"] = sound_speed_compression_g
        data["sound_speed_shear"]       = sound_speed_shear_g
        if kelvin_voigt_model:
            data["eta"]                 = eta_g
            data["eta_sgxy_geom"]       = eta_g
            data["chi"]                 = chi_g

        

        interpdata = extractor.prepare_medium_propperties(data)

        rho0                    = interpdata['rho0']
        rho0_sgx                = interpdata["rho0_sgx"]
        rho0_sgy                = interpdata["rho0_sgy"]
        rho0_sgx_inv            = interpdata["rho0_sgx_inv"]
        rho0_sgy_inv            = interpdata["rho0_sgy_inv"]
        mu                      = interpdata["mu"]
        mu_sgxy                 = interpdata["mu_sgxy_geom"]
        lambd                   = interpdata["lambda"]
        sound_speed_compression = interpdata["sound_speed_compression"]
        sound_speed_shear       = interpdata["sound_speed_shear"]
        if kelvin_voigt_model:
            eta                 = interpdata["eta"]
            eta_sgxy            = interpdata["eta_sgxy_geom"]
            chi                 = interpdata["chi"]


        if not np.isscalar(mu_sgxy):
            mu_sgxy [np.isnan(mu_sgxy)]  = mu [np.isnan(mu_sgxy)]
        if kelvin_voigt_model:
            if not np.isscalar(eta_sgxy):
                eta_sgxy[np.isnan(eta_sgxy)] = eta[np.isnan(eta_sgxy)]

        k = extractor.get_k(c_ref)


        #///////////////////////////////////////////////////////////////////////////////////////////////////////////
        # Absorbing stuff
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////
        if equation_of_state == 'absorbing':

            alpha_coeff_compression = db2neper(alpha_coeff_compression, alpha_power)

            # compute the absorbing fractional Laplacian operator and coefficient
            if absorption:
                absorb_nabla1 = k**(alpha_power - 2)
                absorb_nabla1[np.isinf(absorb_nabla1)] = 0
                absorb_nabla1 = t.ifftshift(absorb_nabla1)
                absorb_tau = -2 * alpha_coeff_compression * sound_speed_compression**(alpha_power - 1)
            else:
                absorb_nabla1 = 0
                absorb_tau = 0

            # compute the dispersive fractional Laplacian operator and coefficient
            if dispersion:
                absorb_nabla2 = k**(alpha_power - 1)
                absorb_nabla2[np.isinf(absorb_nabla2)] = 0
                absorb_nabla2 = t.ifftshift(absorb_nabla2)
                absorb_eta = 2 * alpha_coeff_compression * sound_speed_compression**(alpha_power) * np.tan(np.pi * alpha_power / 2)
            else:
                absorb_nabla2 = 0
                absorb_eta = 0

        if equation_of_state == 'stokes':

            # convert the absorption coefficient to nepers.(rad/s)^-2.m^-1
            alpha_coeff_compression = db2neper(alpha_coeff_compression, 2)

            # compute the absorbing coefficient
            absorb_tau = -2 * alpha_coeff_compression * sound_speed_compression
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////
        if mpi_rank == 0:
            print("Data Interpolated")

        size = (Ny_with_o, Nx_with_o)
       
        dsxxdx = np.zeros(size)
        dsyydy = np.zeros(size)
        dsxydx = np.zeros(size)
        dsxydy = np.zeros(size)

        dsxxdx_dsxydy_sum = np.zeros(size)
        dsyydy_dsxydx_sum = np.zeros(size)

        ux = np.zeros(size)
        uy = np.zeros(size)

        duxdx = np.zeros(size)
        duxdy = np.zeros(size)
        duydx = np.zeros(size)
        duydy = np.zeros(size)

        dudx = np.zeros(size)
        dudy = np.zeros(size)

        dpdx = np.zeros(size)
        dpdy = np.zeros(size)


        if kelvin_voigt_model:
            dduxdxdt = np.zeros(size)
            dduydydt = np.zeros(size)
            dduxdydt = np.zeros(size)
            dduydxdt = np.zeros(size)

        p0 = extractor.get_global_slice(p0_g)

        rhox = p0 / (2 * sound_speed_compression**2)
        rhoy = p0 / (2 * sound_speed_compression**2)

        rho = p0 / (sound_speed_compression**2)

        p = np.copy(p0)

        sxx = np.copy(-p)
        syy = np.copy(-p)
        sxy = np.zeros(size)
        
        if ploting:
            plt.ion()
            if mpi_rank == 0: 
                fig  = plt.figure(1+mpi_rank)
                ax1  = fig.add_subplot(221)
                ax2  = fig.add_subplot(222)
                ax3  = fig.add_subplot(223)
                ax4  = fig.add_subplot(224)
                img1 = ax1.imshow(p0_g)
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(img1, cax=cax, orientation='vertical')
                img2 = ax2.imshow(ux)
                img3 = ax3.imshow(uy)
                img4 = ax4.imshow(p0)

        if mpi_rank == 0:
            gatherd_timing_results=np.zeros((mpi_size,6))
        else:
            gatherd_timing_results=None

        timing_results=np.zeros((6))

        if output_prefix != "":
            f = h5py.File(f"{output_prefix}_{mpi_rank}.h5", 'w')
            time_points = int(Nt / write_freq) + 1
            p_dset  = f.create_dataset("p" , (time_points, size[0], size[1]))
            ux_dset = f.create_dataset("ux", (time_points, size[0], size[1]))
            uy_dset = f.create_dataset("uy", (time_points, size[0], size[1]))  

        if mpi_rank == 0:
            print("Datasets Alocated")

        printing = False
        if model_map[mpi_rank] == ext.FLUID_MODEL:
            ###############################################################################################################

            #                           FLUID MODEL

            ###############################################################################################################
            dpdx = np.real(t.ifftn(ddx_k * shift_pos_x * kappa * t.fftn(bell * p)))
            dpdy = np.real(t.ifftn(ddy_k * shift_pos_y * kappa * t.fftn(bell * p)))
            

            dpx = dpdx * rho0_sgx_inv
            dpy = dpdy * rho0_sgy_inv

            ux_new = dt * dpx / 2
            uy_new = dt * dpy / 2 


            dudx = np.real(t.ifftn(ddx_k * shift_neg_x * kappa * t.fftn(bell * ux)))
            dudy = np.real(t.ifftn(ddy_k * shift_neg_y * kappa * t.fftn(bell * uy)))  
            
            du = rho0 * (duxdx + duydy)
            rho_minus_1 = dt * du

            
            extractor.scatter_overlaps_density([rho, rho_minus_1, du], 0, printing)

            comm.Barrier()

            loop_start = MPI.Wtime()

            ###############################################################################################################

            #                           MAIN LOOP FLUID MODEL

            ###############################################################################################################
            for t_index in range(0,Nt):
                print("{:.2f}%\r".format((t_index/Nt)*100.0), end='')
                if final_dt != None and t_index == (Nt - 1):
                    dt = final_dt

                if profile:
                    comm.Barrier()
                ###############################################################################################################
                # VELOCITY STEP
                velocity_step_start = MPI.Wtime()
                
                ###############################################################################################################

                dpdx = np.real(t.ifftn(ddx_k * shift_pos_x * kappa * t.fftn(bell * p)))
                dpdy = np.real(t.ifftn(ddy_k * shift_pos_y * kappa * t.fftn(bell * p)))

                dpx = dpdx * rho0_sgx_inv
                dpy = dpdy * rho0_sgy_inv

                ux_new = ux - dt * dpx
                uy_new = uy - dt * dpy

                if source["ux_flag"] > t_index:
                    ux_new[source["ux_mask"]] = source["ux"][:,t_index]

                if source["uy_flag"] > t_index:
                    ux_new[source["uy_mask"]] = source["uy"][:,t_index]

                velocity_step_end   = MPI.Wtime()

                ###############################################################################################################       
                
                velocity_sct_start = MPI.Wtime()
                extractor.scatter_overlaps_velocity([ux_new, uy_new, ux, uy, dpx, dpy], t_index, printing)
                velocity_sct_end   = MPI.Wtime()
                
                ux = ux_new
                uy = uy_new

                velocity_gat_start = MPI.Wtime()
                extractor.gather_overlaps_velocity ([ux, uy], t_index+1, printing)
                if profile:
                    comm.Barrier()
                velocity_gat_end   = MPI.Wtime()

                ###############################################################################################################

                if output_prefix != "" and write_freq > 0 and t_index % write_freq == 0:
                    ux_dset[int(t_index/write_freq),:,:] = ux
                    uy_dset[int(t_index/write_freq),:,:] = uy

                
                ###############################################################################################################
                # STRESS/PRESSURE STEP

                pressure_step_1_start = MPI.Wtime()

                duxdx = np.real(t.ifftn(ddx_k * shift_neg_x * t.fftn(kappa * bell * ux)))
                duydy = np.real(t.ifftn(ddy_k * shift_neg_y * t.fftn(kappa * bell * uy)))

                
                if not nonlinear:
                    # use linearised mass conservation equation
                    du = rho0 * (duxdx + duydy)
                    rho_new = rho - dt * du
                    
                else:

                    # use nonlinear mass conservation equation (explicit calculation)
                    du = (2 * rho + rho0) * (duxdx + duydy) 
                    rho_new = rho - dt * du

                if source["p_flag"] > t_index:
                    ux_new[source["p_mask"]] = source["p"][:,t_index]
                    
                pressure_step_1_end = MPI.Wtime()
                
                pressure_sct_start = MPI.Wtime()
                extractor.scatter_overlaps_density([rho_new, rho, du], t_index+1, printing)
                pressure_sct_end   = MPI.Wtime()
                
                rho = rho_new
                
                pressure_gat_start = MPI.Wtime()
                extractor.gather_overlaps_density([rho], t_index+1, printing)
                pressure_gat_end   = MPI.Wtime()

                pressure_step_2_start = MPI.Wtime()

                if not nonlinear:
                        if equation_of_state == 'lossless':
                            # calculate p using a linear adiabatic equation of state
                            p = sound_speed_compression**2 * (rho)
                            
                        if equation_of_state == 'absorbing':
                            
                            # calculate p using a linear absorbing equation of state          
                            p = sound_speed_compression**2 * ((rho) 
                               + absorb_tau * np.real(t.ifftn( absorb_nabla1 * t.fftn(rho0 * (duxdx + duydy)) ))
                               - absorb_eta * np.real(t.ifftn( absorb_nabla2 * t.fftn(rho) )))
                           
                        if equation_of_state == 'stokes':
                            
                            # calculate p using a linear absorbing equation of state
                            # assuming alpha_power = 2
                            p = sound_speed_compression**2 * (rho + absorb_tau * rho0 * (duxdx + duydy))
                else:
                        if equation_of_state == 'lossless':
                            
                            # calculate p using a nonlinear adiabatic equation of state
                            p = sound_speed_compression**2 * (rho + BonA * (rho)**2 / (2 * rho0))
                            
                        if equation_of_state ==  'absorbing':
                            
                            # calculate p using a nonlinear absorbing equation of state
                            p = sound_speed_compression**2 * ((rho) 
                                + absorb_tau * np.real(t.ifftn( absorb_nabla1 * t.fftn(rho0 * (duxdx + duydy)) ))
                                - absorb_eta * np.real(t.ifftn( absorb_nabla2 * t.fftn(rho) ))
                                + BonA * rho**2 / (2 * rho0))
                            
                        if equation_of_state ==  'stokes':
                            
                            # calculate p using a nonlinear absorbing equation of state
                            # assuming alpha_power = 2
                            p = sound_speed_compression**2 * ((rho)
                                + absorb_tau * rho0 * (duxdx + duydy)
                                + BonA * rho**2 / (2 * rho0))

                # p_new = p - dt * du
                if profile:
                    comm.Barrier()
                pressure_step_2_end = MPI.Wtime()
                

                if output_prefix != "" and write_freq > 0 and t_index % write_freq == 0:
                    p_dset[int(t_index/write_freq),:,:] = p
                    

                timing_results[0] += velocity_step_end    - velocity_step_start
                timing_results[1] += velocity_sct_end     - velocity_sct_start
                timing_results[2] += velocity_gat_end     - velocity_gat_start
                timing_results[3] += (pressure_step_1_end - pressure_step_1_start) + (pressure_step_2_end - pressure_step_2_start)
                timing_results[4] += pressure_sct_end     - pressure_sct_start
                timing_results[5] += pressure_gat_end     - pressure_gat_start

                ###############################################################################################################
                if t_index % 30 == 0 and ploting:
                    extractor.gather_global_result(p)
                    if mpi_rank == 0:
                        print(t_index)
                        p_g = np.copy(extractor.get_global_result())
                        img1.set(data=p_g)
                        img1.set_clim(np.min(p_g), np.max(p_g))
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    comm.Barrier()
                
            loop_end = MPI.Wtime()

        else:
            ###############################################################################################################

            #                           ELASTIC MODEL

            ###############################################################################################################
            # extractor.gather_overlaps_stress([sxx, syy, sxy], 0, printing)
            dsxxdx = np.real(t.ifft(ddx_k * shift_pos_x * t.fft(bell * sxx, axis=1), axis=1))
            dsyydy = np.real(t.ifft(ddy_k * shift_pos_y * t.fft(bell * syy, axis=0), axis=0))
            dsxydx = np.real(t.ifft(ddx_k * shift_neg_x * t.fft(bell * sxy, axis=1), axis=1))
            dsxydy = np.real(t.ifft(ddy_k * shift_neg_y * t.fft(bell * sxy, axis=0), axis=0))
            
            dsxxdx_dsxydy_sum = (dsxxdx + dsxydy)
            dsyydy_dsxydx_sum = (dsyydy + dsxydx)

            ux   = -dt / rho0_sgx * dsxxdx_dsxydy_sum / 2.0
            uy   = -dt / rho0_sgy * dsyydy_dsxydx_sum / 2.0

            duxdx = np.real(t.ifft(ddx_k * shift_neg_x * t.fft(bell * ux, axis=1), axis=1))
            duxdy = np.real(t.ifft(ddy_k * shift_pos_y * t.fft(bell * ux, axis=0), axis=0))
            duydx = np.real(t.ifft(ddx_k * shift_pos_x * t.fft(bell * uy, axis=1), axis=1))
            duydy = np.real(t.ifft(ddy_k * shift_neg_y * t.fft(bell * uy, axis=0), axis=0))


            if kelvin_voigt_model:
                dduxdxdt = np.real(t.ifft(ddx_k * shift_neg_x * t.fft(bell * dsxxdx_dsxydy_sum * (1./rho0_sgx), axis=1), axis=1))
                dduydydt = np.real(t.ifft(ddy_k * shift_neg_y * t.fft(bell * dsyydy_dsxydx_sum * (1./rho0_sgy), axis=0), axis=0))

                dduxdydt = np.real(t.ifft(ddy_k * shift_pos_y * t.fft(bell * dsxxdx_dsxydy_sum * (1./rho0_sgx), axis=0), axis=0))
                dduydxdt = np.real(t.ifft(ddx_k * shift_pos_x * t.fft(bell * dsyydy_dsxydx_sum * (1./rho0_sgy), axis=1), axis=1))

                duxx = (((2 * mu + lambd) * duxdx) \
                       + ( (2 * eta + chi) * dduxdxdt) \
                       + ( lambd * duydy) \
                       + ( chi * dduydydt))

                duyy = ((lambd * duxdx) \
                       + (chi * dduxdxdt) \
                       + ((2 * mu + lambd) * duydy) \
                       + ((2 * eta + chi) * dduydydt))

                duxy = ((mu_sgxy * duydx) \
                       + (eta_sgxy * dduydxdt) \
                       + (mu_sgxy * duxdy) \
                       + (eta_sgxy * dduxdydt))


            else:
                duxx = ((2 * mu + lambd) * duxdx) + (lambd * duydy)
                
                duyy = ((2 * mu + lambd) * duydy) + (lambd * duxdx)

                duxy = (mu_sgxy * duydx) + (mu_sgxy * duxdy)

     
            
            sxx_minus_1 = -dt * duxx
            syy_minus_1 = -dt * duyy
            sxy_minus_1 = -dt * duxy
            
            extractor.scatter_overlaps_stress([sxx, syy, sxy, sxx_minus_1, syy_minus_1, sxy_minus_1, duxx, duyy, duxy], 0, printing)
            
            comm.Barrier()

            loop_start = MPI.Wtime()

            for t_index in range(0,Nt):
                print("{:.2f}%\r".format((t_index/Nt)*100.0), end='')

                if profile:
                    comm.Barrier()
                ###############################################################################################################
                # VELOCITY STEP
                velocity_step_start = MPI.Wtime()

                dsxxdx = np.real(t.ifft(ddx_k * shift_pos_x * t.fft(bell * sxx, axis=1), axis=1))
                dsyydy = np.real(t.ifft(ddy_k * shift_pos_y * t.fft(bell * syy, axis=0), axis=0))
                dsxydx = np.real(t.ifft(ddx_k * shift_neg_x * t.fft(bell * sxy, axis=1), axis=1))
                dsxydy = np.real(t.ifft(ddy_k * shift_neg_y * t.fft(bell * sxy, axis=0), axis=0))


                dsxxdx_dsxydy_sum = (dsxxdx + dsxydy) / rho0_sgx
                dsyydy_dsxydx_sum = (dsyydy + dsxydx) / rho0_sgy

                ux_new = ux + dt * dsxxdx_dsxydy_sum
                uy_new = uy + dt * dsyydy_dsxydx_sum

                if source["ux_flag"] > t_index:
                    ux_new[source["ux_mask"]] = source["ux"][:,t_index]

                if source["uy_flag"] > t_index:
                    ux_new[source["uy_mask"]] = source["uy"][:,t_index]

                velocity_step_end  = MPI.Wtime()

                ###############################################################################################################       

                velocity_sct_start = MPI.Wtime()
                extractor.scatter_overlaps_velocity([ux_new, uy_new, ux, uy, -dsxxdx_dsxydy_sum, -dsyydy_dsxydx_sum ], t_index)
                velocity_sct_end   = MPI.Wtime()

                ux = ux_new
                uy = uy_new

                velocity_gat_start = MPI.Wtime()
                extractor.gather_overlaps_velocity ([ux, uy], t_index+1)
                if profile:
                    comm.Barrier()
                velocity_gat_end   = MPI.Wtime()
                ###############################################################################################################
            
                if output_prefix != "" and write_freq > 0 and t_index % write_freq == 0:
                    ux_dset[int(t_index/write_freq),:,:] = ux
                    uy_dset[int(t_index/write_freq),:,:] = uy

                ###############################################################################################################
                # STRESS/PRESSURE STEP
                
                pressure_step_1_start = MPI.Wtime()

                duxdx = np.real(t.ifft(ddx_k * shift_neg_x * t.fft(bell * ux, axis=1), axis=1))
                duxdy = np.real(t.ifft(ddy_k * shift_pos_y * t.fft(bell * ux, axis=0), axis=0))
                duydx = np.real(t.ifft(ddx_k * shift_pos_x * t.fft(bell * uy, axis=1), axis=1))
                duydy = np.real(t.ifft(ddy_k * shift_neg_y * t.fft(bell * uy, axis=0), axis=0))


                if kelvin_voigt_model:
                    dduxdxdt = np.real(t.ifft(ddx_k * shift_neg_x * t.fft(bell * dsxxdx_dsxydy_sum * (1./rho0_sgx), axis=1), axis=1))
                    dduydydt = np.real(t.ifft(ddy_k * shift_neg_y * t.fft(bell * dsyydy_dsxydx_sum * (1./rho0_sgy), axis=0), axis=0))

                    dduxdydt = np.real(t.ifft(ddy_k * shift_pos_y * t.fft(bell * dsxxdx_dsxydy_sum * (1./rho0_sgx), axis=0), axis=0))
                    dduydxdt = np.real(t.ifft(ddx_k * shift_pos_x * t.fft(bell * dsyydy_dsxydx_sum * (1./rho0_sgy), axis=1), axis=1))

                    duxx = (((2 * mu + lambd) * duxdx) \
                       + ( (2 * eta + chi) * dduxdxdt) \
                       + ( lambd * duydy) \
                       + ( chi * dduydydt))

                    duyy = ((lambd * duxdx) \
                           + (chi * dduxdxdt) \
                           + ((2 * mu + lambd) * duydy) \
                           + ((2 * eta + chi) * dduydydt))

                    duxy = ((mu_sgxy * duydx) \
                           + (eta_sgxy * dduydxdt) \
                           + (mu_sgxy * duxdy) \
                           + (eta_sgxy * dduxdydt))


                else:
                    duxx = ((2 * mu + lambd) * duxdx) + (lambd * duydy)
                    
                    duyy = ((2 * mu + lambd) * duydy) + (lambd * duxdx)

                    duxy = (mu_sgxy * duydx) + (mu_sgxy * duxdy)

         
                
                sxx_new = sxx + dt * duxx
                syy_new = syy + dt * duyy
                sxy_new = sxy + dt * duxy


                if source["sxx_flag"] > t_index:
                    sxx_new[source["sxx_mask"]] = source["sxx"][:,t_index]

                if source["syy_flag"] > t_index:
                    sxx_new[source["syy_mask"]] = source["syy"][:,t_index]

                if source["szz_flag"] > t_index:
                    sxx_new[source["szz_mask"]] = source["szz"][:,t_index]

                pressure_step_1_end = MPI.Wtime()

                pressure_sct_start = MPI.Wtime()
                extractor.scatter_overlaps_stress([sxx_new, syy_new, sxy_new, sxx, syy, sxy, duxx, duyy, duxy], t_index+1)
                pressure_sct_end = MPI.Wtime()
             
                sxx = sxx_new
                syy = syy_new
                sxy = sxy_new

                pressure_gat_start = MPI.Wtime()
                extractor.gather_overlaps_stress ([sxx, syy, sxy], t_index+1)
                pressure_gat_end = MPI.Wtime()

                pressure_step_2_start = MPI.Wtime()
                p = -(sxx + syy)/2


                if profile:
                    comm.Barrier()

                if output_prefix != "" and write_freq > 0 and t_index % write_freq == 0:
                    p_dset[int(t_index/write_freq),:,:] = p

                pressure_step_2_end = MPI.Wtime()

                timing_results[0] += velocity_step_end    - velocity_step_start
                timing_results[1] += velocity_sct_end     - velocity_sct_start
                timing_results[2] += velocity_gat_end     - velocity_gat_start
                timing_results[3] += (pressure_step_1_end - pressure_step_1_start) + (pressure_step_2_end - pressure_step_2_start)
                timing_results[4] += pressure_sct_end     - pressure_sct_start
                timing_results[5] += pressure_gat_end     - pressure_gat_start
                ###############################################################################################################
                if t_index % ploting_freq == 0 and ploting:
                    extractor.gather_global_result(p)
                    if mpi_rank == 0:
                        print(t_index)
                        p_g = np.copy(extractor.get_global_result())
                        img1.set(data=p_g)
                        img1.set_clim(np.min(p_g), np.max(p_g))
                        fig.canvas.draw()
                        fig.canvas.flush_events()


            loop_end = MPI.Wtime()


        comm.Barrier()
        
        extractor.gather_global_result(p)
        if mpi_rank == 0:
            p_final = extractor.get_global_result()

        comm.Gather(timing_results, gatherd_timing_results, root=0)

    MPI.COMM_WORLD.Barrier()
    
    if world_mpi_rank == 0:
        all_rank_data = []
        print()
        if logging > 0:
            print(f"Rank {mpi_rank}: Loop wall clock time {(loop_end - loop_start) * 1000} ms")
        avarage_step_time = 0
        avarage_comm_time = 0
        for rank in range(0, mpi_size):
            single_rank_data = {}
            avarage_step_time += gatherd_timing_results[rank][0] + gatherd_timing_results[rank][3]
            avarage_comm_time += gatherd_timing_results[rank][1] + gatherd_timing_results[rank][2] + gatherd_timing_results[rank][4] + gatherd_timing_results[rank][5]
            if logging > 1:
                print(f"======================================================================================")
                print(f"Rank {rank}: velocity step wall clock time {gatherd_timing_results[rank][0] * 1000} ms")
            if logging > 2:
                print(f"--------------------------------------------------------------------------------------")
                print(f"Rank {rank}: velocity comm wall clock time {(gatherd_timing_results[rank][1] + gatherd_timing_results[rank][2]) * 1000} ms")
                print(f"Rank {rank}: velocity scat wall clock time {gatherd_timing_results[rank][1] * 1000} ms")
                print(f"Rank {rank}: velocity gath wall clock time {gatherd_timing_results[rank][2] * 1000} ms")
                print(f"======================================================================================")
                print(f"Rank {rank}: pressure step wall clock time {gatherd_timing_results[rank][3] * 1000} ms")
                print(f"--------------------------------------------------------------------------------------")
                print(f"Rank {rank}: pressure comm wall clock time {(gatherd_timing_results[rank][4] + gatherd_timing_results[rank][5]) * 1000} ms")
                print(f"Rank {rank}: pressure scat wall clock time {gatherd_timing_results[rank][4] * 1000} ms")
                print(f"Rank {rank}: pressure gath wall clock time {gatherd_timing_results[rank][5] * 1000} ms")
                print(f"======================================================================================")
                print(f"Rank {rank}: step combined wall clock time {(gatherd_timing_results[rank][0] + gatherd_timing_results[rank][3]) * 1000} ms")
                print(f"Rank {rank}: comm combined wall clock time {(gatherd_timing_results[rank][1] + gatherd_timing_results[rank][2] + gatherd_timing_results[rank][4] + gatherd_timing_results[rank][5]) * 1000} ms")
                print()

            single_rank_data["Rank"]             = rank
            if model_map[rank] == ext.ELASTIC_MODEL:
                single_rank_data["Model"]            = "Elastic"
            if model_map[rank] == ext.FLUID_MODEL:
                single_rank_data["Model"]            = "Fluid"
            single_rank_data["VelocityStepTime"] = gatherd_timing_results[rank][0]
            single_rank_data["VelocityScatTime"] = gatherd_timing_results[rank][1]
            single_rank_data["VelocityGathTime"] = gatherd_timing_results[rank][2]
            single_rank_data["PressureStepTime"] = gatherd_timing_results[rank][3]
            single_rank_data["PressureScatTime"] = gatherd_timing_results[rank][4]
            single_rank_data["PressureGathTime"] = gatherd_timing_results[rank][5]
            all_rank_data.append(single_rank_data)

        avarage_step_time /= mpi_size
        avarage_comm_time /= mpi_size
        if logging > 1:
            print(f"Average step wall clock time {avarage_step_time * 1000} ms")
            print(f"Average comm wall clock time {avarage_comm_time * 1000} ms")
        if logging > 0:
            print(f"======================================================================================")

        json_dict["LoopTime"] = (loop_end - loop_start) * 1000
        json_dict["StepTime"] = avarage_step_time       * 1000
        json_dict["CommTime"] = avarage_comm_time       * 1000
        json_dict["RankData"] = all_rank_data

        
        output["json_dict"] = json_dict
        output["p_final"]   = p_final

        # p[0:int(Ny/2), 0:int(Nx/2)]   = np.copy(p_d[0][overlap_y_d[0]:-overlap_y_d[0], overlap_x_d[0]:-overlap_x_d[0]])
        # p[0:int(Nx/2), int(Nx/2):Nx]  = np.copy(p_d[1][overlap_y_d[1]:-overlap_y_d[1], overlap_x_d[1]:-overlap_x_d[1]])
        # p[int(Nx/2):Ny, 0:int(Nx/2)]  = np.copy(p_d[2][overlap_y_d[2]:-overlap_y_d[2], overlap_x_d[2]:-overlap_x_d[2]])
        # p[int(Nx/2):Ny, int(Nx/2):Nx] = np.copy(p_d[3][overlap_y_d[3]:-overlap_y_d[3], overlap_x_d[3]:-overlap_x_d[3]])

        # p[:, 0:int(Nx/2)]   = np.copy(p_d[0][overlap_y_d[0]:-overlap_y_d[0], overlap_x_d[0]:-overlap_x_d[0]])
        # p[:, int(Nx/2):Nx]  = np.copy(p_d[1][overlap_y_d[1]:-overlap_y_d[1], overlap_x_d[1]:-overlap_x_d[1]])

        #p[0:Ny, 0:Nx]   = np.copy(p_d[0][overlap_y_d[0]:-overlap_y_d[0], overlap_x_d[0]:-overlap_x_d[0]])
        
        # extractor.gather_global_result(p)
        # if mpi_rank == 0:
        #   p_g = extractor.get_global_result()
        #   img1.set(data=p_g)
        #   img1.set_clim(np.min(p), np.max(p))
        #   fig.canvas.draw()
        #   fig.canvas.flush_events()
    if ploting:
        plt.ioff()
    return p_final

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--Grid",      nargs=2, default=[64, 64], type=int,   help="Grid dimension")
    parser.add_argument("-d", "--Decomp",    nargs=2, default=[1, 1],   type=int,   help="Grid decomposition")
    parser.add_argument("-i", "--Iterations",         default=700,      type=int,   help="Number of iterations")
    parser.add_argument("-s", "--Timestep",           default=0.0001,   type=float, help="Time step")
    parser.add_argument("-p", "--Ploting",            default=False,    type=bool,  help="Turn on ploting")
    parser.add_argument("-m", "--Mode",               default=0,        type=int,   help="MODE")
    parser.add_argument("-o", "--Output",             default=None,     type=str,   help="output file")

    args = parser.parse_args()

    grid    = {}
    medium  = {}
    options = {}


    # model_map = [ELASTIC_MODEL, ELASTIC_MODEL, ELASTIC_MODEL, ELASTIC_MODEL, ELASTIC_MODEL, ELASTIC_MODEL, ELASTIC_MODEL, ELASTIC_MODEL]

    # if args.Mode > 0:
    #     model_map = [FLUID_MODEL, FLUID_MODEL, FLUID_MODEL, FLUID_MODEL, FLUID_MODEL, FLUID_MODEL, FLUID_MODEL, FLUID_MODEL]
    # if args.Mode == 1:
    #     model_map[cneter_tile_y*args.Decomp[1] + cneter_tile_x] = ELASTIC_MODEL
    model_map = []
    comm_freq = []
    dt = []
    N_p_d = []
    overlap = []
    write_freq = []
    domain_corners = [[(0,0), (32,32)], [(0,32), (32,64)]]
    #domain_corners = [[(0,0), (32,32)], [(32,0), (64,32)], [(0,32), (64,64)]]
    num_domains = len(domain_corners)

    for i in range(0, num_domains):
        model_map.append(ext.FLUID_MODEL)
        comm_freq.append(1)

    Ny_g = 32 #args.Grid[0]
    Nx_g = 64 #args.Grid[1]

    cneter_tile_y = int(np.ceil(args.Decomp[0] / 2))-1
    cneter_tile_x = int(np.ceil(args.Decomp[1] / 2))-1

    tile_width_y = int(Ny_g / args.Decomp[0]) 
    tile_width_x = int(Nx_g / args.Decomp[1])

    default_overlap = (8,8)
    default_ds      = (1,1)
    default_sound_speed_compression = 10
    default_sound_speed_shear       = 0
    default_rho                     = 1000
    


    cfl = 0.05;
    default_dt  = cfl*min(default_ds) / default_sound_speed_compression

    for i in range(0, num_domains):
        dt.append(default_dt)
        N_p_d.append((domain_corners[i][1][0] - domain_corners[i][0][0], domain_corners[i][1][1] - domain_corners[i][0][1]))
        write_freq.append(30)
    
    if args.Mode > 0 :
        res_q = 0.75
        #dt     [1]   = 0.004
        #model_map[cnt_tile_idx] = ext.ELASTIC_MODEL
        loc_N = N_p_d
        N_p_d [1] = (int(loc_N[1][0]*res_q), int(loc_N[1][1]*res_q))

    # Computing ds from reampling ratio and base ds
    res_coeffs = []
    ds = []
    for i in range(0, num_domains):
         # Greather than 1 means upsampling
        res_coef_y = N_p_d[i][ext.AXIS_Y_2D] / float(domain_corners[i][1][0] - domain_corners[i][0][0])
        res_coef_x = N_p_d[i][ext.AXIS_X_2D] / float(domain_corners[i][1][1] - domain_corners[i][0][1])
        res_coeffs.append((res_coef_y, res_coef_x))
        ds.append((default_ds[ext.AXIS_Y_2D] / res_coef_y, default_ds[ext.AXIS_X_2D] / res_coef_x))


    base_overlap = []
    utl.setOverlaps(default_overlap, domain_corners, N_p_d, comm_freq, num_domains, base_overlap, overlap)


    overlap_mult = 1
    for i in range(0, num_domains):
        # Inflating overlaps for comm red
        if (comm_freq[i] != 1):
          print("Info: doubling the overlaps to alow communication reduction.")
          overlap_mult = 2

    for i in range(0, num_domains):
        overlap[i] = overlap[i]*overlap_mult

    grid["ds"]             = ds
    grid["dt"]             = dt
    grid["t_end"]          = default_dt * args.Iterations
    grid["N"]              = (Ny_g, Nx_g)
    grid["N_per_domain"]   = N_p_d
    grid["overlap"]        = overlap
    grid["base_overlap"]   = base_overlap
    grid["domain_corners"] = domain_corners
    

    p0_g        = np.zeros((Ny_g, Nx_g))
    p0_g[16,20] = 1

    Ny_win = int(Ny_g/2)
    Nx_win = int(Nx_g/2)

    blackman_y_part = np.blackman(Ny_win).reshape((Ny_win, 1))
    blackman_x_part = np.blackman(Nx_win).reshape((1, Nx_win))

    blackman_y = np.zeros((Ny_g, 1))
    blackman_x = np.zeros((1, Nx_g))

    blackman_y[int(Ny_g/2) - int(Ny_win/2) : int(Ny_g/2) + (Ny_win - int(Ny_win/2)), 0] = blackman_y_part[:, 0]
    blackman_x[0, int(Nx_g/2) - int(Nx_win/2) : int(Nx_g/2) + (Nx_win - int(Nx_win/2))] = blackman_x_part[0, :]

    win       = np.matmul(blackman_y, blackman_x)

    p0_g      = np.real(t.ifftn(t.fftn(p0_g) * t.ifftshift(win)))

    # MEDIUM PROPERTIES
    medium["sound_speed_compression"] = default_sound_speed_compression
    medium["sound_speed_shear"]       = default_sound_speed_shear
    medium["rho"]                     = default_rho
    # rho = default_rho*np.ones_like(p0_g)
    # rho[int(Ny_g/2)+5:,:] = default_rho*0.5
    # medium["rho"] = rho
    medium["p0"]                      = p0_g / np.max(p0_g)

    medium["alpha_coeff_compression"] = 0.75
    medium["alpha_coeff_shear"]       = 0.75
    medium["alpha_power"]             = 1.5
    medium["BonA"]                    = 6
    # MISCELLANEOUS OPTIONS FOR SIMULATION
    options["model_map"] = model_map
    options["mode"]      = args.Mode
    options["output"]    = args.Output

    options["kelvin_voigt_model"] = False
    options["comm_freq"]      = comm_freq
    options["ploting"]        = args.Ploting
    options["write_freq"]     = write_freq
   
    options["use_single_rho"] = True

    options["interpolation"] = 0
    options["logging"] = False
    
    output = {}
    sources = {}
    
    res  = nurts_lfb(grid, medium, options, sources, output)
  

    grid["domain_corners"] = [[(0,0), (32,64)]]
    grid["N_per_domain"]   = [(32 , 64)]
    grid["overlap"]        = [default_overlap]
    grid["dt"]             = [default_dt]
    grid["ds"]             = [default_ds]

    grid["t_end"]          = default_dt * args.Iterations
    grid["N"]              = (Ny_g, Nx_g)
    grid["base_overlap"]   = base_overlap

    options["comm_freq"]   = [1]

    grid["dt"]             = [default_dt, default_dt]
    
    output = {}
    sources = {}
    res_ref = nurts_lfb(grid, medium, options, sources, output)

    if MPI.COMM_WORLD.Get_rank() == 0:
        
        plt.ioff()
        fig = plt.figure()

        ax1  = fig.add_subplot(221)
        ax2  = fig.add_subplot(222)
        ax3  = fig.add_subplot(223)
        ax4  = fig.add_subplot(224)
        
        img1 = ax1.imshow(res)
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img1, cax=cax, orientation='vertical')

        img2 = ax2.imshow(res_ref)
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img2, cax=cax, orientation='vertical')

        img3 = ax3.imshow(res - res_ref)
        
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img3, cax=cax, orientation='vertical')

        img4 = ax4.imshow(np.abs(res - res_ref) / np.max(np.abs(res_ref)))
        
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img4, cax=cax, orientation='vertical')

        print( f"Linf rel {LinfNorm(res, res_ref)}" )
        print( f"L2   rel {L2Norm  (res, res_ref)}" )

        plt.show()

    MPI.Finalize()