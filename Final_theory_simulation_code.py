#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:32:13 2025

@author: ssengupta
"""

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from odeintw import odeintw;
import cmath
#from Ej_bif_vs_det import Ej_bif_vs_del;




def semi_classics_Det(Ej,delta,phi_0,gamma,t_fin,Nsteps,loop,p,a0,theta0,save):
    
    
    
    
    if p==1:
       # z1=1.841
        #A_ii=z1/(2*phi_0)
        avg_n=np.zeros(loop);
        Ej_pl=np.zeros(loop)
        
        #print(1.841/(4*jv(0,1.841)*0.045**2))
      
      
      #  delta_pl=np.zeros(loop)
        eta_save=np.zeros(loop)
        A_save=np.zeros(loop)
        
      
        gamma_optimal_sim=np.zeros(loop)
       
        
        nm=np.zeros(loop)
        sqrt=np.zeros(loop)
        
        r1=np.zeros(loop)
        r2=np.zeros(loop)
        del_tilde=np.zeros(loop)
        
        z1=1.841
        Ej_bif = (gamma*z1)/(4*jv(0, z1)*phi_0**2)
        
        print(Ej_bif)
        
        
        #Ej_vec = np.loadtxt('Fixed_wm_plots/Semi_classics_with_det_code__p_1_phi_0.06_theta_0_1_a0_25.txt', usecols=(0,));
       # delta_vec = np.loadtxt('Fixed_wm_plots/Semi_classics_with_det_code__p_1_phi_0.06_theta_0_1_a0_25.txt', usecols=(5,));
        
        
        
        

        
        for count in range(loop):
           
            #Ej_star=(Ej_vec[count]+290)
            #delta=-delta_vec[count]
            Ej_star = (Ej)*np.exp(-(phi_0**2)/2)
            
            
            
            def coupled_equations(t,alpha):
                A,eta=alpha;
                A_dot=-gamma/2*A-(Ej_star*phi_0/2)*np.sin((eta-np.pi/2))*(jv(0,2*A*phi_0)+jv(2,2*A*phi_0));
                eta_dot=-delta-(Ej_star*phi_0/(2*A))*np.cos((eta-np.pi/2))*(jv(0,2*A*phi_0)-jv(2,2*A*phi_0));
                alpha_dot=[A_dot,eta_dot]
                return alpha_dot ;
            
            t_span = (0, t_fin)  # Solve from t=0 to t=10
            t_eval = np.linspace(0, t_fin, Nsteps)  # Time points to evaluate
            alpha_0 = [a0,theta0];
            
            sol = solve_ivp(coupled_equations, t_span, alpha_0,t_eval=t_eval,
                    rtol=1e-12, atol=1e-8,
                    method='BDF')
            
            
        #    t = sol.t
            A = sol.y[0]
            eta =sol.y[1]
            
            eta_save[count]=eta[-1]
            A_save[count]=A[-1]
            
            # plt.figure()
            # plt.plot(t,eta)
            # plt.show()
            
            # plt.figure()
            # plt.plot(t,A)
            # plt.show()
            
            if delta==0:
                z1=1.841
                A_ii = z1/(2*phi_0)
                Ej_bif = (gamma*z1)/(4*jv(0, z1)*phi_0**2)
                if Ej_star >= Ej_bif:
                    result_eta = theta0 * \
                        (np.arccos((A_ii*gamma)/(2*(Ej_star)*phi_0*jv(2, z1))))
                    avg_n[count] = abs(A_ii)**2
                    eta_save[count]=result_eta
                    A_save[count]=A_ii
                    A[-1]=A_ii
                    eta[-1]=result_eta
                else: 
                    eta_save[count] = 0
                    eta[-1]=0
            
            
            avg_n[count]=abs(A[-1])**2
            
            
            nu=Ej_star*phi_0**2*jv(1,2*phi_0*A[-1])*np.cos(eta[-1]-np.pi/2);
            #r=-1j*(Ej_star*phi_0**2/2)*(jv(-1,2*phi_0*A[-1])*np.exp(1j*(-1)*(eta[-1]-np.pi/2))+jv(3,2*phi_0*A[-1])*np.exp(-1j*3*(eta[-1]-np.pi/2)));
            r=-0.5*(Ej_star*phi_0**2)*(jv(1,2*phi_0*A[-1])*np.exp(-1j*eta[-1])+jv(3,2*phi_0*A[-1])*np.exp(-1j*3*eta[-1]))
            
            S_20=(0.5*(abs(r)**2))/((-delta+nu)**2-abs(r)**2+(gamma**2/4));
            S_30=0.5*(2*(-delta+nu)**2+gamma**2/2-abs(r)**2)/((-delta+nu)**2-abs(r)**2+(gamma**2/4));
            S_10=(r*(S_20+S_30))/(gamma+2*1j*(-delta+nu));
            S_40=(np.conj(r)*(S_20+S_30))/(gamma-2*1j*(-delta+nu));
            
           
            def S1_S2(S12,t):
                S1,S2=S12;
                S1_dot=(1j*(delta-nu)-gamma/2)*S1 +r*S2;
                S2_dot=np.conj(r)*S1+(-1j*(delta-nu)-gamma/2)*S2
                S12_dot=[S1_dot,S2_dot];
                return S12_dot
            
            def S3_S4(S34,t):
                S3,S4=S34;
                S3_dot=(-1j*(-delta+nu)-gamma/2)*S3+r*S4;
                S4_dot=np.conj(r)*S3+(1j*(-delta+nu)-gamma/2)*S4
                S34_dot=[S3_dot,S4_dot];
                return S34_dot

      #   
           
      #       # Time span for the solution
            
            t_eval_Snn = np.linspace(0, t_fin, Nsteps)   # Points to evaluate the solution
            S12_0=[S_10,S_20];
            S34_0=[S_30,S_40]
           
            dt=t_fin/Nsteps;
            sqrt[count]=abs(r)**2-(nu-delta)**2
           
      #       # Solve the system of ODEs
            solution_1 = odeintw(S1_S2, S12_0, t_eval_Snn)
            solution_2 = odeintw(S3_S4, S34_0, t_eval_Snn)
            
            S1=solution_1[:,0];
            S2=solution_1[:,1];
            S3=solution_2[:,0];
            S4=solution_2[:,1];
            
            
            Snn_t=avg_n[count]*(S3+S2+np.exp(-2*1j*eta[-1])*(S4)+np.exp(2*1j*eta[-1])*(S1))
            
            
            N_N_corr=np.fft.fft(Snn_t);
            reversed_fft =N_N_corr[::-1]
            S_nn_real= np.fft.fftshift(reversed_fft.real)*2*dt; 
            S_nn_real=(S_nn_real)-(S_nn_real[-1])
            freq_axis = np.linspace(-np.pi/dt, np.pi/dt,Nsteps);
            
            gamma_opt_rl=np.zeros(int(Nsteps/2));
            Snn_omega_minus= S_nn_real[::-1]
            gamma_opt_rl=S_nn_real[int(Nsteps/2):Nsteps]-Snn_omega_minus[int(Nsteps/2):Nsteps]
            
            #nmr_sim=np.zeros(int(Nsteps/2));
            #nmr_sim=Snn_omega_minus[int(Nsteps/2):Nsteps]/(S_nn_real[int(Nsteps/2):Nsteps]-Snn_omega_minus[int(Nsteps/2):Nsteps])
            
          
            
            
            del_tilde[count]=delta-nu
            omega_o_g_opt=(1/np.sqrt(3))*np.sqrt((del_tilde[count]**2-abs(r)**2)-gamma**2/4+2*np.sqrt(gamma**4/16+gamma**2/4*(del_tilde[count]**2-abs(r)**2)+(del_tilde[count]**2-abs(r)**2)**2))
          
            
          #  print((del_tilde**2-abs(r)**2))
            
            
            re2itheta=np.exp(2*1j*eta[-1])*r
            r1[count]=-Ej_star*phi_0**2*0.5*np.cos(eta[-1])*(jv(1,2*phi_0*A[-1])+jv(3,2*phi_0*A[-1]))
            r2[count]=-Ej_star*phi_0**2*0.5*np.sin(eta[-1])*(jv(1,2*phi_0*A[-1])-jv(3,2*phi_0*A[-1]))
            Snn_omega_th=avg_n[count]*gamma*(((-delta+nu)+freq_axis+re2itheta.imag)**2+(gamma/2+re2itheta.real)**2)/(((nu-delta)**2-freq_axis**2+gamma**2/4-abs(r)**2)**2+gamma**2*freq_axis**2)
            gamma_omega_th=4*(avg_n[count]*gamma*(re2itheta.imag+(nu-delta))*freq_axis)/(((nu-delta)**2-freq_axis**2+gamma**2/4-abs(r)**2)**2+gamma**2*freq_axis**2)    
            omega_o_g_opt_test=r2[count]-del_tilde[count]
           
            nmr_th=((2*re2itheta.real+gamma)**2+4*(-re2itheta.imag+del_tilde[count]+omega_o_g_opt)**2)/(16*(re2itheta.imag-del_tilde[count])*omega_o_g_opt)
            gamma_omega_optimal=4*(avg_n[count]*gamma*(re2itheta.imag+(nu-delta))*omega_o_g_opt)/(((nu-delta)**2-omega_o_g_opt**2+gamma**2/4-abs(r)**2)**2+gamma**2*omega_o_g_opt**2)  
            gamma_omega_optimal_test=4*(avg_n[count]*gamma*(re2itheta.imag+(nu-delta))*omega_o_g_opt_test)/(((nu-delta)**2-omega_o_g_opt_test**2+gamma**2/4-abs(r)**2)**2+gamma**2*omega_o_g_opt_test**2)  
           # print(nmr_th,gamma_omega_optimal)
            nmr_th_test=((2*re2itheta.real+gamma)**2+4*(-re2itheta.imag+del_tilde[count]+omega_o_g_opt_test)**2)/(16*(re2itheta.imag-del_tilde[count])*omega_o_g_opt_test)
          
        
            
          #  print(avg_n,omega_o_g_opt,sqrt)
            
         
          
            if eta[-1]>0:
                gamma_optimal_sim[count]=max(gamma_opt_rl)
            else:
                gamma_optimal_sim[count]=min(gamma_opt_rl)
                
            gm=0.302/(3*10**6);
            g0_gamma=0.0007
            nm[count]=(gamma_omega_optimal*(g0_gamma**2)*nmr_th+(2778*gm))/(gamma_omega_optimal*(g0_gamma**2)+gm)
            nm_test=(gamma_omega_optimal_test*(g0_gamma**2)*nmr_th_test+(2778*gm))/(gamma_omega_optimal*(g0_gamma**2)+gm)
            Ej_pl[count]=Ej_star
          #  Ej=Ej+5
            
            
                
        print(gamma_omega_optimal,gamma_omega_optimal_test,nmr_th,nmr_th_test,nm,nm_test)
        plt.figure()
        plt.plot(freq_axis,Snn_omega_th,freq_axis,S_nn_real,'--')
        axes=plt.gca();
        axes.set_xlim(-5,5)
        # # axes.set_ylim(0,20)
        # #  plt.axvline(x=omega_t);
        # #  plt.axvline(x=peak_rl);
        plt.show()
            
          
            
        plt.figure()
        plt.plot(freq_axis[int(Nsteps/2):Nsteps],gamma_opt_rl,freq_axis[int(Nsteps/2):Nsteps],gamma_omega_th[int(Nsteps/2):Nsteps],'r--')
        axes = plt.gca()
        axes.set_xlim(0,20)
        plt.axvline(x=omega_o_g_opt,ls='--');
        # plt.axvline(x=peak_rl[0]);
        plt.show() 
        
       # plt.figure()
        #plt.plot(Ej_pl,r1,'.',Ej_pl,r2,'+') 
        #plt.axhline(y=-0.5,ls='--');
        #plt.show()
            
        if save==1:
               # gopt=open('Snn/Semi_classics_Snn_phi_'+str(phi_0)+'_Ej_'+str(Ej)+'_del_'+str(delta)+'_theta_0_'+str(theta0)+'_a0_'+str(a0)+'_save.txt','w')
               Ej_var=open('Semiclassics_g_op_vs_Ej/r1r2_phi_'+str(phi_0)+'_del_'+str(delta)+'_theta_0_'+str(theta0)+'_a0_'+str(a0)+'_save.txt','w')
               for ps in range(loop):
                   Ej_var.write('%11f\t'%Ej_pl[ps]+'%11f\t'%r1[ps]+'%11f\t'%r2[ps]+'%11f\n'%del_tilde[ps])
                #    gopt.write('%11f\t'%freq_axis[ps]+'%11f\n'%Snn_omega_th[ps])
            
         