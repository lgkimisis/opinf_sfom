""" 
This file was built to solve numerically 1D Burgers' equation wave equation with the FFT. The equation corresponds to :

$\dfrac{\partial u}{\partial t} + \mu u\dfrac{\partial u}{\partial x} = \nu \dfrac{\partial^2 u}{\partial x^2}$
 
where
 - u represent the signal
 - x represent the position
 - t represent the time
 - nu and mu are constants to balance the non-linear and diffusion terms.

Copyright - © SACHA BINDER - 2021

Minor edits by L. Gkimisis (2024)
"""

############## MODULES IMPORTATION ###############
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
############## SET-UP THE PROBLEM ###############


#Definition of ODE system (PDE ---(FFT)---> ODE system)
def burg_system(u,t,k,mu,nu):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j*k*u_hat
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -mu*u*u_x + nu*u_xx
    return u_t.real
#%% Solve Burgers' equation
mu = 0.5 #advection speed
nu = 0.01 #kinematic viscosity coefficient
    
#Spatial mesh
L_x = 10 #Range of the domain according to x [m]
dx = 0.02 #Infinitesimal distance
N_x = int(L_x/dx) #Points number of the spatial mesh
X = np.linspace(0,L_x,N_x) #Spatial array

#Temporal mesh
L_t = 18 #Duration of simulation [s]
dt = 0.025  #Infinitesimal time
N_t = int(L_t/dt) #Points number of the temporal mesh
T = np.linspace(0,L_t,N_t) #Temporal array

#Wavenumber discretization
k = 2*np.pi*np.fft.fftfreq(N_x, d = dx)

#Def of the initial condition    
u0 = 1*np.exp(-(X-5)**2/1.2)+0.1*np.cos(X/(L_x)*2*np.pi)+0.1*np.cos(X/(L_x)*4*np.pi) #Single space variable fonction that represent the wave form at t = 0
# viz_tools.plot_a_frame_1D(X,u0,0,L_x,0,1.2,'Initial condition')

############## EQUATION SOLVING ###############
#PDE resolution (ODE system resolution)
U = odeint(burg_system, u0, T, args=(k,mu,nu,), mxstep=5000).T

############## PLOT ###############
fig1 = plt.figure()
ax1 = plt.axes(xlim=(0, X[-1]))
plt.xlabel('x (m)',size=16)
plt.ylabel('u',size=16)
cont1=plt.plot(X,U[:,0],color='yellowgreen',marker='o')
#ax1.legend()

############## SAVE DATA ###############
np.save('./1d_burgers_soln', U)
np.save('./1d_burgers_time',T)
np.save('./1d_burgers_x',X)