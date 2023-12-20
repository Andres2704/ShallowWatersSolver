import time
import os.path
import numpy as np
from matplotlib import pyplot as plt 
from sw_exact import *

class sw_multilayer():
    def __init__(self, Xi, Xf, N, tf, CFL, li) -> None:
        self.Xi = Xi                                # Initial point in the domain
        self.Xf = Xf                                # Final point in the domain
        self.N  = N                                 # Mesh elements
        self.t  = 0                                 # Time 
        self.tf = tf                                # Final time of simulation
        self.dt = 0.0                               # Time step
        self.dx = (Xf-Xi)/N                         # Space step
        self.Ll    = 0 
        self.Lr    = 0
        self.CFL   = CFL
        self.li    = li 
        self.N_layers = len(self.li)
        self.x  = np.linspace(Xi, Xf, N+1)          # Discrete domain 
        self.lambd_max  = np.zeros(self.N+1)
            
    def initialise(self, hl, hr, ul, ur, x, type = 'riemann', boundary = 1):
        # Initial conditions
        self.bnd = boundary
        self.W          = np.zeros((self.N_layers+1, self.N+1))                # W[0] = h, W[0] = hu (conservation variables)
        self.h_star     = np.zeros((self.N_layers, self.N+1))
        self.hu_star    = np.zeros((self.N_layers, self.N+1))
        self.W_temp     = np.zeros((2, self.N+1))
        self.lambd      = np.zeros((2, self.N+1))             # Eigenvalues over the domain
        self.W_exact    = np.zeros((2, self.N+1))                # W[0] = h, W[0] = hu (conservation variables)
        self.Flux_height= np.zeros((self.N_layers, self.N+1))
        self.Flux_moment= np.zeros((self.N_layers, self.N+1))
        self.time_steps = np.zeros(self.N_layers)
        self.G          = np.zeros((self.N_layers+1, self.N+1))
        self.u_n        = np.zeros((self.N_layers+1, self.N+1))
        if type == 'riemann':
            for k in range(self.N+1):
                if self.x[k] <= x:
                    self.W[0, k] = hl
                    self.W[1:, k] = ul*hl                       
                else:
                    self.W[0, k] = hr 
                    self.W[1:, k] = ur*hr
                    #self.W[1:self.N_layers+1, k] = 0.0

    def boundary(self, Wl, Wr, type): 
        # Boudary condition
        if type == 1: # Free surface
            return np.array([Wl[0], Wl[1]])
        elif type == 2: # Symmetry
            return np.array([Wl[0], -Wl[1]])

    def numerical_flux(self, WL, WR, k):
        Ll = min([self.lambd[0, k],0])
        Lr = max([self.lambd[1, k],0])
        FWl = np.array([WL[1], WL[1]**2/WL[0] + 0.5*g*WL[0]**2])
        FWr = np.array([WR[1], WR[1]**2/WR[0] + 0.5*g*WR[0]**2])
        F = (Lr*FWl - Ll*FWr + Ll*Lr*(WR - WL))/(Lr - Ll)
        return F 

    def determine_timestep(self, W):
        for k in range(self.N+1):
            C = np.sqrt(g*W[0, k])
            u = W[1, k]/W[0, k]
            self.lambd[0, k] = u - C # lambda L
            self.lambd[1, k] = u + C # lambda R
            self.lambd_max[k] = max(abs(self.lambd[0, k]), abs(self.lambd[1, k]))
        
        self.Ll = min(self.lambd[0, :])
        self.Lr = max(self.lambd[1, :])
        dt = self.CFL*self.dx/np.max(self.lambd_max)
        if dt <= 0.0:
            print("Error in the calculation of time step, dt = ", dt)
        return dt 
    
    def propagate(self):     
        print('===== Running simulation for multilayer shallow water =====')
        print('  ---> Number of layers   : ', self.N_layers)
        print('  ---> Number of x points : ', self.N)
        print('  ---> Stability condition: ', self.CFL)
        print('  ---> Simulation time    : ', self.tf)
        start_time = time.time()
        while (self.t <= self.tf):
            # =========================================================================
            # =========== CALCULATING THE TIME STEP NEEDED AND THE FLUXES =============
            # =========================================================================
            for i in range(self.N_layers):
                self.W_temp = np.vstack((self.W[0, :], self.W[i+1, :])).copy()
                self.time_steps[i] = self.determine_timestep(self.W_temp)
                V = self.W_temp.copy()
                for k in range(self.N):
                    Vl = V[:, k]
                    Vr = V[:, k+1]
                    F = self.numerical_flux(Vl, Vr, k) 
                    self.Flux_height[i, k] = F[0]
                    self.Flux_moment[i, k] = F[1]

            self.dt = np.min(self.time_steps) 

            # =========================================================================
            # =========== PROPAGATING THE SOLUTION FOR TIME STEP N + 1 ================
            # =========================================================================
            for i in range(self.N_layers):
                # =========================================================================
                # ======= STANDARD SHALLOW WATER EQUATION SOLVER FOR (h, hu_i) ============
                # =========================================================================
                self.W_temp = np.vstack((self.W[0, :], self.W[i+1, :])).copy()
                V = self.W_temp.copy()

                for k in range(self.N):
                    Vl = V[:, k]
                    Vr = V[:, k+1]
                    F = np.array([self.Flux_height[i, k], self.Flux_moment[i, k]]).T # Ficar de olho na indexacao
                    self.W_temp[:, k]    = self.W_temp[:, k]   - self.dt*F/self.dx 
                    self.W_temp[:, k+1]  = self.W_temp[:, k+1] + self.dt*F/self.dx 

                self.W_temp[:, 0] = self.boundary(self.W_temp[:,1], self.W_temp[:,2], self.bnd[0])  # Left border 
                self.W_temp[:, self.N] = self.boundary(self.W_temp[:,self.N-1], self.W_temp[:,self.N-2], self.bnd[1]) # Right border
                # =========================================================================
                # =========================================================================
                # =========================================================================
            
                # UPDATING STAR STATE -----------------------------------------------------
                self.h_star[i, :]  = self.W_temp[0, :].copy()
                self.hu_star[i, :] = self.W_temp[1, :].copy()

            # =========================================================================
            # ============== RECONSTRUCTING SOLUTION FROM STAR STATE ==================
            # =========================================================================
            
            self.u_n = self.W[1:self.N_layers+1, :].copy()/self.W[0, :].copy()
            for k in range(self.N+1):
                # Updating height for n+1
                self.W[0, k] = 0.0
                for i in range(self.N_layers):
                    self.W[0, k] = self.W[0, k] + self.li[i]*self.h_star[i, k]

                # Calculating the mass fluxes between layers
                for i in range(self.N_layers):
                    if i == self.N_layers-1:
                        self.G[i+1, k] = 0.0
                    else:
                        self.G[i+1, k] = self.G[i, k] + self.li[i]*(self.W[0, k] - self.h_star[i, k])/self.dt
            
                # Updating momentum for n+1
                for i in range(self.N_layers): 
                    # Interfacial flux and velocity for layer i + 1
                    if i == self.N_layers-1:
                        u_ip1 = 0.0
                        G_p12_p = 0.0
                        G_p12_m = 0.0
                    else:
                        u_ip1 = self.u_n[i+1, k]
                        G_p12_p = np.max([0, self.G[i+1, k]]) 
                        G_p12_m = np.min([0, self.G[i+1, k]]) 

                    # Interfacial flux and velocity for layer i
                    u_i   = self.u_n[i,   k]

                    # Interfacial flux and velocity for layer i - 1
                    if i == 0: 
                        u_im1 = 0.0
                    else:
                        u_im1 = self.u_n[i-1,   k]
                        
                    G_m12_p = np.max([0, self.G[i, k]])
                    G_m12_m = np.min([0, self.G[i, k]]) 
                       
                    # Decentered explicit scheme 
                    self.W[i+1, k] = self.hu_star[i, k] + self.dt*(u_ip1*G_p12_p + u_i*G_p12_m - u_i*G_m12_p - u_im1*G_m12_m)/self.li[i]

            self.t  = self.t + self.dt 

        print('Simulation ended, time to solve it: ', time.time() - start_time,'s')

    def output(self, plot = True, save = False, save_name = 'output_multilayer'):
        plt.rcParams["font.family"] = "serif"
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(self.x, self.W[0,:], color = 'r', label = 'Multilayer approach', marker = 'v', markevery = 40)
        axs[0].set_ylabel(r'$h [m]$')
        axs[0].grid(which='both')
        axs[0].legend()
        markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']
        colors  = ['r', 'b', 'k', 'g', 'c', 'm', 'y', 'orange', 'brown', 'olive', 'gray']
        for i in range(self.N_layers):
            axs[1].plot(self.x, self.W[i+1,:]/self.W[0,:], color = colors[i], label = r'$u$_'+str(i), marker = markers[i], markevery = self.N/5)
            
        axs[1].set_xlabel(r'$x [m]$')
        axs[1].set_ylabel(r'$u [m/s]$')
        axs[1].grid(True)
        axs[1].legend()
        if save:
            np.savetxt(save_name + '.out', (self.x, self.W[0,:], self.W[1, :]))  
            plt.savefig(save_name + '.png', format='png')

        if plot: plt.show()
 