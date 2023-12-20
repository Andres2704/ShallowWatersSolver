import time
import os.path
from matplotlib import pyplot as plt 
from sw_exact import *

class sw_bilayer():
    
    def __init__(self, Xi, Xf, N, tf, CFL, boundary) -> None:
        self.Xi = Xi                                # Initial point in the domain
        self.Xf = Xf                                # Final point in the domain
        self.N  = N                                 # Mesh elements
        self.t  = 0                                 # Time 
        self.tf = tf                                # Final time of simulation
        self.dt = 0.01                                # Time step
        self.dx = (Xf-Xi)/N                         # Space step
        self.Ll    = 0 
        self.Lr    = 0
        self.CFL   = CFL
        self.bnd = boundary
        self.x  = np.linspace(Xi, Xf, N+1)          # Discrete domain 
        self.lambd_max  = np.zeros(self.N+1)
            
    def initialise(self, hl, hr, ul, ur, x, type = 'riemann', layers = 1):
        # Initial conditions
        self.W          = np.zeros((3, self.N+1))                # W[0] = h, W[0] = hu (conservation variables)
        self.lambd      = np.zeros((3, self.N+1))             # Eigenvalues over the domain
        if type == 'riemann':
            for i in range(self.N+1):
                u1_l = -0.5
                u2_l = 1.0
                u1_r = -0.5
                u2_r = 1.0
                ubar_l = 0.5*(u1_l + u2_l)
                uhat_l = 0.5*(u2_l - u1_l)
                ubar_r = 0.5*(u1_r + u2_r)
                uhat_r = 0.5*(u2_r - u1_r)
                if self.x[i] <= x:
                    self.W[0, i] = hl
                    self.W[1, i] = ubar_l*hl 
                    self.W[2, i] = uhat_l 
                else:
                    self.W[0, i] = hr
                    self.W[1, i] = ubar_r*hr 
                    self.W[2, i] = uhat_r 
      
    def boundary(self, W, boundary): 
        # Boudary condition
        if boundary == 1: # Free surface
            return np.array([W[0], W[1], W[2]])
        elif boundary == 2: # Wall
            return np.array([W[0], -W[1], -W[2]])

    def numerical_flux(self, WL, WR, k):
        Ll = min([self.lambd[1, k],0])
        Lr = max([self.lambd[2, k],0])
        FWl = np.array([WL[1], WL[0]*(WL[1]**2/WL[0]**2 + WL[2]**2) + 0.5*g*WL[0]**2, WL[1]*WL[2]/WL[0]])
        FWr = np.array([WR[1], WR[0]*(WR[1]**2/WR[0]**2 + WR[2]**2) + 0.5*g*WR[0]**2, WR[1]*WR[2]/WR[0]])
        F = (Lr*FWl - Ll*FWr + Ll*Lr*(WR - WL))/(Lr - Ll)
        return F 

    def determine_timestep(self):
        for k in range(self.N+1):
            C = np.sqrt(3*self.W[2, k]**2 + g*self.W[0, k])
            u = self.W[1, k]/self.W[0, k]
            self.lambd[0, k] = u
            self.lambd[1, k] = u - C # lambda L
            self.lambd[2, k] = u + C # lambda R
            self.lambd_max[k] = max(abs(self.lambd[0, k]), abs(self.lambd[1, k]), abs(self.lambd[2, k]))
        
        self.Ll = min(self.lambd[0, :])
        self.Lr = max(self.lambd[1, :])
        dt = self.CFL*self.dx/np.max(self.lambd_max)
        if dt <= 0.0:
            print("Error in the calculation of time step, dt = ", dt)
        return dt 
    
    def propagate(self):     
        print('===== Running simulation for two layers shallow water =====')
        print('  ---> Number of layers   : ', 2)
        print('  ---> Number of x points : ', self.N)
        print('  ---> Stability condition: ', self.CFL)
        print('  ---> Simulation time    : ', self.tf)
        start_time = time.time()

        while (self.t < self.tf):
            self.dt = self.determine_timestep()
            V = self.W.copy()
            for k in range(self.N):
                Vl = V[:, k]
                Vr = V[:, k+1]
                F = self.numerical_flux(Vl, Vr, k)
                self.W[:, k]    = self.W[:, k]   - self.dt*F/self.dx 
                self.W[:, k+1]  = self.W[:, k+1] + self.dt*F/self.dx

            self.W[:, 0] = self.boundary(self.W[:,1], self.bnd[0])  # Left border 
            self.W[:, self.N] = self.boundary(self.W[:,self.N-1], self.bnd[1]) # Right border
            
            self.t  = self.t + self.dt  

        print('Simulation ended, time to solve it: ', time.time() - start_time,'s')
   
    def output(self, plot = True, save = True, save_name = 'output_twolayer'):
        u2 = self.W[1,:]/self.W[0,:] + self.W[2,:]
        u1 = u2 - 2*self.W[2,:]
        plt.rcParams["font.family"] = "serif"
        fig, axs = plt.subplots(2, 1) 
        axs[0].plot(self.x, self.W[0,:], color = 'r', label = 'Bicouche', marker = 'x', markevery = 50)
        axs[0].set_ylabel('h [m]')
        axs[0].grid(True)

        axs[1].plot(self.x, u1, color = 'r', label = r'$u_1$', marker = 'x', markevery = 50)
        axs[1].plot(self.x, u2, color = 'k', label = r'$u_2$', marker = 'v', markevery = 40)
        axs[1].set_xlabel('x [m]')
        axs[1].set_ylabel('u [m/s]')
        axs[1].grid(True)
        axs[1].legend()
        if save:
            np.savetxt(save_name + '.out', (self.x, self.W[0,:], self.W[1, :]))  
            plt.savefig(save_name + '.png', format='png')
        if plot: plt.show()

