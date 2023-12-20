import time
import os.path
from matplotlib import pyplot as plt 
from sw_exact import *

def xsi(x, xbar, lamb, a):
    if abs(x - xbar)>(lamb/4):
        return 0
    else: 
        return a*np.sin(2*np.pi*(x - xbar)/lamb + 0.5*np.pi)

class sw_solver():
    def __init__(self, Xi, Xf, N, tf, CFL, sol_type) -> None:
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
        self.x  = np.linspace(Xi, Xf, N+1)          # Discrete domain 
        self.lambd_max  = np.zeros(self.N+1)
        self.sol_type = sol_type
        self.exact = exact_solution(sol_type)
            
    def initialise(self, hl, hr, ul, ur, x, type = 'riemann', boundary = 1):
        # Initial conditions
        self.bnd = boundary
        self.W          = np.zeros((2, self.N+1))                # W[0] = h, W[0] = hu (conservation variables)
        self.lambd      = np.zeros((2, self.N+1))             # Eigenvalues over the domain
        self.W_exact    = np.zeros((2, self.N+1))                # W[0] = h, W[0] = hu (conservation variables)
        if type == 'riemann':
            for i in range(self.N+1):
                if self.x[i] <= x:
                    self.W[0, i] = hl + self.B[i]
                    self.W[1, i] = ul*hl 
                else:
                    self.W[0, i] = hr + self.B[i]
                    self.W[1, i] = ur*hr 
        elif type == 'wave':
            a = 0.5
            lamb = 0.75
            xbar = 0.5
            hbar = 1.0
            cbar = np.sqrt(hbar*g)
            for i in range(self.N+1):
                self.W[0, i] = hbar + xsi(self.x[i], xbar, lamb, a)
                self.W[1, i] = 0.0 #-2*cbar*(np.sqrt(1 + xsi(self.x[i], xbar, lamb, a)/hbar) - 1)*self.W[0, i]
        else:
            q0 = 0.18
            h0 = 0.33
            self.W[0, :] = h0 - self.B
            self.W[1, :] = 0.3*q0 

        # Exact solution
        if self.sol_type == 'bump':
            self.W_exact = self.exact.call_solution(self.x[0], self.tf, self.N+1)
        else:
            for k in range(self.N+1):
                self.W_exact[:, k] = self.exact.call_solution(self.x[k], self.tf, self.N+1)

    def bathymetry(self, type = 'line', default = True):
        self.B = np.zeros(self.N+1)
        self.B_prime = np.zeros(self.N+1)
        if type == 'line':
            if default:
                self.B = 0.4*self.x + 0.2
                self.B_prime = 0.4 + np.zeros((self.N+1))
            else:
                a = float(input('Coef a of ax + b: ', ))
                b = float(input('Coef b of ax + b: '))
                self.B = a*self.x + b
                self.B_prime = a + np.zeros((self.N+1))
        elif type == 'bump':
            self.kmin = 0
            self.kmax = 0
            if default:
                for i in range(self.N):
                    if (self.x[i]>8 and self.x[i]<12):
                        if (self.x[i]>8 and self.kmin == 0): self.kmin = i
                        if (self.x[i+1]>=12 and self.kmax == 0): self.kmax = i
                        self.B[i] = 0.2 - 0.05*(self.x[i] - 10)**2
                        self.B_prime[i] = -0.1*(self.x[i] - 10)
            else:
                a = float(input('Coef sigma of bell curve: '))
                self.B = np.exp(-a*(self.x)**2)/np.sqrt(2*np.pi)
                self.B_prime = -a*self.x*2*np.exp(-a*self.x**2)/np.sqrt(2*np.pi)
        elif type == 'riemann':
            self.B = self.B + 0.5
            for i in range(self.N):
                if (i <= self.N/2):
                    self.B[i] = 0.25

        else:
            self.B = np.zeros((self.N+1))
            self.B_prime = np.zeros((self.N+1))
           
    def boundary(self, Wl, Wr, type): 
        # Boudary condition
        if type == 1: # Free surface
            return np.array([Wl[0], Wl[1]])
        elif type == 2: # Symmetry
            return np.array([Wl[0], -Wl[1]])
        else:
            y = 0.5*(self.boundary_function(self.t) + self.boundary_function(self.t+self.dt))
            return np.array([y, self.Lr*(Wl[0] - Wr[0]) + Wr[1]])
            # q0 = 0.18
            # return np.array([Wl[0], q0])
        
    def boundary_function(self, t):
        y = 0
        hl = 0.50
        hr = 0.25       
        ul = 0.1
        a = -0.05
        sigma=ul+hr*np.sqrt(0.5*g*(1/hl+1/hr))

        if (t < -a/sigma):
            y = hr 
        elif (t >= -a/sigma): 
            y = hl
        return y

    def numerical_flux(self, WL, WR, k):
        Ll = min([self.lambd[0, k],0])
        Lr = max([self.lambd[1, k],0])
        FWl = np.array([WL[1], WL[1]**2/WL[0] + 0.5*g*WL[0]**2])
        FWr = np.array([WR[1], WR[1]**2/WR[0] + 0.5*g*WR[0]**2])
        F = (Lr*FWl - Ll*FWr + Ll*Lr*(WR - WL))/(Lr - Ll)
        return F 

    def determine_timestep(self):
        for k in range(self.N+1):
            C = np.sqrt(g*self.W[0, k])
            u = self.W[1, k]/self.W[0, k]
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
        print('- Running simulation for one layer SW equations')
        start_time = time.time()
        well_balanced = True   
        # Propagate the solution 
        if well_balanced:
            while (self.t < self.tf):
                self.dt = self.determine_timestep()
                V = self.W.copy()
                
                for k in range(self.N):
                    # Left cell information treatment
                    hi_l = np.max([0, V[0, k] + self.B[k] - np.max([self.B[k], self.B[k+1]])])
                    Vl = np.array([hi_l, hi_l*V[1, k]/V[0, k]]).T
                    Sl = np.array([0, 0.5*g*(V[0, k]**2 - hi_l**2)]).T
                
                    # Right cell information treatment
                    hi_r = np.max([0, V[0, k+1] + self.B[k+1] - np.max([self.B[k], self.B[k+1]])])
                    Vr = np.array([hi_r, hi_r*V[1, k+1]/V[0, k+1]]).T
                    Sr = np.array([0, 0.5*g*(V[0, k+1]**2 - hi_r**2)]).T

                    # Numerical flux
                    F = self.numerical_flux(Vl, Vr, k)

                    # Integration
                    self.W[:, k]    = self.W[:, k]   - self.dt*(F)/self.dx - self.dt*Sl
                    self.W[:, k+1]  = self.W[:, k+1] + self.dt*(F)/self.dx + self.dt*Sr
                
                # self.W[:, 0] = np.array([self.W[0, 1], 0.18]).T
                # self.W[:, self.N] = np.array([0.33, 0.33*self.W[1, self.N-1]/self.W[0, self.N-1]]).T
                self.W[:, 0] = self.boundary(self.W[:,1], self.W[:,2], self.bnd[0])  # Left border 
                self.W[:, self.N] = self.boundary(self.W[:,self.N-1], self.W[:,self.N-2], self.bnd[1]) # Right border
                self.t  = self.t + self.dt 
        else:
            while (self.t < self.tf):
                self.dt = self.determine_timestep()
                V = self.W.copy()
                for k in range(self.N):
                    Vl = V[:, k]
                    Vr = V[:, k+1]
                    F = self.numerical_flux(Vl, Vr, k) 
                    Sl = np.array([0.0, - self.W[0, k]*g*(self.B[k+1] - self.B[k])/self.dx])    
                    Sr = np.array([0.0, - self.W[0, k+1]*g*(self.B[k+1] - self.B[k])/self.dx])
                    self.W[:, k]    = self.W[:, k]   - self.dt*F/self.dx - self.dt*Sl
                    self.W[:, k+1]  = self.W[:, k+1] + self.dt*F/self.dx + self.dt*Sr
                # self.W[:, 0] = np.array([self.W[0, 1], 0.18])   
                # self.W[:, self.N] = np.array([0.33, 0.33*self.W[1, self.N-1]/self.W[0, self.N-1]])   
                self.W[:, 0] = self.boundary(self.W[:,1], self.W[:,2], self.bnd[0])  # Left border 
                self.W[:, self.N] = self.boundary(self.W[:,self.N-1], self.W[:,self.N-2], self.bnd[1]) # Right border

                self.t  = self.t + self.dt 

        print('Simulation ended, time to solve it: ', time.time() - start_time,'s')

    def output(self, plot = True, sol_exact = False, save = False):
        if save:
            np.savetxt('output_sw_class_monolayer.out', (self.x, self.W[0,:], self.W[1, :]))  
        
        if plot:
            plt.rcParams["font.family"] = "serif"
            fig, axs = plt.subplots(2, 1)

            axs[0].plot(self.x, self.W[0,:], color = 'r', label = 'SW-solver')
            if (sol_exact):
                axs[0].plot(self.x, self.W_exact[0, :], color = 'b', label = 'Exact')
            
            axs[0].plot(self.x, self.B, color = 'k', label = 'Bathymetry')
            axs[0].set_ylabel('h [m]')
            axs[0].grid(which='both')

            axs[1].plot(self.x, self.W[1,:], color = 'r', label = 'Numerique')
            if (sol_exact):
                axs[1].plot(self.x, self.W_exact[1, :], color = 'b', label = 'Exact')
            
            axs[0].legend()
            axs[1].set_xlabel('x [m]')
            axs[1].set_ylabel('u [m/s]')
            axs[1].grid(which='both')

            # plt.savefig('inutil.eps', format='eps')
            plt.show()