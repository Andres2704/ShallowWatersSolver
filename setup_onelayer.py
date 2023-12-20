
from sw_solver import *

# Simulation definition
xi = 0                      # Initial x-coordinate
xf = 1                      # Final x-coordinate
N  = 500                    # Number of points in x 
tf = 0.1                    # Final time of simulation
CFL = 0.5                   # Stability condition

# Initial conditions 
hl = 0.25                   # Left  layer height 
hr = 0.50                   # Right layer height
ul = 0.0                    # Left  layer velocity
ur = 0.0                    # Right layer velocity
xdiv = 0.5                  # Point of discontinuity
init_type = 'riemann'       # 'riemann' | 'wave' | 'bump'
boundary = [1, 1]           # 1 - Free Border | 2 - Wall | 3 - Imposed

# Bathymetry parameters
bath_type = 'none'
default = True

# Setting the pipelines 
monocouche = sw_solver(xi, xf, N, tf, CFL, 'rarefaction')
monocouche.bathymetry(bath_type, default) # Setting the bathymetry for initial conditions
monocouche.initialise(hl, hr, ul, ur, xdiv, init_type, boundary)
monocouche.propagate()
monocouche.output(plot = True, sol_exact=False, save = True)



# # ---------- Rarefaction -------------------
# hl = 0.25
# hr = 0.5
# ul = 0.1
# ur = ul + 2*((9.81*hr)**0.5 - (9.81*hl)**0.5)
# xdiv = 0.5
# # ----------- Shock -------------------------
# hl = 0.50
# hr = 0.25
# ul = 0.1
# ur = ul+(hr-hl)*np.sqrt(0.5*g*(1/hl+1/hr))
# xdiv = 0.5

# error = np.zeros((4))
# dx = np.array([1/100, 1/300, 1/600, 1/1200])
# N100 = np.loadtxt('output_sw_class_N100.out')
# N300 = np.loadtxt('output_sw_class_N300.out')
# N600 = np.loadtxt('output_sw_class_N600.out')
# N1200 = np.loadtxt('output_sw_class_N1200.out')

# error[0] = np.sqrt(((N100[1, :] - N100[2, :])**2).sum())*dx[0]
# error[1] = np.sqrt(((N300[1, :] - N300[2, :])**2).sum())*dx[1]
# error[2] = np.sqrt(((N600[1, :] - N600[2, :])**2).sum())*dx[2]
# error[3] = np.sqrt(((N1200[1, :] - N1200[2, :])**2).sum())*dx[3]

# plt.rcParams["font.family"] = "serif"
# plt.rcParams.update({'font.size': 14})
# plt.figure()
# plt.loglog(dx, error, color = 'b', marker = 'o')
# plt.ylabel('Error')
# plt.xlabel('dx')
# plt.grid(which='both')
# plt.savefig('convergence_detente_result.eps', format='eps')

# plt.figure()
# plt.plot(N100[0, :], N100[1, :], label = 'N = 100', color = 'r', marker = '.',markevery=5)
# plt.plot(N300[0, :], N300[1, :], label = 'N = 300', color = 'b', marker = 'v',markevery=14)
# plt.plot(N600[0, :], N600[1, :], label = 'N = 600', color = 'y', marker = 'X',markevery=28)
# plt.plot(N1200[0, :], N1200[1, :], label = 'N = 1200', color = 'm', marker = 'o',markevery=55)
# plt.plot(N1200[0, :], N1200[2, :], label = 'Exact Solution', color = 'k', lw = 2.0)
# plt.ylabel('h [m]')
# plt.xlabel('x [m]')
# plt.legend()
# plt.grid(which='both')
# plt.savefig('convergence_detente_result_all.eps', format='eps')
# plt.show()