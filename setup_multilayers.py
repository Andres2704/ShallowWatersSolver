
from sw_multilayer import *

# Simulation definition
xi = 0
xf = 1 
N  = 500
tf = 0.1
dt = 0.01 
CFL = 0.5 

# Initial conditions 
hl = 0.25
hr = 0.50
ul = 0.0
ur = 0.0
xdiv = 0.5
init_type = 'riemann'       # 'riemann' | 'wave' | 'bump'
boundary = [1, 1]           # 1 - Free Border | 2 - Wall | 3 - Imposed


# Setting the options for a multy layer simulation based on previous inputs
li = np.array([0.5, 0.5])
ul = np.array([-0.5, 1.0])
ur = np.array([-0.5, 1.0])

multilayer = sw_multilayer(xi, xf, N, tf, CFL, li, 'rarefaction')
multilayer.initialise(hl, hr, ul, ur, xdiv, init_type, boundary)
multilayer.propagate()
multilayer.output(plot = True, sol_exact=False, save = False)
