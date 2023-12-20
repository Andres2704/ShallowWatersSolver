
from sw_twolayers import *

# Simulation definition
xi = 0
xf = 1 
N  = 500
tf = 0.1
dt = 0.01 
CFL = 0.9 

# Initial conditions 
hl = 0.25
hr = 0.50
ul = 0.0
ur = 0.0
xdiv = 0.5
init_type = 'riemann'       # 'riemann' | 'wave' | 'bump'
boundary = [1, 1]        # 1 - Free Border | 2 - Wall | 3 - Imposed

# Bathymetry parameters
bath_type = 'none'
default = True

# Setting the pipelines 
bicouche = sw_bilayer(xi, xf, N, tf, CFL, dt, boundary = 1)
bicouche.initialise(hl, hr, ul, ur, xdiv, init_type)
bicouche.propagate()
bicouche.output(sol_exact=False)
