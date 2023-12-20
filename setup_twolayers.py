
from sw_twolayers import *

# Simulation definition (Time and space domain)
xi = 0                      # Initial x-coordinate
xf = 1                      # Final x-coordinate
N  = 500                    # Number of points in x 
tf = 0.1                    # Final time of simulation
CFL = 0.9                   # Stability condition

# Initial conditions 
hl = 0.25                   # Left  layer height 
hr = 0.50                   # Right layer height
ul = 0.0                    # Left  layer velocity
ur = 0.0                    # Right layer velocity
xdiv = 0.5                  # Point of discontinuity
init_type = 'riemann'       # 'riemann' 
boundary = [1, 1]           # 1 - Free Border | 2 - Wall  (Type of boundary condition)

# Setting the pipelines 
bicouche = sw_bilayer(xi, xf, N, tf, CFL, boundary)
bicouche.initialise(hl, hr, ul, ur, xdiv, init_type)
bicouche.propagate()
bicouche.output(plot=False, save=True)
