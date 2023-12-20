
from sw_multilayer import *

# Simulation definition (Time and space domain)
xi = 0                      # Initial x-coordinate
xf = 1                      # Final x-coordinate
N  = 500                    # Number of points in x 
tf = 0.1                    # Final time of simulation
CFL = 0.5                   # Stability condition

# Initial conditions 
hl = 0.25                   # Left  layer height 
hr = 0.50                   # Right layer height
xdiv = 0.5                  # Point of discontinuity (if does not exist = 0 )
init_type = 'riemann'       # 'riemann' (Type of initialization)
boundary = [1, 1]           # 1 - Free Border | 2 - Wall (Type of boundary condition)
li = np.array([0.5, 0.5])   # Portion of each layer (li*hi)
ul = np.array([-0.5, 1.0])  # Left layer velocity ([u1l, u2l, ..., uLl]) for L-layers
ur = np.array([-0.5, 1.0])  # Right layer velocity ([u1r, u2r, ..., uLr]) for L-Layers
save_name = 'test_image'    # If save == True, then the output will have this name

# Setting the pipelines
multilayer = sw_multilayer(xi, xf, N, tf, CFL, li)
multilayer.initialise(hl, hr, ul, ur, xdiv, init_type, boundary)
multilayer.propagate()
multilayer.output(plot = True, save = False, save_name=save_name)
