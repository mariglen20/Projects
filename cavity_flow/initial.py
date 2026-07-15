import numpy as np

# Generating null scalar fields (cavity is still at the beginning)
def apply_initial_conditions(grid):
    
    u = np.zeros((grid.Ny, grid.Nx))
    v = np.zeros((grid.Ny, grid.Nx))
    p = np.zeros((grid.Ny, grid.Nx))    # neumann -> definita a meno di una costante and we need gradient (dynamic pressure not thermodinamical)
    
    return u, v, p