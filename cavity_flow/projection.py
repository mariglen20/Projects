import numpy as np
import operators as ops

def compute_dt(u, v, grid, Re, sigma = 0.5):

    u_max = np.max(np.abs(u)) + 1e-10
    v_max = np.max(np.abs(v)) + 1e-10
    dt_conv = min(grid.dx / u_max, grid.dy / v_max)
    dt_diff = Re * min(grid.dx**2, grid.dy**2) / 4.0

    return sigma * min(dt_conv, dt_diff)

def u_intermediate(u, v, dt, Re, grid):

    u_conv, v_conv = ops.convection(u, v, grid)
    u_star = u.copy()
    v_star = v.copy()

    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
        - u_conv[1:-1, 1:-1]
        + ops.laplacian(u, grid)[1:-1, 1:-1] / Re
    )
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
        - v_conv[1:-1, 1:-1]
        + ops.laplacian(v, grid)[1:-1, 1:-1] / Re
    )

    return u_star, v_star

def project(u_star, v_star, p, dt, grid):

    # Backward gradient — consistent with forward divergence
    dpdx, dpdy = ops.gradient_backward(p, grid)
    u_new = u_star - dt * dpdx
    v_new = v_star - dt * dpdy

    return u_new, v_new