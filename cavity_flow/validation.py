from matplotlib import pyplot as plt
import numpy as np

# Ghia et al. (1982), Re = 100, Table I   (mediane verticali e orizzontali)
# col = 0 -> position, col = 1 -> value
GHIA_U_RE100 = np.array([
    [0.0000,  0.00000],
    [0.0547, -0.03717],
    [0.0625, -0.04192],
    [0.0703, -0.04775],
    [0.1016, -0.06434],
    [0.1719, -0.10150],
    [0.2813, -0.15662],
    [0.4531, -0.21090],
    [0.5000, -0.20581],
    [0.6172, -0.13641],
    [0.7344,  0.00332],
    [0.8516,  0.23151],
    [0.9531,  0.68717],
    [0.9609,  0.73722],
    [0.9688,  0.78871],
    [0.9766,  0.84123],
    [1.0000,  1.00000],
])

GHIA_V_RE100 = np.array([
    [0.0000,  0.00000],
    [0.0625,  0.09233],
    [0.0703,  0.10091],
    [0.0781,  0.10890],
    [0.0938,  0.12317],
    [0.1563,  0.16077],
    [0.2188,  0.17507],
    [0.2344,  0.17527],
    [0.5000,  0.05454],
    [0.8047, -0.24533],
    [0.8594, -0.22445],
    [0.9063, -0.16914],
    [0.9453, -0.10313],
    [0.9531, -0.08864],
    [0.9609, -0.07391],
    [0.9688, -0.05906],
    [1.0000,  0.00000],
])


def get_profiles(u, v, grid):
    """""Getting the coordinates and relatives velcity profiles"""
    i_mid = grid.Nx // 2
    j_mid = grid.Ny // 2

    return grid.y, u[:, i_mid], grid.x, v[j_mid, :]


def compute_errors(y, u_profile, x, v_profile):
    """""Compute the interpolation and the errors"""
    u_interp = np.interp(GHIA_U_RE100[:, 0], y, u_profile)
    v_interp = np.interp(GHIA_V_RE100[:, 0], x, v_profile)

    err_u = GHIA_U_RE100[:, 1] - u_interp
    err_v = GHIA_V_RE100[:, 1] - v_interp

    return{
        "u_l2": np.mean(np.sqrt(err_u**2)),
        "u_linf": np.max(np.abs(err_u)),
        "v_l2": np.mean(np.sqrt(err_v**2)),
        "v_linf": np.max(np.abs(err_v))
    }


def plot_comparison(u, v, grid, ghia_u=GHIA_U_RE100, ghia_v=GHIA_V_RE100, Re=100):
    """Plotting the comparison witH Ghia's results"""
    y, u_profile, x, v_profile = get_profiles(u, v, grid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(u_profile, y, 'b-', label='Solver')
    axes[0].plot(ghia_u[:,1], ghia_u[:,0], 'ro', markersize=4, label='Ghia et al.')
    axes[0].set_xlabel('u'); axes[0].set_ylabel('y')
    axes[0].set_title(f'u(0.5,y), Re={Re}')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(x, v_profile, 'b-', label='Solver')
    axes[1].plot(ghia_v[:,0], ghia_v[:,1], 'ro', markersize=4, label='Ghia et al.')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('v')
    axes[1].set_title(f'v(x,0.5), Re={Re}')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()