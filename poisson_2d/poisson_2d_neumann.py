from matplotlib import pyplot as plt
import scipy as sp
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def get_params():
    Lx = 1
    Ly = 1
    nx = 500
    ny = 500
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    return{
        "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny, "dx": dx, "dy": dy
    }

def make_grid(Lx, Ly, nx, ny):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    X, Y = np.meshgrid(x, y)
    return X, Y;

def forcing(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y);

def is_boundary(i, j, nx, ny):
    return i == 0 or i == nx - 1 or j == 0 or j == ny - 1;

def boundary_conditions(i, j, x, y, dx, dy, nx, ny, k, A, b):
    # Corners
    if (i == 0 and j == 0) or (i == 0 and j == ny - 1) or (i == nx - 1 and j == 0) or (i == nx - 1 and j == ny - 1):
        A[k, k] = 1
        b[k] = 0
    # Left border: Neumann dp/dx = sin(y)
    elif i == 0:
        A[k, k] = 2 / dx**2 + 2 / dy**2
        A[k, k + 1] = -2 / dx**2
        A[k, k - nx] = -1 / dy**2
        A[k, k + nx] = -1 / dy**2
        b[k] = forcing(0, y) - 2 / dx * np.sin(y)
    # Lower border: Neumann dp/dy = x**2
    elif j == 0:
        A[k, k] = 2 / dx**2 + 2 / dy**2
        A[k, k - 1] = -1 / dx**2
        A[k, k + 1] = -1 / dx**2
        A[k, k + nx] = -2 / dy**2
        b[k] = forcing(x, 0) - 2 / dy * x**2
    # Upper border: Dirichlet p(x, Ly) = exp(x)
    elif j == ny - 1:
        A[k, k] = 1
        b[k] = np.exp(x)
    # Right border: Dirichlet p(Lx, y) = 0
    elif i == nx - 1:
        A[k, k] = 1
        b[k] = 0

    return A, b;


def build_system(nx, ny, dx, dy):
    N = nx * ny

    A = lil_matrix((N, N))
    b = np.zeros(N)

    for j in range(ny):
        for i in range(nx):
            y = j * dy
            x = i * dx
            k = i + j * nx

            if is_boundary(i, j, nx, ny):
                A, b = boundary_conditions(i, j, x, y, dx, dy, nx, ny, k, A, b)
            else:
                A[k, k] = 2 / dx**2 + 2 / dy**2
                A[k, k - 1] = -1 / dx**2
                A[k, k + 1] = -1 / dx**2
                A[k, k - nx] = -1 / dy**2
                A[k, k + nx] = -1 / dy**2

                b[k] = forcing(x, y)

    return A.tocsr(), b;


def solve_poisson(A, b, nx, ny):
    p_vec = spsolve(A, b)
    p = p_vec.reshape((ny, nx))
    return p;   

def plot_heatmap(p, Lx, Ly):
    plt.figure(figsize=(7, 5))

    plt.imshow(
        p,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        aspect="equal"
    )

    plt.colorbar(label="p(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Poisson 2D solution - heat map")

    plt.show()

if __name__ == "__main__":
    params = get_params()
    A, b = build_system(params["nx"], params["ny"], params["dx"], params["dy"])

    p = solve_poisson(A, b, params["nx"], params["ny"])

    plot_heatmap(p, params["Lx"], params["Ly"])