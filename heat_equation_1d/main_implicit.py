import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg

def get_params():
    a = -0.5
    b = 0.5
    L = b - a
    nx = 10000
    dt = 0.01
    nt = 5000
    alpha = 0.01
    dx = L / (nx - 1)
    r = alpha * dt / dx**2

    return{
        "L": L, "nx": nx, "dt": dt,
        "nt": nt, "alpha": alpha,
        "dx": dx, "r": r,
        "a": a, "b": b
    }

def make_grid(a, b, nx):
    x = np.linspace(a, b, nx)
    return x;

def apply_bc(u):
    u[0] = 3
    u[-1] = 10
    return u;

def init_cond(x):
    u = np.sin(np.pi * x)
    u = apply_bc(u)
    return u;

def build_matrix(nx, r):
    n = nx - 2
    ab = np.zeros((3, n))
    ab[0, 1:] = -r
    ab[2, :-1] = -r
    ab[1, :] = 1 + 2 * r
    return ab;

def build_forcing(u, r):
    b = u[1:-1].copy()
    b[0] += r * u[0]
    b[-1] += r * u[-1]
    return b;

def solver(ab, u, nt, r):
    history = [u.copy()]
    for _ in range(nt):
        b = build_forcing(u, r)
        u[1:-1] = linalg.solve_banded((1,1), ab, b)
        history.append(u.copy())
    return np.array(history);

def plot_solution(x, history, steps_to_plot):
    fig, ax = plt.subplots()
    for i in steps_to_plot:
        ax.plot(x, history[i], label=f"t={i}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid()  # Abilita la griglia
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = get_params()
    ab  = build_matrix(p["nx"], p["r"])
    x = make_grid(p["a"], p["b"], p["nx"])
    u0 = init_cond(x)
    history = solver(ab, u0, p["nt"], p["r"])
    plot_solution(x, history, steps_to_plot=[0, 100, 300, 500, 5000])

    
