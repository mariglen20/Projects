import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def get_params():
    L     = 1.0
    nx    = 50
    dt    = 0.001
    nt    = 5000
    alpha = 0.01
    dx    = L / (nx - 1)
    r     = alpha * dt / dx**2
    
    print(f"r = {r:.4f} — {'stabile' if r <= 0.5 else 'INSTABILE'}")
    
    return {
        "L": L, "nx": nx, "dt": dt,
        "nt": nt, "alpha": alpha,
        "dx": dx, "r": r
    }


def make_grid(L, nx):
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    return x, dx;

def apply_bc(u):
    u[0] = 0
    u[-1] = 0
    return u;

def init_field(nx, L):
    x = np.linspace(0, L, nx)
    u = np.sin(1 * x)
    return u;

def explicit_scheme(u, r):
    u_new = u.copy();
    u_new[1:-1] = u[1:-1] + r * ( u[:-2] - 2 * u[1:-1] + u[2:] )
    return apply_bc(u_new);


def solve(u, r, nt):
    history = [u.copy()];
    for _ in range(nt):
        u = explicit_scheme(u, r)
        history.append(u.copy())
    return np.array(history);

# --- Plot ---
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

# --- Main ---
if __name__ == "__main__":
    p = get_params()
    x, dx = make_grid(p["L"], p["nx"])
    u0 = init_field(p["nx"], p["L"])
    u0 = apply_bc(u0)                                        # BC compatibili con CI
    history = solve(u0, p["r"], p["nt"])
    plot_solution(x, history, steps_to_plot=[0, 100, 300, 500, 5000])