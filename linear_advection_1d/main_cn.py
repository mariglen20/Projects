import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import scipy as sp
from scipy.sparse import diags


def get_params():
    a = -1
    b = 1
    L = b - a
    nx = 1000
    dx = L / (nx - 1)

    alpha = 1                       # physical wave speed

    sigma = 0.5                     # sigma = alpha * dt / dx = number of cell traveled in one time step (increasing sigma means increasing time step)
    r = sigma / 2                   # definition of r for crank nicolson method

    T = 5                           # physical time to simulate
    dt = sigma * dx / alpha         # time step calculated through sigma 
    nt = int(T / dt)                # number of time steps calculated in the end (truncated to lowest int)
    
    return{
        "a": a, "b": b, "L": L, "nx": nx, "dx": dx,
        "alpha": alpha, "nt": nt, "dt": dt, "r": r
    }


def make_grid(a, b, nx):
    return np.linspace(a, b, nx);

def apply_bc(u, t):
    u[0] = np.sin(t)
    return u;

def init_cond(x, t):
    u = np.zeros(len(x))
    u = apply_bc(u, t)
    return u;

def get_matrix(r, nx):
    ab = np.zeros((2, nx - 1))
    ab[0, :] = 1 + r
    ab[1, :-1] = -r         # :-1 means arriving till before-last term (lower diag is shorter)
    return ab;

def get_rhs(B, u_n, u_n1, r, nx):
    rhs = B @ u_n[1:]
    bc = np.zeros(nx - 1) 
    bc[0] = r * (u_n[0] + u_n1[0])
    rhs += bc
    return rhs;


def cn_scheme(ab, rhs):
    u_new = sp.linalg.solve_banded((1,0), ab, rhs)
    return u_new;


def solver(u, nt, dt, nx, ab, B, r):
    history = [u.copy()]                        # Creating history starting from the initial condition of u

    for i in range(1, (nt + 1)):                # Iterating from first dt till nt* dt = T_actual < T 
        t = i * dt;                             # Defining time
        u_new = u.copy()
        u_new = apply_bc(u_new, t)              # Getting u^(n+1) + applying bc at time step 

        rhs = get_rhs(B, u, u_new, r, nx)       # Defining new rhs because of boundary conditions
        u_new[1:] = cn_scheme(ab, rhs)          # BC doesn't change (calculated before) -> changing next values with solve_banded

        history.append(u_new.copy())            # Appending the u^(n+1)

        u = u_new                               # The u^(n+1) become the next u^n for the next iteration

    return np.array(history);


# def plot_solution(x, history, steps_to_plot, dt):
#     fig, ax = plt.subplots()

#     for i in steps_to_plot:
#         ax.plot(x, history[i], label = f"t = {i * dt: .2f}")
    
#     ax.set_xlabel("x")
#     ax.set_ylabel("u")

#     ax.grid()
#     ax.legend()

#     plt.tight_layout()
#     plt.show()

def animate_solution(x, history, dt, n_frames=300):
    fig, ax = plt.subplots()

    frames = np.linspace(0, len(history) - 1, n_frames, dtype=int)

    line, = ax.plot(x, history[0])
    ax.set_ylim(history.min() - 0.1, history.max() + 0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(i):
        line.set_ydata(history[i])
        time_text.set_text(f"t = {i * dt:.2f}")
        return line, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=20,
        blit=True
    )

    plt.show()


if __name__ == "__main__":
    p = get_params()
    x = make_grid(p["a"], p["b"], p["nx"])
    u0 = init_cond(x, 0)

    ab = get_matrix(p["r"], p["nx"])
    B = diags([np.ones(p["nx"] - 1) * (1 - p["r"]), np.ones(p["nx"] - 2) * p["r"]], (0, -1))

    history = solver(u0, p["nt"], p["dt"], p["nx"], ab, B, p["r"])

    # plot_solution(x, history, steps_to_plot={0, 100, 200, 500}, dt = p["dt"])
    animate_solution(x, history, p["dt"], n_frames=300)