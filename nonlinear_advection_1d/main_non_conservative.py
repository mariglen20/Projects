import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def get_space_params():
    L = 1.0
    nx = 1000
    dx = L / (nx - 1)

    return {
        "L": L,
        "nx": nx,
        "dx": dx
    }

# CFL = sigma depends on u_max that is obtained from CI so get_params needs to be called
# after initializing u(x, 0) IC
def get_time_params(u_max, dx):
    sigma = 0.5
    T = 10.0

    dt = sigma * dx / u_max
    nt = int(np.ceil(T / dt))
    dt = T / nt

    return {
        "sigma": sigma,
        "T": T,
        "dt": dt,
        "nt": nt
    }


def get_grid(L, nx):
    return np.linspace(0, L, nx)

def apply_bc(u, t):
    u[0] = np.abs(np.sin(t))
    return u;

def init_cond(x, t):
    u = np.zeros(len(x))
    u = np.cos(x)
    u = apply_bc(u, t)
    return u;

def explicit(u, dt, dx):
    u_new = u.copy()
    u_new[1:] = u[1:] - dt / dx * u[1:] * (u[1:] - u[:-1])
    return u_new;

def solver(u, nt, dt, dx):
    history = [u.copy()]

    for i in range(1, nt + 1):
        t = i * dt

        u_new = explicit(u, dt, dx)
        u_new = apply_bc(u_new, t)

        history.append(u_new.copy())

        u = u_new
        
    return np.array(history)


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
    p_space = get_space_params()

    x = get_grid(p_space["L"], p_space["nx"])
    u0 = init_cond(x, 0)
    
    p_time = get_time_params(np.max(abs(u0)), p_space["dx"])

    history = solver(u0, p_time["nt"], p_time["dt"], p_space["dx"])
    animate_solution(x, history, p_time["dt"], n_frames=300)