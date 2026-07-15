from matplotlib import pyplot as plt
import scipy as sp
import numpy as np
from matplotlib.animation import FuncAnimation

def get_space_params():
    L = 1
    nx = 500
    dx = L / (nx - 1)

    return{ 
        "L": L, "nx": nx, "dx": dx
    }

def get_time_params(u_max, dx):
    T = 9
    sigma = 0.5
    dt = sigma * dx / u_max
    nt = int(T / dt)

    return{
        "T": T, "sigma": sigma, "dt": dt, "nt": nt
    }

def make_grid(L, nx):
    return np.linspace(0, L, nx);

def apply_bc(u, t):
    u[0] = 0
    return u;

def init_cond(x, t):
    u = np.zeros(len(x))
    u = np.sin(2 * np.pi * x)
    return u;

def flux(u):
    return 0.5 * u**2;

def numerical_flux(uR, uL, dx, dt, scheme = "rusanov"):
    fR = flux(uR)
    fL = flux(uL)
    
    if scheme == "rusanov":
        alpha = np.maximum(np.abs(uR), np.abs(uL))
    elif scheme == "lax_friedrichs":
        alpha = dx / dt
    elif scheme == "lax_wendroff":
        a = 0.5 * (uR + uL)
        alpha = a**2 * dt / dx
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    F = 0.5 * (fR + fL) - alpha / 2 * (uR - uL)
    return F;

def explicit_scheme(u, dt, dx, scheme = "rusanov"):
    F = numerical_flux(u[1:], u[:-1], dx, dt, scheme)           # We get the numerical flux ranging from F_1/2 to F_nx-3/2

    u_new = u.copy()
    u_new[1:-1] = u[1:-1] - dt / dx * (F[1:] - F[:-1])
    return u_new;

def solver(u, dt, nt, dx, scheme = "rusanov"):
    history = [u.copy()]

    print(f"Using {scheme}")
    for i in range(1, nt + 1):
        t = i * dt

        u_new = explicit_scheme(u, dt, dx, scheme)
        u_new = apply_bc(u_new, t)

        u = u_new

        history.append(u_new.copy())

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

    x = make_grid(p_space["L"], p_space["nx"])
    u0 = init_cond(x, 0)

    p_time = get_time_params(np.max(np.abs(u0)), p_space["dx"])

    history = solver(u0, p_time["dt"], p_time["nt"], p_space["dx"], "lax_friedrichs")

    animate_solution(x, history, p_time["dt"], n_frames=1000)