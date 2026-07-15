import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def get_params():
    a = -2
    b = 2
    L = b - a
    nx = 500
    dx = L / (nx - 1)

    alpha = 1

    T = 200
    r = 0.9
    dt = r * dx / alpha
    nt = int(T / dt) 

    return{
        "a": a, "b": b, "L": L, "nx": nx, "dx": dx,
        "alpha": alpha, "nt": nt, "dt": dt, "r": r
    }

def make_grid(a, b, nx):
    x = np.linspace(a, b, nx)
    return x;

def apply_bc(u, t):
    u[0] = np.cos(t)
    return u;

def init_cond(x):
    u = np.zeros(len(x))
    u = np.sin(x)
    u = apply_bc(u, 0)
    return u;

def explicit_scheme(u, r, dt):
    u[1:] = (1 - r) * u[1:] + r * u[:-1]
    u = apply_bc(u, dt)
    return u;

def solver(u, r, nt, dt):
    history = [u.copy()]
    for i in range(nt):
        u = explicit_scheme(u, r, i * dt)
        history.append(u.copy())
    return np.array(history);


# def plot_solution(x, history, steps_to_plot, dt):
#     fig, ax = plt.subplots()
#     for i in steps_to_plot:
#         ax.plot(x, history[i], label=f"t = {i * dt:.2f}")
#     ax.set_xlabel("x")
#     ax.set_ylabel("u")
#     ax.grid()
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

def animate_solution(x, history, dt):
    fig, ax = plt.subplots()
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

    ani = FuncAnimation(fig, update, frames=len(history), interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    p = get_params()

    x = make_grid(p["a"], p["b"], p["nx"])
    u0 = init_cond(x)

    history = solver(u0, p["r"], p["nt"], p["dt"])

    # plot_solution(x, history, steps_to_plot=[0, 100, 300, 500], dt = p["dt"])
    animate_solution(x, history, p["dt"])
