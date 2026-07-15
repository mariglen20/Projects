from matplotlib import pyplot as plt
import scipy as sp
import numpy as np 
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded

# u_t + f(u)_x = nu * u_xx

def get_space_params():
    L = 1
    nx = 1000
    dx = L / (nx - 1)
    nu = 0.01
    return{ "L": L, "nx": nx, "dx": dx, "nu": nu}

def get_time_params():
    T = 5
    CFL = 1
    return{ "T": T, "CFL": CFL}

def get_dt(CFL, dx, u_max):
    dt_adapt = CFL * dx / u_max
    dt = np.minimum(dt_adapt, 0.01)
    return dt;

def make_grid(L, nx):
    return np.linspace(0, L, nx);

def apply_bc(u, t):
    u[0] = np.sin(t)
    u[-1] = np.sin(t)
    return u;

def init_cond(x):
    u = np.zeros(len(x))
    u = np.sin(2 * np.pi * x)
    u = apply_bc(u, 0)
    return u;

def build_matrix(nu, dx, nx, dt):
    r = nu * dt / dx**2
    n = nx - 2

    ab = np.zeros((3, n))
    ab[1, :] = 1 + 2 * r
    ab[0, 1:] = -r
    ab[2, :-1] = -r
    return ab, r;

def flux(u):
    return 0.5 * u**2;

def num_flux(uL, uR, dt, dx, scheme = "rusanov"):
    fL = flux(uL)
    fR = flux(uR)

    if scheme == "rusanov":
        alpha = np.maximum(np.abs(uL), np.abs(uR))
    elif scheme == "lax_friedrichs":
        alpha = dx / dt
    elif scheme == "lax_wendroff":
        a = 0.5 * (uR + uL)
        alpha = a**2 * dt / dx
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    F = 0.5 * ( fR + fL - alpha * (uR - uL) )
    return F;
                
def build_forcing(u, r, dt, t, dx, nx, scheme = "rusanov"):
    F = num_flux(u[:-1], u[1:], dt, dx, scheme)

    bc = np.zeros(nx - 2)
    forcing = u[1:-1] + r * apply_bc(bc, t)  - dt / dx * (F[1:] - F[:-1])
    return forcing;


def solver(u, nu, CFL, T, dx, nx, scheme = "rusanov"):
    history = [u.copy()]
    times = [0.0]     
    t = 0

    while t < T:
        u_max = np.max(np.abs(u))
        dt = get_dt(CFL, dx, u_max)    
        dt = np.minimum(dt, T - t)

        t += dt
        
        ab, r = build_matrix(nu, dx, nx, dt)
        b = build_forcing(u, r, dt, t, dx, nx, scheme)

        u[1:-1] = solve_banded((1,1), ab, b)
        u = apply_bc(u, t)

        history.append(u.copy())
        times.append(t) 

    return np.array(history), np.array(times)

def animate_solution(x, history, times, n_frames=500):
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
        time_text.set_text(f"t = {times[i]:.3f}")
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
    p_time = get_time_params()

    x = make_grid(p_space["L"], p_space["nx"])
    u0 = init_cond(x)

    history, times = solver(u0, p_space["nu"], p_time["CFL"], p_time["T"], 
                     p_space["dx"], p_space["nx"], scheme = "lax_wendroff")
    
    animate_solution(x, history, times, n_frames=300)