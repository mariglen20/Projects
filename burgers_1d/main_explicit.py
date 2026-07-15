from matplotlib import pyplot as plt
import scipy as sp
import numpy as np
from matplotlib.animation import FuncAnimation

# Solving PDE: u_t + (u**2 / 2)_x = nu * u_xx

# Space related parameters (divided from time params for better control)
def get_space_params():
    L = 1
    nx = 1000
    dx = L / (nx - 1)
    nu = 0.01
    return{
        "L": L, "nx": nx, "dx": dx, "nu": nu
    }

def get_dt(CFL, u_max, dx, nu):
    dt = CFL * 1 / ( u_max / dx + 2 * nu / dx**2)       # adaptive dt^n for explicit scheme
    return dt;

def get_time_params():
    T = 10
    CFL = 1
    return{
        "T": T, "CFL": CFL
    }

# Initializing grid
def make_grid(L, nx):
    return np.linspace(0, L, nx);

# Applying BC 
def apply_bc(u, t, dx):
    u[0] = 1 + np.sin(t)                            # Dirichlet BC on left extremity 
    u[-1] = u[-2]                                   # Neumann BC on right extremity
    return u;

# Determining IC
def init_cond(x, dx):
    u = np.zeros(len(x))
    u[1:-1] += 1                                # Onda quadra unitaria
    u = apply_bc(u, 0, dx)                      # Compatibility with BC 
    return u;

# Defining physical flux
def flux(u):
    return 0.5 * u**2;

# Defining numerical flux and various conservative schemes
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

    F = 0.5 * ( (fL + fR) - alpha * (uR - uL) )
    return F;

# Definition of the explicit scheme used at the n step
def explicit_scheme(u, dt, dx, nu, scheme = "rusanov"):
    F = numerical_flux(u[1:], u[:-1], dx, dt, scheme)

    u_new = u.copy()
    u_new[1:-1] = u[1:-1] - dt / dx * (F[1:] - F[:-1]) + nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])

    return u_new;
    
def solver(u, T, dx, nu, scheme, CFL):
    history = [u.copy()]
    times = [0.0]                               # Get real time 
    t = 0

    print(f"Scheme: {scheme}")

    while t < T:
        u_max = np.max(np.abs(u))

        dt = get_dt(CFL, u_max, dx, nu)         # Getting the time step at the n iteration
        dt = min(dt, T - t)                     # Avoiding time overshoot
        t += dt                                 # Increasing physical time 

        u = explicit_scheme(u, dt, dx, nu, scheme)      # Solving for the current step 
        u = apply_bc(u, t, dx)                          # Then applying the BC 

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

    u0 = init_cond(x, p_space["dx"])

    history, times = solver(u0, p_time["T"], p_space["dx"], p_space["nu"], "lax_wendroff", p_time["CFL"])

    animate_solution(x, history, times, n_frames=300)