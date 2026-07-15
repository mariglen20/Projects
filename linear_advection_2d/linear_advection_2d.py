import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def get_space_params():
    Lx = 1
    Ly = 1
    nx = 200
    ny = 200
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    nu = 0.01
    a = 1
    b = 0.5
    return{
        "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny, "dx": dx, "dy": dy, "nu": nu, "a": a, "b": b
    }

def get_dt(dx, dy, nu, a, b, CFL):
    dt = CFL / (abs(a) / dx + abs(b) / dy + 2 * nu * (1 / dx**2 + 1/ dy**2))
    return dt;

def get_time_params():
    T = 2
    CFL = 1
    return{
        "T": T, "CFL": CFL
    }

def make_grid(Lx, Ly, nx, ny):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    X, Y = np.meshgrid(x, y)
    return x, y, X, Y;

def apply_bc(u, x, y):
    u[0, :] = 0                     # Dirichlet: u(x, 0) = sin(x) 
    u[:, 0] = 0                     # Dirichlet: u(0, y) = e^y 
    u[-1, :] = u[-2, :]             # Neumann: (du/dy)_y=Ly = 0 
    u[:, -1] = u[:, -2]             # Neumann: (du/dx)_x=Lx = 0 
    return u;

def init_cond(x, y, X, Y):
    u = np.zeros((len(y), len(x)))
    u = np.exp( -(Y - 0.5)**2 - (X - 0.5)**2)
    u = apply_bc(u, x, y)
    return u;

def flux_x(u, a):
    return a * u;
    
def flux_y(u, b):
    return b * u;

def numerical_flux(uL, uR, uD, uU, a, b):
    fL = flux_x(uL, a)
    fR = flux_x(uR, a)

    fD = flux_y(uD, b)
    fU = flux_y(uU, b)
    
    alpha_x = abs(a)
    alpha_y = abs(b)

    F = 0.5 * ( (fL + fR) - alpha_x * (uR - uL) ) 
    G = 0.5 * ( (fD + fU) - alpha_y * (uU - uD) )

    return F, G;

def explicit_scheme(u, a, b, nu, dt, dx, dy):
    F, G = numerical_flux(u[:, :-1], u[:, 1:], u[:-1, :], u[1:, :], a, b)

    u_new = u.copy()
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] \
        - dt / dx * (F[1:-1, 1:] - F[1:-1, :-1]) \
        - dt / dy * (G[1:, 1:-1] - G[:-1, 1:-1]) \
        + nu * dt/ dx**2 * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) \
        + nu * dt/ dy**2 * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
    return u_new;

def solver(u, a, b, nu, T, dt, dx, dy, x, y):
    history = [u.copy()]
    times = [0.0]
    t = 0

    while t < T:
        t += dt
        u = explicit_scheme(u, a, b, nu, dt, dx, dy)
        u = apply_bc(u, x, y)

        history.append(u.copy())
        times.append(t)

    return np.array(history), np.array(times);


def animate_solution(history, times, Lx, Ly):

    fig, ax = plt.subplots()
    
    vmin, vmax = history.min(), history.max()
    im = ax.imshow(history[0], origin='lower', extent=[0,Lx,0,Ly],
               aspect='auto', vmin=vmin, vmax=vmax)

    plt.colorbar(im, ax=ax)

    time_text = ax.set_title(f"t = {times[0]:.3f}")

    def update(i):
        im.set_data(history[i])
        ax.set_title(f"t = {times[i]:.3f}")
        return [im]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=30,
        blit=False
    )

    plt.show()

if __name__ == "__main__":
    p_space = get_space_params()
    p_time = get_time_params()

    x, y, X, Y = make_grid(p_space["Lx"], p_space["Ly"], p_space["nx"], p_space["ny"])

    u0 = init_cond(x, y, X, Y)

    dt = get_dt(p_space["dx"], p_space["dy"], p_space["nu"], p_space["a"], p_space["b"], p_time["CFL"])

    history, times = solver(u0, p_space["a"], p_space["b"], p_space["nu"],
                        p_time["T"], dt, p_space["dx"], p_space["dy"], x, y)


    animate_solution(history, times, p_space["Lx"], p_space["Ly"])