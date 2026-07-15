import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _style_axis(ax):
    ax.set_facecolor("#0f0f0f")
    ax.tick_params(colors="gray", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def _attach_colorbar(fig, ax, im, visible=True):
    """
    Attaches a colorbar of fixed relative size to ax using
    make_axes_locatable, so all axes keep the same width
    regardless of whether their colorbar is shown.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    if visible:
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(colors="gray", labelsize=7)
        return cb
    else:
        cax.set_visible(False)
        return None


def make_figure(grid):
    """
    Creates the figure and axes for real-time visualization.
    Layout: 2x2 grid -> |u|, p, streamlines, vorticity.
    All four axes have equal width: streamlines gets an invisible
    colorbar slot to match the others.
    """
    fig = plt.figure(figsize=(11, 9), facecolor="#0f0f0f")
    fig.suptitle("Lid-Driven Cavity Flow", fontsize=14, color="white", y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.35,
                            left=0.07, right=0.93, top=0.93, bottom=0.06)
    ax0 = fig.add_subplot(gs[0, 0])   # velocity magnitude
    ax1 = fig.add_subplot(gs[0, 1])   # pressure
    ax2 = fig.add_subplot(gs[1, 0])   # streamlines
    ax3 = fig.add_subplot(gs[1, 1])   # vorticity

    for ax in [ax0, ax1, ax2, ax3]:
        _style_axis(ax)
        ax.set_aspect("equal")

    extent = [0, grid.Lx, 0, grid.Ly]

    # --- velocity magnitude ---
    vel = np.zeros((grid.Ny, grid.Nx))
    im0 = ax0.imshow(vel, origin="lower", cmap="inferno",
                     extent=extent, vmin=0, vmax=1.2)
    _attach_colorbar(fig, ax0, im0)
    ax0.set_title(r"$|\mathbf{u}|$", color="white", fontsize=11)
    ax0.set_xlabel("x", color="gray", fontsize=8)
    ax0.set_ylabel("y", color="gray", fontsize=8)

    # --- pressure ---
    p_dummy = np.zeros((grid.Ny, grid.Nx))
    im1 = ax1.imshow(p_dummy, origin="lower", cmap="RdBu_r", extent=extent)
    _attach_colorbar(fig, ax1, im1)
    ax1.set_title(r"$p$", color="white", fontsize=11)
    ax1.set_xlabel("x", color="gray", fontsize=8)
    ax1.set_ylabel("y", color="gray", fontsize=8)

    # --- streamlines: invisible colorbar slot to align with row above ---
    ax2.set_xlim(0, grid.Lx)
    ax2.set_ylim(0, grid.Ly)
    _attach_colorbar(fig, ax2, None, visible=False)
    ax2.set_title("Streamlines", color="white", fontsize=11)
    ax2.set_xlabel("x", color="gray", fontsize=8)
    ax2.set_ylabel("y", color="gray", fontsize=8)

    # --- vorticity ---
    om_dummy = np.zeros((grid.Ny, grid.Nx))
    im3 = ax3.imshow(om_dummy, origin="lower", cmap="RdBu_r", extent=extent)
    _attach_colorbar(fig, ax3, im3)
    ax3.set_title(r"Vorticity $\omega$", color="white", fontsize=11)
    ax3.set_xlabel("x", color="gray", fontsize=8)
    ax3.set_ylabel("y", color="gray", fontsize=8)

    step_text = fig.text(0.5, 0.01, "step 0", ha="center",
                          color="gray", fontsize=9)

    plt.ion()
    plt.show()

    objects = {
        "im0": im0,
        "im1": im1,
        "ax2": ax2,
        "im3": im3,
        "step_text": step_text,
    }

    return fig, (ax0, ax1, ax2, ax3), objects


def update_plots(fig, axes, objects, u, v, p, omega, grid, step, every=50):
    """
    Call inside the time loop. Updates all four plots every `every` steps.

    u, v, p, omega : np.ndarray of shape (Ny, Nx)
    step           : current iteration number
    every          : update interval (default 50)
    """
    if step % every != 0:
        return

    ax0, ax1, ax2, ax3 = axes
    im0 = objects["im0"]
    im1 = objects["im1"]
    im3 = objects["im3"]
    step_text = objects["step_text"]

    # --- velocity magnitude ---
    vel = np.sqrt(u**2 + v**2)
    im0.set_data(vel)
    im0.set_clim(vmin=0, vmax=max(vel.max(), 1e-8))

    # --- pressure (scaled on interior nodes, excludes corner singularities) ---
    p_max = max(np.abs(p[2:-2, 2:-2]).max(), 1e-8)
    im1.set_data(p)
    im1.set_clim(vmin=-p_max, vmax=p_max)

    # --- streamlines: redraw from scratch ---
    ax2.cla()
    _style_axis(ax2)
    ax2.set_aspect("equal")
    ax2.streamplot(
        grid.x, grid.y, u, v,
        color=vel, cmap="inferno",
        linewidth=0.8, density=1.2, arrowsize=0.8,
    )
    ax2.set_xlim(0, grid.Lx)
    ax2.set_ylim(0, grid.Ly)
    ax2.set_title("Streamlines", color="white", fontsize=11)
    ax2.set_xlabel("x", color="gray", fontsize=8)
    ax2.set_ylabel("y", color="gray", fontsize=8)

    # --- vorticity (symmetric colorbar, interior nodes) ---
    om_max = max(np.abs(omega[1:-1, 1:-1]).max(), 1e-8)
    im3.set_data(omega)
    im3.set_clim(vmin=-om_max, vmax=om_max)

    step_text.set_text(f"step {step}")

    fig.canvas.draw_idle()
    plt.pause(0.001)


def final_plot(u, v, p, omega, grid):
    """
    Static high-quality plot at the end of the simulation.
    Same 2x2 aligned layout, no update overhead.
    """
    fig = plt.figure(figsize=(11, 9), facecolor="#0f0f0f")
    fig.suptitle("Lid-Driven Cavity — converged field", fontsize=14, color="white")

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.35,
                            left=0.07, right=0.93, top=0.93, bottom=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax in [ax0, ax1, ax2, ax3]:
        _style_axis(ax)
        ax.set_aspect("equal")

    extent = [0, grid.Lx, 0, grid.Ly]

    # --- velocity magnitude ---
    vel = np.sqrt(u**2 + v**2)
    im0 = ax0.imshow(vel, origin="lower", cmap="inferno", extent=extent)
    _attach_colorbar(fig, ax0, im0)
    ax0.set_title(r"$|\mathbf{u}|$", color="white", fontsize=11)
    ax0.set_xlabel("x", color="gray", fontsize=8)
    ax0.set_ylabel("y", color="gray", fontsize=8)

    # --- pressure ---
    p_max = max(np.abs(p[2:-2, 2:-2]).max(), 1e-8)
    im1 = ax1.imshow(p, origin="lower", cmap="RdBu_r",
                     extent=extent, vmin=-p_max, vmax=p_max)
    _attach_colorbar(fig, ax1, im1)
    ax1.set_title(r"$p$", color="white", fontsize=11)
    ax1.set_xlabel("x", color="gray", fontsize=8)
    ax1.set_ylabel("y", color="gray", fontsize=8)

    # --- streamlines ---
    _attach_colorbar(fig, ax2, None, visible=False)
    ax2.streamplot(
        grid.x, grid.y, u, v,
        color=vel, cmap="inferno",
        linewidth=0.9, density=1.4, arrowsize=0.9,
    )
    ax2.set_xlim(0, grid.Lx)
    ax2.set_ylim(0, grid.Ly)
    ax2.set_title("Streamlines", color="white", fontsize=11)
    ax2.set_xlabel("x", color="gray", fontsize=8)
    ax2.set_ylabel("y", color="gray", fontsize=8)

    # --- vorticity ---
    om_max = max(np.abs(omega[1:-1, 1:-1]).max(), 1e-8)
    im3 = ax3.imshow(omega, origin="lower", cmap="RdBu_r",
                     extent=extent, vmin=-om_max, vmax=om_max)
    _attach_colorbar(fig, ax3, im3)
    ax3.set_title(r"Vorticity $\omega$", color="white", fontsize=11)
    ax3.set_xlabel("x", color="gray", fontsize=8)
    ax3.set_ylabel("y", color="gray", fontsize=8)

    plt.ioff()
    plt.show()