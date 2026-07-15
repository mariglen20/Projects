import numpy as np


def laplacian(u, grid):
    """
    Discrete Laplacian with centered finite differences.
    Standard 5-point stencil on interior nodes [1:-1, 1:-1].
    """
    lap = np.zeros_like(u)

    lap[1:-1, 1:-1] = (
        - u[1:-1, 1:-1] * (2/grid.dx**2 + 2/grid.dy**2)
        + u[2:,   1:-1] / grid.dy**2
        + u[:-2,  1:-1] / grid.dy**2
        + u[1:-1, 2:]   / grid.dx**2
        + u[1:-1, :-2]  / grid.dx**2
    )
    return lap


def gradient(p, grid):
    """
    Centered finite difference gradient of scalar p.
    Returns (dpdx, dpdy), both zero on borders.
    Used for: general diagnostics, pressure gradient check.
    """
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)

    dpdx[1:-1, 1:-1] = (p[1:-1, 2:]  - p[1:-1, :-2]) / (2*grid.dx)
    dpdy[1:-1, 1:-1] = (p[2:,   1:-1] - p[:-2, 1:-1]) / (2*grid.dy)

    return dpdx, dpdy


def gradient_backward(p, grid):
    """
    Backward finite difference gradient of scalar p.
    Returns (dpdx, dpdy), both zero on borders.
    Used for: velocity correction in projection step.
    Consistent with divergence_forward so that div(grad(p)) = laplacian(p).
    """
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)

    dpdx[1:-1, 1:-1] = (p[1:-1, 1:-1] - p[1:-1, :-2]) / grid.dx
    dpdy[1:-1, 1:-1] = (p[1:-1, 1:-1] - p[:-2,  1:-1]) / grid.dy

    return dpdx, dpdy


def divergence(u, v, grid):
    """
    Centered finite difference divergence of vector field (u, v).
    Returns scalar field, zero on borders.
    Used for: diagnostics and convergence checks.
    """
    dudx, _ = gradient(u, grid)
    _, dvdy = gradient(v, grid)

    return dudx + dvdy


def divergence_forward(u, v, grid):
    """
    Forward finite difference divergence of vector field (u, v).
    Returns scalar field, zero on borders.
    Used for: right-hand side of Poisson equation in projection method.
    Consistent with gradient_backward so that div_forward(grad_backward(p)) = laplacian(p).
    """
    div = np.zeros((grid.Ny, grid.Nx))

    div[1:-1, 1:-1] = (
        (u[1:-1, 2:]  - u[1:-1, 1:-1]) / grid.dx +
        (v[2:,   1:-1] - v[1:-1, 1:-1]) / grid.dy
    )
    
    return div


def convection(u, v, grid):
    """
    Convective term (u . grad)u in non-conservative form.
    Returns (u_term, v_term):
        u_term = u * du/dx + v * du/dy
        v_term = u * dv/dx + v * dv/dy
    """
    dudx, dudy = gradient(u, grid)
    dvdx, dvdy = gradient(v, grid)

    u_term = u * dudx + v * dudy
    v_term = u * dvdx + v * dvdy

    return u_term, v_term


def vorticity(u, v, grid):
    """"
    Vorticity operator.
    Returns (dv/dx - du/dy)
    """
    _, dudy = gradient(u, grid)
    dvdx, _ = gradient(v, grid)

    return dvdx - dudy
