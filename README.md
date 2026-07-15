# CFD & Numerical Methods — Python

A collection of Python solvers for classic partial differential equations, built from scratch with NumPy/SciPy as a hands-on path through computational fluid dynamics — from the 1D heat equation up to the incompressible Navier–Stokes equations.

## Projects

| Project | Equation | Methods |
|---|---|---|
| [`heat_equation_1d`](heat_equation_1d) | 1D heat (diffusion) equation | Explicit FTCS; implicit scheme via linear system solve |
| [`linear_advection_1d`](linear_advection_1d) | 1D linear advection | Explicit upwind; Crank–Nicolson |
| [`linear_advection_2d`](linear_advection_2d) | 2D linear advection | Finite differences with CFL-based time stepping |
| [`nonlinear_advection_1d`](nonlinear_advection_1d) | 1D nonlinear (inviscid Burgers) advection | Conservative vs. non-conservative form; numerical flux formulation |
| [`burgers_1d`](burgers_1d) | 1D viscous Burgers equation | Explicit with adaptive CFL time step; semi-implicit |
| [`poisson_2d`](poisson_2d) | 2D Poisson equation | Iterative solvers with Dirichlet and Neumann boundary conditions |
| [`cavity_flow`](cavity_flow) | 2D incompressible Navier–Stokes (lid-driven cavity) | Chorin projection method on a staggered grid, pressure Poisson solver, validated against Ghia et al. (1982) |

## Highlights — `cavity_flow`

The most complete project: a modular solver for the lid-driven cavity benchmark at Re = 1000.

- Projection (fractional-step) method: intermediate velocity → pressure Poisson equation → divergence-free correction
- Modular design: separate modules for grid, boundary conditions, operators, Poisson solver, projection step, visualization, and validation
- Adaptive time step based on CFL condition
- Validation of centerline velocity profiles against the reference data of Ghia, Ghia & Shin (1982) — see [`cavity_flow/graphs`](cavity_flow/graphs)

## Requirements

```
numpy
scipy
matplotlib
```

## Running

Each project is self-contained. Run any script directly, e.g.:

```
cd cavity_flow
python main.py
```

Most scripts produce live matplotlib animations or plots of the solution.
