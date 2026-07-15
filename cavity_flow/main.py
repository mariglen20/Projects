import numpy as np
import grid
import boundary as bnd
import initial as init
import poisson as pssn
import projection as proj
import operators as ops
import visualizer as vsl
import validation as val


if __name__ == "__main__":

    "Initializing values"
    Re = 1000.0
    n_steps = 20000
    tol = 1e-6
    
    # Grid creation and instanciating poisson object
    g = grid.Grid(128, 128, 1, 1)
    poisson = pssn.PoissonSolver(g)
    poisson.test(ops, g)   # testing solver's results such as residuals

    # Initializing u, v, p fields and making sure they respect BCs
    u, v, p = init.apply_initial_conditions(g)
    u, v = bnd.apply_bc_velocity(u, v)

    # Creating plotting obejcts
    fig, axes, objects = vsl.make_figure(g)
    
    # Beginning loop
    for n in range(n_steps):
        dt = proj.compute_dt(u, v, g, Re, 0.5)                    

        u_star, v_star = proj.u_intermediate(u, v, dt, Re, g)       # obtaining intermediate velocity field (ignoring pressure gradient)
        u_star, v_star = bnd.apply_bc_velocity(u_star, v_star)      # applying BCs 

        p_old = p.copy() 

        rhs = ops.divergence_forward(u_star, v_star, g) / dt        # defining rhs of the poisson solver
        p = poisson.solve(rhs)                                      # obtaining p^(n+1) solving the possion equation


        u_new, v_new = proj.project(u_star, v_star, p, dt, g)       # projecting new pressure field to obtain u^(n+1)
        u_new, v_new = bnd.apply_bc_velocity(u_new, v_new)          # applying BCs 
        
        omega = ops.vorticity(u, v, g)                              # defining vorticity
        

        diff = np.max(np.abs(u_new - u))                            # checking convergence to stationary condition and if reached break
        if diff < tol:
            print(f"Converged at step {n}, dt={dt:.4e}")
            
            break 
        
        if n % 50 == 0:
        # dpdt = np.max(np.abs(p - p_old))
        # p_old = p.copy()
        # print("pressure change:", dpdt)
        # print("u change:", np.max(np.abs(u_new - u)))
        # print("v change:", np.max(np.abs(v_new - v)))
        # print("rhs max:", np.max(np.abs(rhs)))
            print(f"Tolerance: {diff}")

        u, v = u_new, v_new                                         # substituting values to beginning ones to restart loop

        vsl.update_plots(fig, axes, objects, u, v, p, omega, g, step = n, every = 50)


    errors = val.compute_errors(*val.get_profiles(u, v, g)[::1])
    print(f"u: L2 = {errors['u_l2']:.4e}, Linf = {errors['u_linf']:.4e}")
    print(f"v: L2 = {errors['v_l2']:.4e}, Linf = {errors['v_linf']:.4e}")

    val.plot_comparison(u, v, g, Re = Re)

    vsl.final_plot(u, v, p, omega, g)

    AHAHAHAHAH

# Converged at step 11333, dt=7.7500e-04
# u: L2=3.7953e-03, Linf=1.2274e-02
# v: L2=5.1419e-03, Linf=8.7488e-03