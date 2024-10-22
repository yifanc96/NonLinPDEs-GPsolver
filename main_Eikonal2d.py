#%%

# argparse for command lines
import argparse

# jax
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as onp
# solver
from src.solver import solver_GP
from reference_solver.Cole_Hopf_for_Eikonal import solve_Eikonal

# solving regularized Eikonal: |grad u|^2 = f + eps*Delta u
def get_parser():
    parser = argparse.ArgumentParser(description='Eikonal equation GP solver')
    
    # equation parameters
    parser.add_argument("--eps", type=float, default = 1e-1)
    
    # kernel setting
    parser.add_argument("--kernel", type=str, default='Gaussian')
    parser.add_argument("--kernel_parameter", type = float, default = 0.2)
    parser.add_argument("--nugget", type = float, default = 1e-5)
    parser.add_argument("--nugget_type", type = str, default = "adaptive", choices = ["adaptive","identity", 'none'])
    
    # sampling points
    parser.add_argument("--sampled_type", type = str, default = 'random', choices=['random','grid'])
    parser.add_argument("--N_domain", type = int, default = 1000)
    parser.add_argument("--N_boundary", type = int, default = 200)
    
    # GN iterations
    parser.add_argument("--method", type = str, default = 'elimination')
    parser.add_argument("--initial_sol", type = str, default = 'zero')
    parser.add_argument("--GNsteps", type=int, default=8)
    parser.add_argument("--step_size", type=int, default=1)
    
    # logs and visualization
    parser.add_argument("--print_hist", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=True)
    args = parser.parse_args()    
    
    return args

# get the parameters
cfg = get_parser()

##### step 0: initialize the solver
solver = solver_GP(cfg, PDE_type = "Eikonal")

###### step 1: set the equation, rhs, bdy
def u(x1, x2):
    return 0
def f(x1, x2):
    return 1
solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[0,1]]))

##### step 2: sample points
solver.auto_sample(cfg.N_domain, cfg.N_boundary, sampled_type = cfg.sampled_type)
if cfg.show_figure:
    solver.show_sample()  # show the scattered figure of the sample

##### step 3: solve the equation using GP + GN iterations
solver.solve()
if cfg.show_figure:
    solver.show_loss_hist() # show the plot of the loss hist

##### step 4: error calculation
# GP interpolation and test accuracy
N_pts = 60
xx= jnp.linspace(0, 1, N_pts)[1:-1]
yy = jnp.linspace(0, 1, N_pts)[1:-1]
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)
solver.test(X_test) # get the GP solution at the test points
XX, YY, test_truth = solve_Eikonal(N_pts-2, cfg.eps) # true solution
solver.get_test_error(test_truth.flatten())
if cfg.show_figure:
    solver.contour_of_test_err(XX,YY)