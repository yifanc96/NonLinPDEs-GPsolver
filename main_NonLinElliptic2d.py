#%%
# argparse for command lines
import argparse

# jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import jax
jax.config.update("jax_enable_x64", True)

# numpy
import numpy as onp

# solver
from src.solver import solver_GP

# solving nonlinear elliptic: -Delta u + alpha*u^m = f in [0,1]^2
def get_parser():
    parser = argparse.ArgumentParser(description='NonLinElliptic equation GP solver')
    
    # equation parameters
    parser.add_argument("--alpha", type=float, default = 1.0)
    parser.add_argument("--m", type=float, default = 3.0)
    
    # kernel setting
    parser.add_argument("--kernel", type=str, default='Gaussian')
    parser.add_argument("--kernel_parameter", type = float, default = 0.2)
    parser.add_argument("--nugget", type = float, default = 1e-13)
    parser.add_argument("--nugget_type", type = str, default = "adaptive", choices = ["adaptive","identity", 'none'])
    
    # sampling points
    parser.add_argument("--sampled_type", type = str, default = 'random', choices=['random','grid'])
    parser.add_argument("--N_domain", type = int, default = 900)
    parser.add_argument("--N_boundary", type = int, default = 124)
    
    # GN iterations
    parser.add_argument("--method", type = str, default = 'elimination', choices=['elimination','relaxation'])
    parser.add_argument("--pen_lambda", type =float, default = 1e-10) # for relaxation approach
    parser.add_argument("--initial_sol", type = str, default = 'rdm')
    parser.add_argument("--GNsteps", type=int, default=4)
    parser.add_argument("--step_size", type=int, default=1)
    
    # logs and visualization
    parser.add_argument("--print_hist", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=True)
    args = parser.parse_args()    
    
    return args

# get the parameters
cfg = get_parser()

##### step 0: initialize the solver
solver = solver_GP(cfg, PDE_type = "Nonlinear_elliptic")

##### step 1: set the equation, rhs, bdy
alpha = cfg.alpha
m = cfg.m
@jit # just in time compilation makes the code faster
def u(x1, x2):
    return jnp.sin(jnp.pi*x1) * jnp.sin(jnp.pi*x2) + 2*jnp.sin(4*jnp.pi*x1) * jnp.sin(4*jnp.pi*x2)
@jit
def f(x1, x2):
    return -grad(grad(u,0),0)(x1, x2)-grad(grad(u,1),1)(x1, x2)+alpha*(u(x1, x2)**m)
solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[0,1]]), print_option=cfg.print_hist)

##### step 2: sample points
solver.auto_sample(cfg.N_domain, cfg.N_boundary, sampled_type = cfg.sampled_type, print_option=cfg.print_hist)
if cfg.show_figure:
    solver.show_sample()  # show the scattered figure of the sample

##### step 3: solve the equation using GP + GN iterations
solver.solve(method = cfg.method, pen_lambda = cfg.pen_lambda, print_option=cfg.print_hist)
if cfg.show_figure:
    solver.show_loss_hist() # show the plot of the loss hist

##### step 4: error calculation on training points
pts_truth = vmap(u)(solver.eqn.X_domain[:,0],solver.eqn.X_domain[:,1])
solver.collocation_pts_err(pts_truth)

##### step 5: error calculation on test points
# GP interpolation to test points and test accuracy
N_pts = 60  # grid points, in each dimension
xx= jnp.linspace(0, 1, N_pts)
yy = jnp.linspace(0, 1, N_pts)
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1) # test data points
test_truth = vmap(u)(X_test[:,0],X_test[:,1])
solver.test(X_test)
solver.get_test_error(test_truth)
if cfg.show_figure:
    solver.contour_of_test_err(XX,YY)
    
