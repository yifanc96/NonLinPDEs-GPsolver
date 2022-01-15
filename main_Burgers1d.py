#%%

# argparse for command lines
import argparse

# jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.config import config; 
config.update("jax_enable_x64", True)
import numpy as onp

# solver
from src.solver import solver_GP

# solving Burgers: u_t+ alpha u u_x- nu u_xx=0
def get_parser():
    parser = argparse.ArgumentParser(description='NonLinElliptic equation GP solver')
    
    # equation parameters
    parser.add_argument("--alpha", type=float, default = 1.0)
    parser.add_argument("--nu", type=float, default = 0.02)
    
    # kernel setting
    parser.add_argument("--kernel", type=str, default='anisotropic_Gaussian')
    parser.add_argument("--kernel_parameter", type = float, nargs='+', default = [0.3,0.5])
    parser.add_argument("--nugget", type = float, default = 1e-5)
    parser.add_argument("--nugget_type", type = str, default = "adaptive", choices = ["adaptive","identity", 'none'])
    
    # sampling points
    parser.add_argument("--sampled_type", type = str, default = 'random', choices=['random','grid'])
    parser.add_argument("--N_domain", type = int, default = 1000)
    parser.add_argument("--N_boundary", type = int, default = 200)
    
    # GN iterations
    parser.add_argument("--method", type = str, default = 'elimination')
    parser.add_argument("--initial_sol", type = str, default = 'rdm')
    parser.add_argument("--GNsteps", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=1)
    
    # logs and visualization
    parser.add_argument("--print_hist", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=True)
    args = parser.parse_args()    
    
    return args

# get the parameters
cfg = get_parser()

##### step 0: initialize the solver
alpha = cfg.alpha
nu = cfg.nu
solver = solver_GP(cfg, PDE_type = "Burgers")

##### step 1: set the equation, rhs, bdy
@jit
def u(x1, x2):
    return -jnp.sin(jnp.pi*x2)*(x1==0) + 0*(x2==0)
@jit
def f(x1, x2):
    return 0
solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[-1,1]]))
# domain (t,x) in (0,1)*(-1,1)

##### step 2: sample points
solver.auto_sample(cfg.N_domain, cfg.N_boundary, sampled_type = cfg.sampled_type)
if cfg.show_figure:
    solver.show_sample()  # show the scattered figure of the sample

###### step 3: solve the equation using GP + GN iterations
solver.solve()
if cfg.show_figure:
    solver.show_loss_hist() # show the plot of the loss hist

##### step 4: error calculation on test points
# GP interpolation and test accuracy
# get truth solution
[Gauss_pts, weights] = onp.polynomial.hermite.hermgauss(80)
def u_truth(x1, x2):
    temp = x2-jnp.sqrt(4*nu*x1)*Gauss_pts
    val1 = weights * jnp.sin(jnp.pi*temp) * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*nu))
    val2 = weights * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*nu))
    return -jnp.sum(val1)/jnp.sum(val2)

N_pts = 60
xx= jnp.linspace(0, 1, N_pts)
yy = jnp.linspace(-1, 1, N_pts)
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)

test_truth = vmap(u_truth)(X_test[:,0],X_test[:,1])
solver.test(X_test)
solver.get_test_error(test_truth)
if cfg. show_figure:
    solver.contour_of_test_err(XX,YY)