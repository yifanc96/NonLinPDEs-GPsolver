#%%
# jax
import jax.numpy as jnp
from jax import vmap
from jax.config import config; 
config.update("jax_enable_x64", True)
import numpy as onp
# solver
from solver import solver_GP
# dict to class attribute for configuration (cfg) file
import munch

# solving nonlinear elliptic (NLE): -Delta u + alpha*u^m = f in [0,1]^2
cfg_Burgers =munch.munchify({
    # basic set-up for equations
    'alpha': 1,
    'nu': 0.02,
    # kernel selection
    'kernel': 'anisotropic Gaussian', 
    'kernel_parameter': [1/3,1/20],
    'nugget': 1e-5,
    'nugget_type': 'adaptive',
    # optimiation
    'max_iter': 10, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

show_figure = True # whether to show solution and loss figures

# step 0: initialize the solver
alpha = cfg_Burgers.alpha
nu = cfg_Burgers.nu
solver = solver_GP(cfg_Burgers, PDE_type = "Burgers")

# step 1: set the equation, rhs, bdy
def u(x1, x2):
    return -jnp.sin(jnp.pi*x2)*(x1==0) + 0*(x2==0)

def f(x1, x2):
    return 0
solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[-1,1]]))
# domain (t,x) in (0,1)*(-1,1)

# step 2: sample points
# we use automatic random sampling here
N_domain = 900
N_boundary = 124
solver.auto_sample(N_domain, N_boundary, sampled_type = 'random')
if show_figure:
    solver.show_sample()  # show the scattered figure of the sample

# step 3: solve the equation using GP + GN iterations
solver.solve()
if show_figure:
    solver.show_loss_hist() # show the plot of the loss hist

# GP interpolation and test accuracy
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
if show_figure:
    solver.contour_of_test_err(XX,YY)