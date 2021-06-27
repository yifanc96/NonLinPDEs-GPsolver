#%%
# jax
import jax.numpy as jnp
from jax import grad, vmap
from jax.config import config; 
config.update("jax_enable_x64", True)
import numpy as onp
# solver
from solver import solver_GP
# dict to class attribute for configuration (cfg) file
import munch

# solving nonlinear elliptic (NLE): -Delta u + alpha*u^m = f in [0,1]^2
cfg_NLE =munch.munchify({
    # basic set-up for equations
    'alpha': 1,
    'm': 3,
    # kernel selection
    'kernel': 'Gaussian', 
    'kernel_parameter': 0.2,
    'nugget': 1e-13,
    'nugget_type': 'adaptive',
    # optimiation
    'max_iter': 4, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

show_figure = True # whether to show solution and loss figures

# step 0: initialize the solver
solver = solver_GP(cfg_NLE, PDE_type = "Nonlinear_elliptic")

# step 1: set the equation, rhs, bdy
alpha = cfg_NLE.alpha
m = cfg_NLE.m
def u(x1, x2):
    return jnp.sin(jnp.pi*x1) * jnp.sin(jnp.pi*x2) + 2*jnp.sin(4*jnp.pi*x1) * jnp.sin(4*jnp.pi*x2)
def f(x1, x2):
    return -grad(grad(u,0),0)(x1, x2)-grad(grad(u,1),1)(x1, x2)+alpha*(u(x1, x2)**m)
solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[0,1]]))

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

# error calculation
# collocation points error
pts_truth = vmap(u)(solver.eqn.X_domain[:,0],solver.eqn.X_domain[:,1])
solver.collocation_pts_err(pts_truth)


# GP interpolation and test accuracy
N_pts = 60
xx= jnp.linspace(0, 1, N_pts)
yy = jnp.linspace(0, 1, N_pts)
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)

test_truth = vmap(u)(X_test[:,0],X_test[:,1])
solver.test(X_test)
solver.get_test_error(test_truth)
if show_figure:
    solver.contour_of_test_err(XX,YY)
