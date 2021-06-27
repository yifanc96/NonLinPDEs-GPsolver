#%%
# jax
import jax.numpy as jnp
from jax.config import config; 
config.update("jax_enable_x64", True)
import numpy as onp
# solver
from solver import solver_GP
from standard_solver.Cole_Hopf_for_Eikonal import solve_Eikonal
# dict to class attribute for configuration (cfg) file
import munch

# solving nonlinear elliptic (NLE): -Delta u + alpha*u^m = f in [0,1]^2
cfg_Eikonal =munch.munchify({
    # basic set-up for equations
    'eps': 1e-1,
    # kernel selection
    'kernel': 'Gaussian', 
    'kernel_parameter': 0.2,
    'nugget': 1e-10,
    'nugget_type': 'adaptive',
    # optimiation
    'max_iter': 8, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

show_figure = True # whether to show solution and loss figures

# step 0: initialize the solver
solver = solver_GP(cfg_Eikonal, PDE_type = "Eikonal")

# step 1: set the equation, rhs, bdy
def u(x1, x2):
    return 0
def f(x1, x2):
    return 1
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
# GP interpolation and test accuracy
N_pts = 60
xx= jnp.linspace(0, 1, N_pts)[1:-1]
yy = jnp.linspace(0, 1, N_pts)[1:-1]
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)
solver.test(X_test)

XX, YY, test_truth = solve_Eikonal(N_pts-2, cfg_Eikonal.eps)
solver.get_test_error(test_truth.flatten())
if show_figure:
    solver.contour_of_test_err(XX,YY)
