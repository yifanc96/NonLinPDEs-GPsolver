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

from scipy.interpolate import griddata
from standard_solver.FD_for_Darcy_flow import FD_Darcy_flow_2d

# visulization: plot figures
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# figure format; comment out them if errors appear
fsize = 15
tsize = 15
tdir = 'in'
major = 5.0
minor = 3.0
lwidth = 0.8
lhandle = 2.0
plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.linewidth'] = lwidth
plt.rcParams['legend.handlelength'] = lhandle

fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))


# solving Darcy flow -div(a grad u) = f
cfg_Darcy_flow2d =munch.munchify({
    # kernel selection
    'kernel': 'Gaussian', 
    'kernel_parameter': 0.2,
    'nugget': 1e-10,
    'nugget_type': 'adaptive',
    # data
    'noise_level': 1e-3,
    # optimiation
    'max_iter': 12, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

show_figure = True # whether to show solution and loss figures

# step 0: initialize the solver
solver = solver_GP(cfg_Darcy_flow2d, PDE_type = "Darcy_flow2d")

# step 1: set the equation, rhs, bdy
def u(x1, x2):
    return 0
def f(x1, x2):
    return 1
solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[0,1]]))

# step 2: sample points
# we use automatic random sampling here
N_domain = 400
N_boundary = 100
N_data = 60
solver.auto_sample_IP(N_domain, N_boundary, N_data, sampled_type = 'random')
if show_figure:
    solver.show_sample_IP()  # show the scattered figure of the sample

# step 3: get the observed data
N_pts_per_dim = 80
xx = onp.linspace(0, 1, N_pts_per_dim)
yy = onp.linspace(0, 1, N_pts_per_dim)
XX, YY = onp.meshgrid(xx, yy)
XXv = onp.array(XX.flatten())
YYv = onp.array(YY.flatten())
# a(x) truth
def a(x1, x2):
    c=1
    return jnp.exp(c*jnp.sin(2*jnp.pi*x1)+c*jnp.sin(2*jnp.pi*x2))+jnp.exp(-c*jnp.sin(2*jnp.pi*x1)-c*jnp.sin(2*jnp.pi*x2))
u_truth_grid = FD_Darcy_flow_2d(N_pts_per_dim-2, a,f)
u_truth_grid_vec = onp.reshape(u_truth_grid, (-1,1))
def get_data_u(x,y):
    return griddata((XXv, YYv), u_truth_grid_vec, (x,y), method='linear')
data_u = onp.vectorize(get_data_u)(solver.eqn.X_data[:,0], solver.eqn.X_data[:,1])
solver.get_observed_data(data_u, cfg_Darcy_flow2d.noise_level)

# step 4: solve the equation using GP + GN iterations
solver.solve()
if show_figure:
    solver.show_loss_hist() # show the plot of the loss hist


# GP interpolation and test accuracy
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)
solver.test(X_test)
test_u = onp.reshape(solver.eqn.extended_sol_u,(N_pts_per_dim,N_pts_per_dim))
test_a = onp.reshape(solver.eqn.extended_sol_a,(N_pts_per_dim,N_pts_per_dim))

test_truth_u = u_truth_grid
test_truth_a = onp.reshape(vmap(a)(X_test[:,0],X_test[:,1]),(N_pts_per_dim,N_pts_per_dim))

# plot true and obtained solutions
if show_figure:
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(221)
    a_true_contourf=ax.contourf(XX, YY, test_truth_a, 50, cmap=plt.cm.coolwarm)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Truth $a(x)$')
    fig.colorbar(a_true_contourf, format=fmt)


    ax = fig.add_subplot(222)
    a_contourf=ax.contourf(XX, YY, onp.exp(test_a), 50, cmap=plt.cm.coolwarm)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Recovered $a(x)$')
    fig.colorbar(a_contourf, format=fmt)

    ax = fig.add_subplot(223)
    u_true_contourf=ax.contourf(XX, YY, test_truth_u, 50, cmap=plt.cm.coolwarm)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Truth $u(x)$')
    fig.colorbar(u_true_contourf, format=fmt)


    ax = fig.add_subplot(224)
    u_contourf=ax.contourf(XX, YY, test_u, 50, cmap=plt.cm.coolwarm)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Recovered $u(x)$')
    fig.colorbar(u_contourf, format=fmt)

    plt.show()



