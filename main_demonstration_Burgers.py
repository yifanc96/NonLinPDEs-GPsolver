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

# solving Burgers: u_t+ alpha u u_x- nu u_xx=0
cfg_Burgers =munch.munchify({
    # basic set-up for equations
    'alpha': 1,
    'nu': 0.02,
    # kernel selection
    'kernel': 'anisotropic Gaussian', 
    'kernel_parameter': [1/3,1/20],
    'nugget': 1e-10,
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
N_domain = 2000
N_boundary = 400
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
    
# # plot figures
# # visulization: plot figures
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# # figure format; comment out them if errors appear
# fsize = 15
# tsize = 15
# tdir = 'in'
# major = 5.0
# minor = 3.0
# lwidth = 0.8
# lhandle = 2.0
# plt.style.use('default')
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = fsize
# plt.rcParams['legend.fontsize'] = tsize
# plt.rcParams['xtick.direction'] = tdir
# plt.rcParams['ytick.direction'] = tdir
# plt.rcParams['xtick.major.size'] = major
# plt.rcParams['xtick.minor.size'] = minor
# plt.rcParams['ytick.major.size'] = 5.0
# plt.rcParams['ytick.minor.size'] = 3.0
# plt.rcParams['axes.linewidth'] = lwidth
# plt.rcParams['legend.handlelength'] = lhandle

# fmt = ticker.ScalarFormatter(useMathText=True)
# fmt.set_powerlimits((0, 0))

# fig = plt.figure(figsize=(15,5))

# # plot the collocation pts
# ax = fig.add_subplot(131)
# u_truth_contourf=ax.contourf(YY, XX, test_truth.reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
# fig.colorbar(u_truth_contourf,format=fmt)

# int_data=ax.scatter(solver.eqn.X_domain[:, 1], solver.eqn.X_domain[:, 0], marker="x", label='Interior nodes')
# bd_data=ax.scatter(solver.eqn.X_boundary[:, 1], solver.eqn.X_boundary[:, 0], marker="x", label='Boundary nodes')
# int_data.set_clip_on(False)
# bd_data.set_clip_on(False)

# ax.legend(loc="upper right")
# plt.title('Collocation points')
# plt.xlabel('$x$')
# plt.ylabel('$t$')

# # plot the iteration history
# ax = fig.add_subplot(132)
# plt.plot(onp.arange(solver.config.max_iter+1),solver.eqn.loss_hist)
# plt.yscale("log")
# plt.title('Loss function history')
# plt.xlabel('Gauss-Newton step')

# # plot the contour error
# ax = fig.add_subplot(133)
# u_contourf=ax.contourf(YY, XX, solver.test_err_all.reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
# plt.xlabel('$x$')
# plt.ylabel('$t$')
# plt.title('Contour of errors')
# fig.colorbar(u_contourf, format=fmt)
# plt.show()

# fig.tight_layout()
# fig.savefig('data_Burgers_eqn_demon.pdf', bbox_inches='tight',dpi=100,pad_inches=0.1)