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

# solving regularized Eikonal: |grad u|^2 = f + eps*Delta u
cfg_Eikonal =munch.munchify({
    # basic set-up for equations
    'eps': 1e-1,
    # kernel selection
    'kernel': 'Gaussian', 
    'kernel_parameter': 0.2,
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
solver = solver_GP(cfg_Eikonal, PDE_type = "Eikonal")

# step 1: set the equation, rhs, bdy
def u(x1, x2):
    return 0
def f(x1, x2):
    return 1
solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[0,1]]))

# step 2: sample points
# we use automatic random sampling here
N_domain = 2400-240
N_boundary = 240
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
# u_truth_contourf=ax.contourf(XX, YY, test_truth.reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
# fig.colorbar(u_truth_contourf,format=fmt)

# int_data=ax.scatter(solver.eqn.X_domain[:, 0], solver.eqn.X_domain[:, 1], marker="x", label='Interior nodes')
# bd_data=ax.scatter(solver.eqn.X_boundary[:, 0], solver.eqn.X_boundary[:, 1], marker="x", label='Boundary nodes')
# int_data.set_clip_on(False)
# bd_data.set_clip_on(False)

# ax.legend(loc="upper right")
# plt.title('Collocation points')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$') 

# # plot the iteration history
# ax = fig.add_subplot(132)
# plt.plot(onp.arange(solver.config.max_iter+1),solver.eqn.loss_hist)
# plt.yscale("log")
# plt.title('Loss function history')
# plt.xlabel('Gauss-Newton step')

# # plot the contour error
# ax = fig.add_subplot(133)
# u_contourf=ax.contourf(XX, YY, solver.test_err_all.reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.title('Contour of errors')
# fig.colorbar(u_contourf, format=fmt)
# plt.show()

# fig.tight_layout()
# fig.savefig('data_Eikonal_eqn_demon.pdf', bbox_inches='tight',dpi=100,pad_inches=0.1)
# # %%
