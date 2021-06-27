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
import copy
from standard_solver.FD_for_Darcy_flow import FD_Darcy_flow_2d

# visulization: plot figures
import matplotlib.pyplot as plt

class error_curve(object):
    def __init__(self, arr_N_pts_per_dim):
        self.arr_N_pts_per_dim = arr_N_pts_per_dim
        self.arr_num = onp.shape(arr_N_pts_per_dim)[0]
        self.L2err = onp.zeros((1,self.arr_num))
        self.Maxerr = onp.zeros((1,self.arr_num))
        self.config = []

arr_N_pts_per_dim = 2**onp.array([3,4,5,6])
print('\n [Goal] error curve for elliptic equations')
print(f'[Setting] array of number of points per dimension: {arr_N_pts_per_dim}')
print('[Setting] alpha = 1 or 0, sigma adapted to # points or fixed; compared with Finite Difference')
err_alpha_1_adapt_sigma = error_curve(arr_N_pts_per_dim)
err_alpha_1_const_sigma = error_curve(arr_N_pts_per_dim)
err_alpha_0_adapt_sigma = error_curve(arr_N_pts_per_dim)
err_alpha_0_const_sigma = error_curve(arr_N_pts_per_dim)
err_alpha_0_FD = error_curve(arr_N_pts_per_dim) # Finite difference

# solving nonlinear elliptic (NLE): -Delta u + alpha*u^m = f in [0,1]^2
cfg_default =munch.munchify({
    # basic set-up for equations
    'alpha': 1,
    'm': 3,
    # kernel selection
    'kernel': 'Gaussian', 
    'kernel_parameter': 0.2,
    'nugget': 1e-13,
    'nugget_type': 'adaptive',
    # optimiation
    'max_iter': 5, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

def get_collocation_pts_err(cfg, N_domain, N_boundary):
    # grid sampling points
    alpha = cfg.alpha
    m = cfg.m
    def u(x1, x2):
        return jnp.sin(jnp.pi*x1) * jnp.sin(jnp.pi*x2) + 2*jnp.sin(4*jnp.pi*x1) * jnp.sin(4*jnp.pi*x2)
    def f(x1, x2):
        return -grad(grad(u,0),0)(x1, x2)-grad(grad(u,1),1)(x1, x2)+alpha*(u(x1, x2)**m)
    solver = solver_GP(cfg, PDE_type = "Nonlinear_elliptic")
    solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[0,1]]))
    solver.auto_sample(N_domain, N_boundary, sampled_type = 'grid')
    solver.solve()
    pts_truth = vmap(u)(solver.eqn.X_domain[:,0],solver.eqn.X_domain[:,1])
    solver.collocation_pts_err(pts_truth)
    return solver.pts_max_err, solver.pts_L2_err

# for err_alpha_1_adapt_sigma
for iter in range(err_alpha_1_adapt_sigma.arr_num):
    # get number of points in each dimension
    N_pts_per_dim = err_alpha_1_adapt_sigma.arr_N_pts_per_dim[iter]
    # get the configuration
    cfg = copy.copy(cfg_default)
    cfg.alpha = 1
    cfg.kernel_parameter = 1/jnp.sqrt(N_pts_per_dim)
    # sampled points
    N_domain = (N_pts_per_dim-2)**2
    N_boundary = 4*(N_pts_per_dim-1)
    pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary)
    err_alpha_1_adapt_sigma.L2err[0,iter] = pts_L2_err 
    err_alpha_1_adapt_sigma.Maxerr[0,iter] = pts_max_err
    err_alpha_1_adapt_sigma.config.append(cfg)

print('\n Finished err_alpha_1_adapt_sigma')
print(f'Max err: {err_alpha_1_adapt_sigma.Maxerr}')
print(f'L2 err: {err_alpha_1_adapt_sigma.L2err}\n')

# for err_alpha_1_const_sigma
for iter in range(err_alpha_1_const_sigma.arr_num):
    # get number of points in each dimension
    N_pts_per_dim = err_alpha_1_const_sigma.arr_N_pts_per_dim[iter]
    # get the configuration
    cfg = copy.copy(cfg_default)
    cfg.alpha = 1
    cfg.kernel_parameter = 0.2
    # sampled points
    N_domain = (N_pts_per_dim-2)**2
    N_boundary = 4*(N_pts_per_dim-1)
    pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary)
    err_alpha_1_const_sigma.L2err[0,iter] = pts_L2_err 
    err_alpha_1_const_sigma.Maxerr[0,iter] = pts_max_err
    err_alpha_1_const_sigma.config.append(cfg)
    
print('\n Finished err_alpha_1_const_sigma')
print(f'Max err: {err_alpha_1_const_sigma.Maxerr}')
print(f'L2 err: {err_alpha_1_const_sigma.L2err}\n')

# for err_alpha_0_adapt_sigma
for iter in range(err_alpha_0_adapt_sigma.arr_num):
    # get number of points in each dimension
    N_pts_per_dim = err_alpha_0_adapt_sigma.arr_N_pts_per_dim[iter]
    # get the configuration
    cfg = copy.copy(cfg_default)
    cfg.alpha = 0
    cfg.kernel_parameter = 1/jnp.sqrt(N_pts_per_dim)
    # sampled points
    N_domain = (N_pts_per_dim-2)**2
    N_boundary = 4*(N_pts_per_dim-1)
    pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary)
    err_alpha_0_adapt_sigma.L2err[0,iter] = pts_L2_err 
    err_alpha_0_adapt_sigma.Maxerr[0,iter] = pts_max_err
    err_alpha_0_adapt_sigma.config.append(cfg)
    
print('\n Finished err_alpha_0_adapt_sigma')
print(f'Max err: {err_alpha_0_adapt_sigma.Maxerr}')
print(f'L2 err: {err_alpha_0_adapt_sigma.L2err}\n')

# for err_alpha_0_const_sigma
for iter in range(err_alpha_0_const_sigma.arr_num):
    # get number of points in each dimension
    N_pts_per_dim = err_alpha_0_const_sigma.arr_N_pts_per_dim[iter]
    # get the configuration
    cfg = copy.copy(cfg_default)
    cfg.alpha = 0
    cfg.kernel_parameter = 0.2
    # sampled points
    N_domain = (N_pts_per_dim-2)**2
    N_boundary = 4*(N_pts_per_dim-1)
    pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary)
    err_alpha_0_const_sigma.L2err[0,iter] = pts_L2_err 
    err_alpha_0_const_sigma.Maxerr[0,iter] = pts_max_err
    err_alpha_0_const_sigma.config.append(cfg)
    
print('\n Finished err_alpha_0_const_sigma')
print(f'Max err: {err_alpha_0_const_sigma.Maxerr}')
print(f'L2 err: {err_alpha_0_const_sigma.L2err}\n')

# for err_alpha_0_FD
for iter in range(err_alpha_0_const_sigma.arr_num):
    def u(x1, x2):
        return jnp.sin(jnp.pi*x1) * jnp.sin(jnp.pi*x2) + 2*jnp.sin(4*jnp.pi*x1) * jnp.sin(4*jnp.pi*x2)
    def f(x1, x2):
        return -grad(grad(u,0),0)(x1, x2)-grad(grad(u,1),1)(x1, x2)
    def a(x1, x2):
        return 1
    # get number of points in each dimension
    N_pts_per_dim = err_alpha_0_const_sigma.arr_N_pts_per_dim[iter]
    sol = FD_Darcy_flow_2d(N_pts_per_dim-2, a, f)
    sol = jnp.reshape(sol, (-1,1))
    xx= jnp.linspace(0,1,N_pts_per_dim)
    yy = jnp.linspace(0,1,N_pts_per_dim)
    XX, YY = onp.meshgrid(xx, yy)
    u_truth = vmap(u)(XX.reshape(-1,1), YY.reshape(-1,1))
    
    err_alpha_0_FD.L2err[0,iter] = max(sol-u_truth)
    err_alpha_0_FD.Maxerr[0,iter] = jnp.sqrt(jnp.sum((sol-u_truth)**2) / (N_pts_per_dim**2))

print('\n Finished err_alpha_0_FD')
print(f'Max err: {err_alpha_0_FD.Maxerr}')
print(f'L2 err: {err_alpha_0_FD.L2err}\n')

# figure:
# plot figures
here_fontsize=13
# --------------figure---------------------
fig = plt.figure(figsize=(12,5))

# plot the contour error
ax = fig.add_subplot(121)

ax.plot(err_alpha_0_const_sigma.arr_N_pts_per_dim**2, err_alpha_0_const_sigma.L2err[0,:], label='$L^2$, $\sigma = 0.2$')
ax.plot(err_alpha_0_const_sigma.arr_N_pts_per_dim**2, err_alpha_0_const_sigma.Maxerr[0,:], label='$L^\infty$, $\sigma = 0.2$')
ax.plot(err_alpha_0_adapt_sigma.arr_N_pts_per_dim**2, err_alpha_0_adapt_sigma.L2err[0,:], label='$L^2$, $\sigma = M^{-1/4}$')
ax.plot(err_alpha_0_adapt_sigma.arr_N_pts_per_dim**2, err_alpha_0_adapt_sigma.Maxerr[0,:], label='$L^\infty$, $\sigma = M^{-1/4}$')
ax.plot(err_alpha_0_FD.arr_N_pts_per_dim**2, err_alpha_0_FD.L2err[0,:], label='$L^2$ (FD)')
ax.plot(err_alpha_0_FD.arr_N_pts_per_dim**2, err_alpha_0_FD.Maxerr[0,:], label='$L^\infty$ (FD)')

plt.xlabel('$M$', fontsize=here_fontsize)
plt.ylabel('Error', fontsize=here_fontsize)
plt.yscale("log")
plt.xscale("log")
plt.title(r'$\tau (u) = 0$', fontsize=here_fontsize)
ax.legend(loc="upper right")

# plot the collocation pts
ax = fig.add_subplot(122)

ax.plot(err_alpha_1_const_sigma.arr_N_pts_per_dim**2, err_alpha_1_const_sigma.L2err[0,:], label='$L^2$, $\sigma = 0.2$')
ax.plot(err_alpha_1_const_sigma.arr_N_pts_per_dim**2, err_alpha_1_const_sigma.Maxerr[0,:], label='$L^\infty$, $\sigma = 0.2$')
ax.plot(err_alpha_1_adapt_sigma.arr_N_pts_per_dim**2, err_alpha_1_adapt_sigma.L2err[0,:], label='$L^2$, $\sigma = M^{-1/4}$')
ax.plot(err_alpha_1_adapt_sigma.arr_N_pts_per_dim**2, err_alpha_1_adapt_sigma.Maxerr[0,:], label='$L^\infty$, $\sigma = M^{-1/4}$')

plt.xlabel('$M$', fontsize=here_fontsize)
plt.ylabel('Error', fontsize=here_fontsize)
plt.yscale("log")
plt.xscale("log")
plt.title(r'$\tau (u) = u^3$', fontsize=here_fontsize)
ax.legend(loc="upper right")

plt.show()
fig.tight_layout()

onp.savez('data_elliptic_convergence_curve.npz', alpha_1_adapt_sigma=err_alpha_1_adapt_sigma, alpha_1_const_sigma=err_alpha_1_const_sigma, alpha_0_adapt_sigma=err_alpha_0_adapt_sigma, alpha_0_const_sigma=err_alpha_0_const_sigma, alpha_0_FD=err_alpha_0_FD)

fig.savefig('Elliptic_eqn_convergence_curve_Float64.pdf', bbox_inches='tight',dpi=100,pad_inches=0.1)