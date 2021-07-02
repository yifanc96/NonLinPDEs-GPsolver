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

class error_test_nugget(object):
    def __init__(self, arr_N_domain, arr_N_boundary, array_nugget, num_random=10):
        self.arr_N_domain = arr_N_domain
        self.arr_N_boundary = arr_N_boundary
        self.num_random = num_random
        self.num_nugget = onp.shape(array_nugget)[0]
        self.L2err = onp.zeros((num_random,self.num_nugget))
        self.Maxerr = onp.zeros((num_random,self.num_nugget))

arr_N_domain = 900
arr_N_boundary = 124
arr_nugget = 1 / 10 ** onp.arange(1,14,1)
error = error_test_nugget(arr_N_domain, arr_N_boundary, arr_nugget, num_random = 10)

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
    solver.auto_sample(N_domain, N_boundary, sampled_type = 'random')
    solver.solve()
    pts_truth = vmap(u)(solver.eqn.X_domain[:,0],solver.eqn.X_domain[:,1])
    solver.collocation_pts_err(pts_truth)
    return solver.pts_max_err, solver.pts_L2_err

for iter in range(onp.shape(arr_nugget)[0]):
    cfg = copy.copy(cfg_default)
    cfg.nugget = arr_nugget[iter]
    # sampled points
    N_domain = arr_N_domain
    N_boundary = arr_N_boundary
    for iter_rdm in range(error.num_random):
        pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary)
        error.L2err[iter_rdm,iter] = pts_L2_err
        error.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{error.num_random} finished \n ')
        
error_nonadaptive = error_test_nugget(arr_N_domain, arr_N_boundary, arr_nugget, num_random = 10)
for iter in range(onp.shape(arr_nugget)[0]):
    cfg = copy.copy(cfg_default)
    cfg.nugget = arr_nugget[iter]
    cfg.nugget_type = 'identity'
    # sampled points
    N_domain = arr_N_domain
    N_boundary = arr_N_boundary
    for iter_rdm in range(error_nonadaptive.num_random):
        pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary)
        error_nonadaptive.L2err[iter_rdm,iter] = pts_L2_err
        error_nonadaptive.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{error_nonadaptive.num_random} finished \n ')

import pickle
with open('data_test_nugget.pkl', 'wb') as file_name:
    pickle.dump(error, file_name)
    pickle.dump(error_nonadaptive, file_name)