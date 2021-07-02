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

class error_curve(object):
    def __init__(self, arr_N_domain, arr_N_boundary,num_random = 1):
        self.arr_N_domain = arr_N_domain
        self.arr_num = onp.shape(arr_N_domain)[0]
        self.arr_N_boundary = arr_N_boundary
        self.num_random = num_random
        self.L2err = onp.zeros((num_random,self.arr_num))
        self.Maxerr = onp.zeros((num_random,self.arr_num))
        self.config = []
        
arr_N_domain = [300-30,600-60,1200-120,2400-240]
arr_N_boundary = [30,60,180,120,240]

print('\n [Goal] error curve for nonlinear elliptic equations')
print(f'[Setting] random sampling, array of N_domain {arr_N_domain}, array of N_boundary {arr_N_boundary}')

err_relaxation = error_curve(arr_N_domain, arr_N_boundary,num_random = 10)
error_elimination = error_curve(arr_N_domain, arr_N_boundary,num_random = 10)

cfg_default =munch.munchify({
    # basic set-up for equations
    'alpha': 1,
    'm': 3,
    # kernel selection
    'kernel': 'Gaussian', 
    'kernel_parameter': 0.2,
    'nugget': 1e-12,
    'nugget_type': 'adaptive',
    # optimiation
    'max_iter': 10, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

def get_collocation_pts_err(cfg, N_domain, N_boundary, method = 'relaxed', pen_lambda = 1e-10):
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
    solver.solve(method = method, pen_lambda = pen_lambda)
    pts_truth = vmap(u)(solver.eqn.X_domain[:,0],solver.eqn.X_domain[:,1])
    solver.collocation_pts_err(pts_truth)
    return solver.pts_max_err, solver.pts_L2_err


for iter in range(err_relaxation.arr_num):
    # get number of points in each dimension
    N_domain = err_relaxation.arr_N_domain[iter]
    N_boundary = err_relaxation.arr_N_boundary[iter]
    # get the configuration
    cfg = copy.copy(cfg_default)
    cfg.nugget = 1e-12
    err_relaxation.config.append(cfg)
    for iter_rdm in range(err_relaxation.num_random):
        pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary, method = 'relaxation', pen_lambda = 1e-12)
        err_relaxation.L2err[iter_rdm, iter] = pts_L2_err 
        err_relaxation.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{err_relaxation.num_random} finished \n ')

for iter in range(error_elimination.arr_num):
    # get number of points in each dimension
    N_domain = error_elimination.arr_N_domain[iter]
    N_boundary = error_elimination.arr_N_boundary[iter]
    # get the configuration
    cfg = copy.copy(cfg_default)
    cfg.nugget = 1e-12
    error_elimination.config.append(cfg)
    for iter_rdm in range(error_elimination.num_random):
        pts_max_err, pts_L2_err = get_collocation_pts_err(cfg, N_domain, N_boundary, method = 'elimination')
        error_elimination.L2err[iter_rdm, iter] = pts_L2_err 
        error_elimination.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{error_elimination.num_random} finished \n ')

import pickle
with open('data_compare_elimination_relaxation.pkl', 'wb') as file_name:
    pickle.dump(err_relaxation, file_name)
    pickle.dump(error_elimination, file_name)