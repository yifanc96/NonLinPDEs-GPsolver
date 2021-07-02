#%%
# jax
import jax.numpy as jnp
from jax import grad, vmap
from jax.config import config; 
config.update("jax_enable_x64", True)
import numpy as onp
# solver
from solver import solver_GP
from standard_solver.Cole_Hopf_for_Eikonal import solve_Eikonal
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

print('\n [Goal] error curve for Eikonal equations')
print(f'[Setting] random sampling, array of N_domain {arr_N_domain}, array of N_boundary {arr_N_boundary}')
print('[Setting] eps=1e-1')
err_Eikonal_nugget1e_5 = error_curve(arr_N_domain, arr_N_boundary, num_random = 10)
err_Eikonal_nugget1e_10 = error_curve(arr_N_domain, arr_N_boundary, num_random = 10)

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
    'max_iter': 20, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

# True solution:
N_pts = 1999
XX, YY, fine_scale_truth = solve_Eikonal(N_pts, cfg_Eikonal.eps)
test_truth = fine_scale_truth[19:-1:20, 19:-1:20]
N_test = 99
hg = 1/(N_test+1)
x_grid = (onp.arange(1,N_test+1,1))*hg
XX, YY = onp.meshgrid(x_grid, x_grid)
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)
assert X_test.shape[0] == test_truth.shape[0] * test_truth.shape[1]
# Equation:
def u(x1, x2):
    return 0
def f(x1, x2):
    return 1

def get_test_err(cfg, N_domain, N_boundary):
    # grid sampling points
    solver = solver_GP(cfg, PDE_type = "Eikonal")
    solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[0,1]]))
    solver.auto_sample(N_domain, N_boundary, sampled_type = 'random')
    solver.solve()
    solver.test(X_test)
    solver.get_test_error(test_truth.flatten())
    return solver.test_max_err, solver.test_L2_err

# nugget 1e-5
for iter in range(err_Eikonal_nugget1e_5.arr_num):
    # get number of points in each dimension
    N_domain = err_Eikonal_nugget1e_5.arr_N_domain[iter]
    N_boundary = err_Eikonal_nugget1e_5.arr_N_boundary[iter]
    # get the configuration
    cfg = copy.copy(cfg_Eikonal)
    cfg.nugget = 1e-5
    # cfg.kernel_parameter = 1/jnp.sqrt(jnp.sqrt(N_domain+N_boundary))
    cfg.kernel_parameter = 0.2
    err_Eikonal_nugget1e_5.config.append(cfg)
    for iter_rdm in range(err_Eikonal_nugget1e_5.num_random):
        # sampled points
        pts_max_err, pts_L2_err = get_test_err(cfg, N_domain, N_boundary)
        err_Eikonal_nugget1e_5.L2err[iter_rdm, iter] = pts_L2_err 
        err_Eikonal_nugget1e_5.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{err_Eikonal_nugget1e_5.num_random} finished \n ')
        
print('\n Finished nugget 1e-5\n')

# nugget 1e-10
for iter in range(err_Eikonal_nugget1e_10.arr_num):
    # get number of points in each dimension
    N_domain = err_Eikonal_nugget1e_10.arr_N_domain[iter]
    N_boundary = err_Eikonal_nugget1e_10.arr_N_boundary[iter]
    # get the configuration
    cfg = copy.copy(cfg_Eikonal)
    cfg.kernel_parameter = 1/jnp.sqrt(jnp.sqrt(N_domain+N_boundary))
    err_Eikonal_nugget1e_10.config.append(cfg)
    for iter_rdm in range(err_Eikonal_nugget1e_10.num_random):
        # sampled points
        pts_max_err, pts_L2_err = get_test_err(cfg, N_domain, N_boundary)
        err_Eikonal_nugget1e_10.L2err[iter_rdm, iter] = pts_L2_err 
        err_Eikonal_nugget1e_10.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{err_Eikonal_nugget1e_5.num_random} finished \n ')
# save

import pickle
with open('data_Eikonal_convergence_curve.pkl', 'wb') as file_name:
    pickle.dump(err_Eikonal_nugget1e_5, file_name)
    pickle.dump(err_Eikonal_nugget1e_10, file_name)
