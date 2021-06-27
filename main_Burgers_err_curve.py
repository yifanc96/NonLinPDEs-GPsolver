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

# visulization: plot figures
import matplotlib.pyplot as plt

class error_curve(object):
    def __init__(self, arr_N_domain, arr_N_boundary,num_random = 1):
        self.arr_N_domain = arr_N_domain
        self.arr_num = onp.shape(arr_N_domain)[0]
        self.arr_N_boundary = arr_N_boundary
        self.num_random = num_random
        self.L2err = onp.zeros((num_random,self.arr_num))
        self.Maxerr = onp.zeros((num_random,self.arr_num))
        self.config = []

arr_N_domain = [960,1440,1920,2400]
arr_N_boundary = [240,360,480,600]

print('\n [Goal] error curve for Burgers equations')
print(f'[Setting] random sampling, array of N_domain {arr_N_domain}, array of N_boundary {arr_N_boundary}')
print('[Setting] alpha = 1, nu = 0.02')
err_Burgers_nugget1e_5 = error_curve(arr_N_domain, arr_N_boundary, num_random = 50)
err_Burgers_nugget1e_10 = error_curve(arr_N_domain, arr_N_boundary, num_random = 50)

# solving Burgers: u_t+ alpha u u_x- nu u_xx=0
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
    'max_iter': 16, 
    'step_size': 1,
    'initial_sol': 'rdm', 
    'print_hist' : True,  # print training loss history
})

# True solution:
nu = cfg_Burgers.nu
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

# Equation:
def u(x1, x2):
    return -jnp.sin(jnp.pi*x2)*(x1==0) + 0*(x2==0)
def f(x1, x2):
    return 0

def get_test_err(cfg, N_domain, N_boundary):
    # grid sampling points
    solver = solver_GP(cfg, PDE_type = "Burgers")
    solver.set_equation(bdy = u, rhs = f, domain=onp.array([[0,1],[-1,1]]))
    solver.auto_sample(N_domain, N_boundary, sampled_type = 'random')
    solver.solve()
    solver.test(X_test)
    solver.get_test_error(test_truth)
    return solver.test_max_err, solver.test_L2_err

# nugget 1e-5
for iter in range(err_Burgers_nugget1e_5.arr_num):
    # get number of points in each dimension
    N_domain = err_Burgers_nugget1e_5.arr_N_domain[iter]
    N_boundary = err_Burgers_nugget1e_5.arr_N_boundary[iter]
    # get the configuration
    cfg = copy.copy(cfg_Burgers)
    cfg.nugget = 1e-5
    err_Burgers_nugget1e_5.config.append(cfg)
    for iter_rdm in range(err_Burgers_nugget1e_5.num_random):
        # sampled points
        pts_max_err, pts_L2_err = get_test_err(cfg, N_domain, N_boundary)
        err_Burgers_nugget1e_5.L2err[iter_rdm, iter] = pts_L2_err 
        err_Burgers_nugget1e_5.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{err_Burgers_nugget1e_5.num_random} finished \n ')
        
print('\n Finished nugget 1e-5\n')

# nugget 1e-10
for iter in range(err_Burgers_nugget1e_10.arr_num):
    # get number of points in each dimension
    N_domain = err_Burgers_nugget1e_10.arr_N_domain[iter]
    N_boundary = err_Burgers_nugget1e_10.arr_N_boundary[iter]
    # get the configuration
    cfg = copy.copy(cfg_Burgers)
    cfg.nugget = 1e-10
    err_Burgers_nugget1e_10.config.append(cfg)
    for iter_rdm in range(err_Burgers_nugget1e_10.num_random):
        # sampled points
        pts_max_err, pts_L2_err = get_test_err(cfg, N_domain, N_boundary)
        err_Burgers_nugget1e_10.L2err[iter_rdm, iter] = pts_L2_err 
        err_Burgers_nugget1e_10.Maxerr[iter_rdm,iter] = pts_max_err
        print(f'\n random trial: {iter_rdm}/{err_Burgers_nugget1e_5.num_random} finished \n ')
# save
onp.savez('data_Burgers_convergence_curve.npz', err_Burgers_nugget1e_5 = err_Burgers_nugget1e_5, err_Burgers_nugget1e_10=err_Burgers_nugget1e_10)
