from PDEs import *
import jax.numpy as jnp
from jax import grad
from jax.config import config; 
config.update("jax_enable_x64", True)

alpha = 1
m = 3
def f(x1, x2):
    return -grad(grad(u,0),0)(x1, x2)-grad(grad(u,1),1)(x1, x2)+alpha*(u(x1, x2)**m)
def u(x1, x2):
    return jnp.sin(jnp.pi*x1) * jnp.sin(jnp.pi*x2) + 2*jnp.sin(4*jnp.pi*x1) * jnp.sin(4*jnp.pi*x2)

eqn = Nonlinear_elliptic2d(alpha = alpha, m = m, bdy = u, rhs = f)

N_domain = 30**2
N_boundary = 32**2-N_domain

eqn.sampled_pts(N_domain, N_boundary, rdm = True)
eqn.Gram_matrix(kernel = 'Gaussian', kernel_parameter = 0.2, nugget = 1e-10, nugget_type = 'adaptive')
eqn.Gram_Cholesky()
eqn.GN_method(max_iter = 4, step_size = 1, initial_sol = 'rdm', print_hist = True)

N_pts = 40
xx= jnp.linspace(0, 1, N_pts)
yy = jnp.linspace(0, 1, N_pts)
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)
eqn.extend_sol(X_test)

