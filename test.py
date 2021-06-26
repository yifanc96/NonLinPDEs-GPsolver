from PDEs import *
import jax.numpy as jnp
from jax import grad
from Sample_points import sampled_pts_grid

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
eqn.Gram_matrix(kernel = 'Gaussian', kernel_parameter = 0.2, nugget = 1e-4, nugget_type = 'adaptive')
eqn.Gram_Cholesky()
eqn.GN_method(max_iter = 3, step_size = 1, initial_sol = 'rdm', print_hist = True)

N_pts = 40
N_domain = N_pts**2
N_boundary = 4*(N_pts+1)
X_test = jnp.concatenate(sampled_pts_grid(N_domain, N_boundary), axis=0)
eqn.extend_sol(X_test)

