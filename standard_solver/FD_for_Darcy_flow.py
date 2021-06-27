import numpy as onp
from jax import vmap
# Scipy
import scipy.sparse
from scipy.sparse import diags


def FD_Darcy_flow_2d(N, fun_a, f):
    hg = 1/(N+1)
    x_mid = (onp.arange(0,N+1,1)+0.5)*hg
    x_grid = (onp.arange(1,N+1,1))*hg
    mid, grid = onp.meshgrid(x_mid, x_grid)
    a1 = onp.reshape(vmap(fun_a)(mid.flatten(), grid.flatten()), (N, N+1))
    a2 = onp.transpose(onp.reshape(vmap(fun_a)(grid.flatten(), mid.flatten()), (N,N+1)))

    # diagonal element of A
    a_diag = onp.reshape(a1[:,:N]+a1[:,1:]+a2[:N,:]+a2[1:,:], (1,-1))
    
    # off-diagonals
    a_super1 = onp.reshape(onp.append(a1[:,1:N], onp.zeros((N,1)), axis = 1), (1,-1))
    a_super2 = onp.reshape(a2[1:N,:], (1,-1))
    
    A = diags([[-a_super2[onp.newaxis, :]], [-a_super1[onp.newaxis, :]], [a_diag], [-a_super1[onp.newaxis, :]], [-a_super2[onp.newaxis, :]]], [-N,-1,0,1,N], shape=(N**2, N**2))/(hg**2)
    
    XX, YY = onp.meshgrid(x_grid, x_grid)
    fv = vmap(f)(XX.flatten(), YY.flatten())
    fv = fv[:, onp.newaxis]
    sol_u = scipy.sparse.linalg.spsolve(A, fv)
    sol_u = onp.reshape(sol_u, (N, N))
    sol_u_plus_bd = onp.zeros((N+2,N+2))
    sol_u_plus_bd[1:N+1,1:N+1]=sol_u
    
    return sol_u_plus_bd