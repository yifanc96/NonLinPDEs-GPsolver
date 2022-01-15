import numpy as onp
# Scipy
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import identity

def solve_Eikonal(N, epsilon):
    hg = onp.array(1/(N+1))
    x_grid = (onp.arange(1,N+1,1))*hg
    a1 = onp.ones((N,N+1))
    a2 = onp.ones((N+1,N))

    # diagonal element of A
    a_diag = onp.reshape(a1[:,:N]+a1[:,1:]+a2[:N,:]+a2[1:,:], (1,-1))
    
    # off-diagonals
    a_super1 = onp.reshape(onp.append(a1[:,1:N], onp.zeros((N,1)), axis = 1), (1,-1))
    a_super2 = onp.reshape(a2[1:N,:], (1,-1))
    
    A = diags([[-a_super2[onp.newaxis, :]], [-a_super1[onp.newaxis, :]], [a_diag], [-a_super1[onp.newaxis, :]], [-a_super2[onp.newaxis, :]]], [-N,-1,0,1,N], shape=(N**2, N**2), format = 'csr')
    XX, YY = onp.meshgrid(x_grid, x_grid)
    f = onp.zeros((N,N))
    f[0,:] = f[0,:] + epsilon**2 / (hg**2)
    f[N-1,:] = f[N-1,:] + epsilon**2 / (hg**2)
    f[:, 0] = f[:, 0] + epsilon**2 / (hg**2)
    f[:, N-1] = f[:, N-1] + epsilon**2 / (hg**2)
    fv = f.flatten()
    fv = fv[:, onp.newaxis]
    
    mtx = identity(N**2)+(epsilon**2)*A/(hg**2)
    sol_v = scipy.sparse.linalg.spsolve(mtx, fv)
    # sol_v, exitCode = scipy.sparse.linalg.cg(mtx, fv)
    # print(exitCode)
    sol_u = -epsilon*onp.log(sol_v)
    sol_u = onp.reshape(sol_u, (N,N))
    return XX, YY, sol_u