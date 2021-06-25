# JAX
import jax.numpy as jnp
from jax import grad, vmap

# numpy
import numpy as onp
from numpy import random 

from Sample_points import sampled_pts_rdm, sampled_pts_grid
from Gram_matrice import Gram_matrix_assembly

class Nonlinear_elliptic2d(object):
    def __init__(self, alpha, m, bdy, rhs):
        # -Delta u + alpha*u^m = f in [0,1]^2
        self.alpha = alpha
        self.m = m
        self.bdy = bdy
        self.rhs = rhs
        
    def get_bd(self, x):
        return self.bdy(x)
    
    def get_rhs(self, x):
        return self.rhs(x)
    
    def sampled_pts(self, N_domain, N_boundary, rdm = True):
    # if rdm is true, sample points uniformly randomly, else in a uniform grid
        if rdm:
            X_domain, X_boundary = sampled_pts_rdm(N_domain, N_boundary, time_dependent = False)
        else:
            assert N_boundary == 4*(onp.sqrt(N_domain)+1)
            X_domain, X_boundary = sampled_pts_grid(N_domain, N_boundary, time_dependent = False)
        self.X_domain = X_domain
        self.N_domain = N_domain
        self.X_boundary = X_boundary
        self.N_boundary = N_boundary
        self.rhs_f = vmap(self.get_rhs)(X_domain[:,0], X_domain[:,1])
        self.bdy_g = vmap(self.get_bd)(X_boundary[:,0], X_boundary[:,1])
    
    def Gram_matrix(self, kernel = 'Gaussian', kernel_parameter = 0.2, nugget = 1e-15, nugget_type = 'adaptive'):
        Theta = Gram_matrix_assembly(self.X_domain, self.X_boundary, eqn = 'Nonlinear_elliptic', kernel = kernel, parameter = kernel_parameter)
        self.nugget_type = nugget_type
        self.nugget = nugget
        self.kernel = kernel
        self.kernel_parameter = kernel_parameter
        if nugget_type == 'adaptive':
            # calculate trace
            trace1 = jnp.trace(Theta[:self.N_domain, :self.N_domain])
            trace2 = jnp.trace(Theta[self.N_domain:, self.N_domain:])
            ratio = trace1/trace2
            self.ratio = ratio
            temp=jnp.concatenate((ratio*jnp.ones((1,self.N_domain)),jnp.ones((1,self.N_domain+self.N_boundary))), axis=1)
            self.Theta = Theta + nugget*jnp.diag(temp[0])
        elif nugget_type == 'identity':
            self.Theta = Theta + nugget*jnp.eye(2*self.N_domain+self.N_boundary)
        elif nugget_type == 'none':
            self.Theta = Theta
    
    def Gram_Cholesky(self):
        self.L = jnp.linalg.cholesky(self.Theta)
    
    def loss(self, z):
        zz = jnp.append(self.alpha*(z**self.m) - self.rhs_f, z) 
        zz = jnp.append(zz, self.bdy_g)
        zz = jnp.linalg.solve(self.L, zz)
        return jnp.dot(zz, zz)
    
    def grad_loss(self, z):
        return grad(self.loss)(z)
    
    def Hessian_GN(self,z,z_old):
        zz = jnp.append(self.alpha*self.m*(z_old**(self.m-1))*(z-z_old), z) 
        zz = jnp.append(zz, self.bdy_g)
        zz = jnp.linalg.solve(self.L, zz)
        return jnp.dot(zz, zz)
    
    def GN_method(self, max_iter = 3, step_size = 1, initial_sol = 'rdm', print_hist = True):
        if initial_sol == 'rdm':
            sol = random.normal(0.0, 1.0, (self.N_domain))
        self.init_sol = sol
        loss_hist = [] # history of loss function values
        loss_now = self.loss(sol)
        loss_hist.append(loss_now)
        
        if print_hist:
            print('iter = 0', 'Loss =', loss_now) # print out history
        
        for iter_step in range(1, max_iter+1):
            temp = jnp.linalg.solve(self.Hessian_GN(sol,sol), self.grad_loss(sol))
            sol = sol - step_size*temp  
            loss_now = self.loss(sol)
            loss_hist.append(loss_now)
            if print_hist:
                # print out history
                print('iter = ', iter_step, 'Gauss-Newton step size =', step_size, ' J = ', loss_now) 
        self.max_iter = max_iter
        self.step_size = step_size
        self.loss_hist = loss_hist
        self.sol_vec = sol
        
        
    
        
        