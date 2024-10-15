# JAX
import jax.numpy as jnp
from jax import grad, vmap, hessian, jit
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)

# numpy
import numpy as onp
from numpy import random 

from .sample_points import sampled_pts_rdm, sampled_pts_grid
from .Gram_matrice import Gram_matrix_assembly, construct_Theta_test

class Darcy_flow2d(object):
    def __init__(self, bdy =None, rhs=None, domain=onp.array([[0,1],[0,1]])):
        self.bdy = bdy
        self.rhs = rhs
        self.domain = domain
      
    @partial(jit, static_argnums=(0,))  
    def get_bd(self, x1,x2):
        return self.bdy(x1,x2)
    
    @partial(jit, static_argnums=(0,))
    def get_rhs(self, x1,x2):
        return self.rhs(x1,x2)
    
    # sampling points according to random or grid rules
    # in the N_domain collocation points, the first N_data-th points are selected as the observed data points 
    def sampled_pts(self, N_domain, N_boundary, N_data, sampled_type = 'random'):
    # if rdm is true, sample points uniformly randomly, else in a uniform grid
        if sampled_type == 'random':
            X_domain, X_boundary = sampled_pts_rdm(N_domain, N_boundary, self.domain, time_dependent = False)
            X_data = X_domain[0:N_data,:]
        elif sampled_type == 'grid':
            X_domain, X_boundary = sampled_pts_grid(N_domain, N_boundary, self.domain, time_dependent = False)
            X_data = X_domain[0:N_data,:]
        self.X_domain = X_domain
        self.N_domain = X_domain.shape[0]
        self.X_boundary = X_boundary
        self.N_boundary = X_boundary.shape[0]
        self.X_data = X_data
        self.N_data = N_data
        self.rhs_f = vmap(self.get_rhs)(X_domain[:,0], X_domain[:,1])
        self.bdy_g = vmap(self.get_bd)(X_boundary[:,0], X_boundary[:,1])

    # directly given sampled points
    # without loss of generalization, we always assume that observed data points are included in the N_domain collocation points
    # and, the data points are always the first N_data ones of X_domain
    def get_sampled_points(self, X_domain, X_boundary, X_data):
        self.X_domain = X_domain
        self.N_domain = X_domain.shape[0]
        self.X_boundary = X_boundary
        self.N_boundary = X_boundary.shape[0]
        self.X_data = X_data
        self.N_data = X_data.shape[0]
        self.rhs_f = vmap(self.get_rhs)(X_domain[:,0], X_domain[:,1])
        self.bdy_g = vmap(self.get_bd)(X_boundary[:,0], X_boundary[:,1])
        
    def get_observation(self, data_u, noise_level):
        self.data_u = data_u + noise_level*random.normal(0, 1.0, onp.shape(data_u)[0])
        self.noise_level = noise_level
        
    def Gram_matrix(self, kernel = 'Gaussian', kernel_parameter = 0.2, nugget = 1e-10, nugget_type = 'adaptive'):
        Theta_u, Theta_a = Gram_matrix_assembly(self.X_domain, self.X_boundary, eqn = 'Darcy_flow2d', kernel = kernel, kernel_parameter = kernel_parameter)
        self.nugget_type = nugget_type
        self.nugget = nugget
        self.kernel = kernel
        self.kernel_parameter = kernel_parameter
        if nugget_type == 'adaptive':
            # Theta_u
            trace1_u = onp.trace(Theta_u[:self.N_domain, :self.N_domain])
            trace2_u = onp.trace(Theta_u[self.N_domain:2*self.N_domain, self.N_domain:2*self.N_domain])
            trace3_u = onp.trace(Theta_u[2*self.N_domain:3*self.N_domain, 2*self.N_domain:3*self.N_domain])
            trace4_u = onp.trace(Theta_u[3*self.N_domain:, 3*self.N_domain:])
            ratio_u = [trace1_u/trace4_u, trace2_u/trace4_u, trace3_u/trace4_u]
            
            # calculate trace
            # Theta_a
            trace1_a = onp.trace(Theta_a[:self.N_domain, :self.N_domain])
            trace2_a = onp.trace(Theta_a[self.N_domain:2*self.N_domain, self.N_domain:2*self.N_domain])
            trace3_a = onp.trace(Theta_a[2*self.N_domain:3*self.N_domain, 2*self.N_domain:3*self.N_domain])
            ratio_a = [trace1_a/trace3_a, trace2_a/trace3_a]
            
            temp=onp.concatenate((ratio_u[0]*onp.ones((1,self.N_domain)), ratio_u[1]*onp.ones((1,self.N_domain)), ratio_u[2]*onp.ones((1,self.N_domain)), onp.ones((1,self.N_domain+self.N_boundary))), axis=1)
            self.Theta_u = Theta_u + nugget*onp.diag(temp[0])
            
            temp=onp.concatenate((ratio_a[0]*onp.ones((1,self.N_domain)), ratio_a[1]*onp.ones((1,self.N_domain)), onp.ones((1,self.N_domain))), axis=1)
            self.Theta_a = Theta_a + nugget*onp.diag(temp[0])
            
        elif nugget_type == 'identity':
            self.Theta_u = Theta_u + nugget*jnp.eye(Theta_u.shape[0])
            self.Theta_a = Theta_a + nugget*jnp.eye(Theta_a.shape[0])
            
        elif nugget_type == 'none':
            self.Theta_u = Theta_u
            self.Theta_a = Theta_a
    
    def Gram_Cholesky(self):
        self.L_u = jnp.linalg.cholesky(self.Theta_u)
        self.L_a = jnp.linalg.cholesky(self.Theta_a)
    
    @partial(jit, static_argnums=(0,))
    def loss(self,z):
        w0 = z[0:self.N_domain]
        w1 = z[self.N_domain:2*self.N_domain]
        w2 = z[2*self.N_domain:3*self.N_domain]
        
        v0 = z[3*self.N_domain:4*self.N_domain]
        v1 = z[4*self.N_domain:5*self.N_domain]
        v2 = z[5*self.N_domain:6*self.N_domain]
        v3 = -v1*w1-v2*w2+(-self.rhs_f)*jnp.exp(-w0)
        
        w_all = jnp.concatenate((w1,w2,w0), axis=0)
        v_all = jnp.concatenate((v1,v2,v3,v0,self.bdy_g), axis=0)
        temp_a = jnp.linalg.solve(self.L_a,w_all)
        temp_u = jnp.linalg.solve(self.L_u,v_all)
        return jnp.dot(temp_a, temp_a) + jnp.dot(temp_u, temp_u) + (1/self.noise_level**2)*jnp.sum((v0[:self.N_data]-self.data_u)**2)
    
    @partial(jit, static_argnums=(0,))
    def grad_loss(self, z):
        return grad(self.loss)(z)
    
    @partial(jit, static_argnums=(0,))
    def GN_loss(self, z, z_old):
        w0_old = z_old[0:self.N_domain]
        w1_old = z_old[self.N_domain:2*self.N_domain]
        w2_old = z_old[2*self.N_domain:3*self.N_domain] 
        v1_old = z_old[4*self.N_domain:5*self.N_domain]
        v2_old = z_old[5*self.N_domain:6*self.N_domain] 

        w0 = z[0:self.N_domain]
        w1 = z[self.N_domain:2*self.N_domain]
        w2 = z[2*self.N_domain:3*self.N_domain]
        v0 = z[3*self.N_domain:4*self.N_domain]
        v1 = z[4*self.N_domain:5*self.N_domain]
        v2 = z[5*self.N_domain:6*self.N_domain]
        v3 = (-self.rhs_f)*(-jnp.exp(-w0_old))*w0 + (-v1_old)*w1 + (-v2_old)*w2 + (-w1_old)*v1 + (-w2_old)*v2
        
        w_all = jnp.concatenate((w1,w2,w0), axis=0)
        v_all = jnp.concatenate((v1,v2,v3,v0,self.bdy_g), axis=0)
        
        temp_a = jnp.linalg.solve(self.L_a,w_all)
        temp_u = jnp.linalg.solve(self.L_u,v_all)
        return jnp.dot(temp_a, temp_a) + jnp.dot(temp_u, temp_u) + (1/self.noise_level**2)*jnp.sum((v0[:self.N_data]-self.data_u)**2)
    
    @partial(jit, static_argnums=(0,))
    def Hessian_GN(self,z,z_old):
        return hessian(self.GN_loss)(z,z_old)
    
    def GN_method(self, max_iter = 3, step_size = 1, initial_sol = 'rdm', print_hist = True):
        if initial_sol == 'rdm':
            sol = random.normal(0.0, 1.0, (6*self.N_domain))
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
                print('iter = ', iter_step, 'Gauss-Newton step size =', step_size, ' Loss = ', loss_now) 
        self.max_iter = max_iter
        self.step_size = step_size
        self.loss_hist = loss_hist
        
        w = sol[self.N_domain:3*self.N_domain]
        self.sol_vec_a = onp.append(w, sol[:self.N_domain])
        
        w0 = sol[0:self.N_domain]
        w1 = sol[self.N_domain:2*self.N_domain]
        w2 = sol[2*self.N_domain:3*self.N_domain]
        v0 = sol[3*self.N_domain:4*self.N_domain]
        v1 = sol[4*self.N_domain:5*self.N_domain]
        v2 = sol[5*self.N_domain:6*self.N_domain]
        v3 = -v1*w1-v2*w2+(-self.rhs_f)*jnp.exp(-w0)
        self.sol_vec_u = jnp.concatenate((v1,v2,v3,v0,self.bdy_g), axis=0)

    def extend_sol(self,X_test):
        Theta_u_test, Theta_a_test = construct_Theta_test(X_test, self.X_domain, self.X_boundary, eqn = 'Darcy_flow2d', kernel = self.kernel, kernel_parameter = self.kernel_parameter)
        temp = jnp.linalg.solve(jnp.transpose(self.L_a),jnp.linalg.solve(self.L_a,self.sol_vec_a))
        self.X_test = X_test
        self.N_test = X_test.shape[0]
        self.extended_sol_a = jnp.matmul(Theta_a_test,temp)
        
        temp = jnp.linalg.solve(jnp.transpose(self.L_u),jnp.linalg.solve(self.L_u,self.sol_vec_u))
        self.extended_sol_u = jnp.matmul(Theta_u_test,temp)


                                       