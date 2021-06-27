
# jax
import jax.numpy as jnp
from jax.config import config; 
config.update("jax_enable_x64", True)

import numpy as onp

# equation
from PDEs import Nonlinear_elliptic2d, Burgers, Eikonal
from InverseProblems import Darcy_flow2d

# visulization: plot figures
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# figure format; comment out them if errors appear
fsize = 15
tsize = 15
tdir = 'in'
major = 5.0
minor = 3.0
lwidth = 0.8
lhandle = 2.0
plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.linewidth'] = lwidth
plt.rcParams['legend.handlelength'] = lhandle

fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))

class solver_GP(object):
    def __init__(self,cfg = None, PDE_type = "Nonlinear_elliptic"):
        self.config = cfg
        self.PDE_type = PDE_type
        
    def set_equation(self, bdy = None, rhs = None, domain=onp.array([[0,1],[0,1]]),print_option = True):
        # generate the equation
        if self.PDE_type == "Nonlinear_elliptic":
            self.eqn = Nonlinear_elliptic2d(alpha = self.config.alpha, m = self.config.m, bdy = bdy, rhs = rhs, domain = domain)
            if print_option:
                print('\n Solver started')
                print('[Equation type] Nonlinear elliptic equation')
                print('[Equation form] - \Delta u + alpha*u^m = f')
                print(f'[Equation domain] [{domain[0,0]},{domain[0,1]}]*[{domain[1,0]},{domain[1,1]}]')
                print(f'[Equation parameter] alpha = {self.config.alpha}, m = {self.config.m}')
                print('[Equation data] Right hand side and boundary values set by the user')
        elif self.PDE_type == "Burgers":
            self.eqn = Burgers(alpha = self.config.alpha, nu = self.config.nu, bdy = bdy, rhs = rhs, domain = domain)
            if print_option:
                print('\n Solver started')
                print('[Equation type] Burgers equation')
                print('[Equation form] u_t+ alpha u u_x- nu u_xx=0')
                print(f'[Equation domain] [{domain[0,0]},{domain[0,1]}]*[{domain[1,0]},{domain[1,1]}]')
                print(f'[Equation parameter] alpha = {self.config.alpha}, m = {self.config.nu}')
                print('[Equation data] Right hand side and boundary values set by the user')
        elif self.PDE_type == "Eikonal":
            self.eqn = Eikonal(eps = self.config.eps, bdy = bdy, rhs = rhs, domain = domain)
            if print_option:
                print('\n Solver started')
                print('[Equation type] Eikonal equation')
                print('[Equation form] |grad u|^2 = f + eps*Delta u')
                print(f'[Equation domain] [{domain[0,0]},{domain[0,1]}]*[{domain[1,0]},{domain[1,1]}]')
                print(f'[Equation parameter] eps = {self.config.eps}')
                print('[Equation data] Right hand side and boundary values set by the user')
        elif self.PDE_type == "Darcy_flow2d":
            self.eqn = Darcy_flow2d(bdy = bdy, rhs = rhs, domain = domain)
            if print_option:
                print('\n Solver started')
                print('[Inverse problem type] Darcy flow 2d')
                print('[Inverse problem form] -div(a grad u) = f, infer a from f and some observed u')
                print(f'[Equation domain] [{domain[0,0]},{domain[0,1]}]*[{domain[1,0]},{domain[1,1]}]')
                print('[Equation data] Right hand side and boundary values set by the user')
                
    def get_sample(self, X_domain, X_boundary, print_option = True):
        # sampling points
        self.eqn.get_sampled_points(self, X_domain, X_boundary)
        if print_option:
            print('[Sample points] Collocation points sampled, specified by the user')
            print(f'[Sample points] N_domain = {self.eqn.N_domain}, N_boundary = {self.eqn.N_boundary}')
        
    def auto_sample(self, N_domain, N_boundary, sampled_type = 'random', print_option = True):
        self.eqn.sampled_pts(N_domain, N_boundary, sampled_type = sampled_type)
        if print_option:
            print(f'[Sample points] Collocation points sampled, type {sampled_type}')
            print(f'[Sample points] N_domain = {self.eqn.N_domain}, N_boundary = {self.eqn.N_boundary}')
            
    def show_sample(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        int_data=ax.scatter(self.eqn.X_domain[:, 0], self.eqn.X_domain[:, 1], marker="x", label='Interior nodes')
        bd_data=ax.scatter(self.eqn.X_boundary[:, 0], self.eqn.X_boundary[:, 1], marker="x", label='Boundary nodes')
        int_data.set_clip_on(False)
        bd_data.set_clip_on(False)
        ax.legend(loc="upper right")
        plt.title('Collocation points')
        
    # for inverse problems:
    def get_sample_IP(self, X_domain, X_boundary, X_data, print_option = True):
        # sampling points
        self.eqn.get_sampled_points(self, X_domain, X_boundary, X_data)
        if print_option:
            print('[Sample points] Collocation points sampled, specified by the user')
            print(f'[Sample points] N_domain = {self.eqn.N_domain}, N_boundary = {self.eqn.N_boundary}, N_data = {self.eqn.N_data}')
        
    def auto_sample_IP(self, N_domain, N_boundary, N_data, sampled_type = 'random', print_option = True):
        self.eqn.sampled_pts(N_domain, N_boundary, N_data, sampled_type = sampled_type)
        if print_option:
            print(f'[Sample points] Collocation points sampled, type {sampled_type}')
            print(f'[Sample points] N_domain = {self.eqn.N_domain}, N_boundary = {self.eqn.N_boundary}, N_data = {self.eqn.N_data}')
            
    def show_sample_IP(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        int_data=ax.scatter(self.eqn.X_domain[:, 0], self.eqn.X_domain[:, 1], label='Interior nodes')
        bd_data=ax.scatter(self.eqn.X_boundary[:, 0], self.eqn.X_boundary[:, 1], label='Boundary nodes')
        observed_data = ax.scatter(self.eqn.X_domain[:self.eqn.N_data, 0], self.eqn.X_domain[:self.eqn.N_data, 1], label='Data nodes')
        int_data.set_clip_on(False)
        bd_data.set_clip_on(False)
        observed_data.set_clip_on(False)
        ax.legend(loc="upper right")
        plt.title('Collocation and data points')
    
    def get_observed_data(self, data_u, noise_level, print_option = True):
        self.eqn.get_observation(data_u, noise_level)
        if print_option:
            print('[Observed Data] Get observed data from solving the PDE using FD and interpolation')
            print(f'[Observed Data] Noise level {noise_level}')
        
    def solve(self, print_option=True):
        if print_option:
            print('[Kernel] ' + self.config.kernel)
            print(f'[Kernel parameter]: {self.config.kernel_parameter}')
        # form Gram matrix
        self.eqn.Gram_matrix(kernel = self.config.kernel, kernel_parameter = self.config.kernel_parameter, nugget = self.config.nugget, nugget_type = self.config.nugget_type)
        if print_option:
            print(f'[Gram matrix] Finish assembly of the Gram matrix, nugget {self.config.nugget}, type {self.config.nugget_type}')
        # Cholesky
        self.eqn.Gram_Cholesky()
        if print_option:
            print('[Gram matrix] Finish Cholesky factorization of the Gram matrix')
        # GN algorithm
        if print_option:
            print('[Gauss Newton] Start Gauss Newton iteration')
        self.eqn.GN_method(max_iter = self.config.max_iter, step_size = self.config.step_size, initial_sol = self.config.initial_sol, print_hist = self.config.print_hist)
        if print_option:
            print('[Gauss Newton] Gauss Newton iteration finished')
            
    def show_loss_hist(self):
        fig = plt.figure()
        plt.plot(jnp.arange(self.eqn.max_iter+1),self.eqn.loss_hist)
        plt.yscale("log")
        plt.title('Loss function history')
        plt.xlabel('Gauss-Newton step')
    
    def collocation_pts_err(self, truth, print_option = True):
         # truth is the true value of the function on collocation points
        if print_option:
            print('[Calculating collocation errors...]')
        self.pts_err_all= abs(truth-self.eqn.sol_sampled_pts)
        self.pts_max_err = jnp.max(self.pts_err_all)
        self.pts_L2_err = jnp.sqrt(jnp.sum(self.pts_err_all**2) / (self.eqn.N_domain))
        if print_option:
            print(f'[Collocation point error] Max error {self.pts_max_err}')
            print(f'[Collocation point error] L2 error {self.pts_L2_err}')
        
    def test(self, X_test, print_option = True):
        if print_option:
            print(f'[Testing...] Number of test points: {X_test.shape[0]}')
        # GP test and extension
        self.eqn.extend_sol(X_test)
    
    def get_test_error(self, truth, print_option = True):
        # truth is the true value of the function on X_test
        self.test_err_all= abs(truth-self.eqn.extended_sol)
        self.test_max_err = jnp.max(self.test_err_all)
        self.test_L2_err = jnp.sqrt(jnp.sum(self.test_err_all**2) / (self.eqn.N_test))
        if print_option:
            print(f'[Test error] Max error {self.test_max_err}')
            print(f'[Test error] L2 error {self.test_L2_err}')
    
    def contour_of_test_err(self, XX, YY):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        err_contourf=ax.contourf(XX, YY, self.test_err_all.reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Contour of errors')
        fig.colorbar(err_contourf, format=fmt)
        plt.show()
        