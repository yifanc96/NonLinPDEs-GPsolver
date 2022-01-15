# JAX
import jax.numpy as jnp
from jax import vmap
from jax.config import config; 
import jax.ops as jop
config.update("jax_enable_x64", True)

# numpy
import numpy as onp
from .kernels import Gaussian_kernel, Anisotropic_Gaussian_kernel

def Gram_matrix_assembly(X_domain, X_boundary, eqn = 'Nonlinear_elliptic', kernel = 'Gaussian', kernel_parameter = 0.2):
    N_domain = X_domain.shape[0]
    N_boundary = X_boundary.shape[0]
    
    # introduce auxiliary values that are used to compute block interactions in the Gram matrix
    Xd0=X_domain[:N_domain, 0]
    Xd1=X_domain[:N_domain, 1]

    Xdb0=jnp.concatenate([Xd0, X_boundary[:N_boundary, 0]])
    Xdb1=jnp.concatenate([Xd1, X_boundary[:N_boundary, 1]])

    # interior -- interior interactions
    XXdd0=jnp.transpose(jnp.tile(Xd0,(N_domain,1)))
    XXdd1=jnp.transpose(jnp.tile(Xd1,(N_domain,1)))

    # interior+boudary -- interior+bpundary interactions
    XXdbdb0=jnp.transpose(jnp.tile(Xdb0,(N_domain+N_boundary,1)))
    XXdbdb1=jnp.transpose(jnp.tile(Xdb1,(N_domain+N_boundary,1)))

    # interior v.s. interior+boundary interactions
    XXddb0=jnp.transpose(jnp.tile(Xd0,(N_domain+N_boundary,1)))
    XXddb1=jnp.transpose(jnp.tile(Xd1,(N_domain+N_boundary,1)))
    XXddb0_2=jnp.tile(Xdb0,(N_domain,1))
    XXddb1_2=jnp.tile(Xdb1,(N_domain,1))
    
    if kernel == 'Gaussian':
        K = Gaussian_kernel()
    elif kernel == 'anisotropic_Gaussian':
        K = Anisotropic_Gaussian_kernel()
            
    if eqn == 'Nonlinear_elliptic':
        Theta = onp.zeros((2*N_domain + N_boundary, 2*N_domain + N_boundary))
        # Construct kernel matrix
        # interior v.s. interior (Laplace)
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_Delta_y_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta[0:N_domain, 0:N_domain] =  onp.reshape(val, (N_domain, N_domain))
        
        # interior+boundary v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXdbdb0.flatten(),XXdbdb1.flatten(),onp.transpose(XXdbdb0).flatten(),onp.transpose(XXdbdb1).flatten())
        Theta[N_domain:, N_domain:] =  onp.reshape(val, (N_domain+N_boundary, N_domain+N_boundary))

        # interior v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta[:N_domain, N_domain:] = onp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta[N_domain:, :N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain+N_boundary)))
        return Theta
    
    elif eqn == 'Burgers':
        # Construct kernel matrix
        Theta = jnp.zeros((4*N_domain + N_boundary, 4*N_domain + N_boundary))
        # interior v.s. interior 
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta = jop.index_update(Theta, jop.index[0:N_domain, 0:N_domain], jnp.reshape(val, (N_domain, N_domain)))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta = jop.index_update(Theta, jop.index[0:N_domain, N_domain:2*N_domain], jnp.reshape(val, (N_domain, N_domain)))
        Theta = jop.index_update(Theta, jop.index[N_domain:2*N_domain, 0:N_domain], jnp.transpose(jnp.reshape(val, (N_domain, N_domain))))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_DD_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta = jop.index_update(Theta, jop.index[0:N_domain, 2*N_domain:3*N_domain], jnp.reshape(val, (N_domain, N_domain)))
        Theta = jop.index_update(Theta, jop.index[2*N_domain:3*N_domain, 0:N_domain], jnp.transpose(jnp.reshape(val, (N_domain, N_domain))))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta = jop.index_update(Theta, jop.index[N_domain:2*N_domain, N_domain:2*N_domain], jnp.reshape(val, (N_domain, N_domain)))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_DD_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta = jop.index_update(Theta, jop.index[N_domain:2*N_domain, 2*N_domain:3*N_domain], jnp.reshape(val, (N_domain, N_domain)))
        Theta = jop.index_update(Theta, jop.index[2*N_domain:3*N_domain, N_domain:2*N_domain], jnp.transpose(jnp.reshape(val, (N_domain, N_domain))))
        
        val = vmap(lambda x1, x2, y1, y2: K.DD_x2_DD_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta = jop.index_update(Theta, jop.index[2*N_domain:3*N_domain, 2*N_domain:3*N_domain], jnp.reshape(val, (N_domain, N_domain)))
        
        # interior+boundary v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXdbdb0.flatten(),XXdbdb1.flatten(),jnp.transpose(XXdbdb0).flatten(),jnp.transpose(XXdbdb1).flatten())
        Theta = jop.index_update(Theta, jop.index[3*N_domain:, 3*N_domain:], jnp.reshape(val, (N_domain+N_boundary, N_domain+N_boundary)))
        
        # interior v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta = jop.index_update(Theta, jop.index[0:N_domain, 3*N_domain:], jnp.reshape(val, (N_domain, N_domain+N_boundary)))
        Theta = jop.index_update(Theta, jop.index[3*N_domain:, 0:N_domain], jnp.transpose(jnp.reshape(val, (N_domain, N_domain+N_boundary))))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta = jop.index_update(Theta, jop.index[N_domain:2*N_domain, 3*N_domain:], jnp.reshape(val, (N_domain, N_domain+N_boundary)))
        Theta = jop.index_update(Theta, jop.index[3*N_domain:, N_domain:2*N_domain], jnp.transpose(onp.reshape(val, (N_domain, N_domain+N_boundary))))
        
        val = vmap(lambda x1, x2, y1, y2: K.DD_x2_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta = jop.index_update(Theta, jop.index[2*N_domain:3*N_domain, 3*N_domain:], jnp.reshape(val, (N_domain, N_domain+N_boundary)))
        Theta = jop.index_update(Theta, jop.index[3*N_domain:, 2*N_domain:3*N_domain], jnp.transpose(jnp.reshape(val, (N_domain, N_domain+N_boundary))))
        return Theta
    elif eqn == 'Eikonal':
        Theta = onp.zeros((4*N_domain + N_boundary, 4*N_domain + N_boundary))
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta[0:N_domain, 0:N_domain] =  jnp.reshape(val, (N_domain, N_domain))
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta[N_domain:2*N_domain, N_domain:2*N_domain] =  jnp.reshape(val, (N_domain, N_domain))
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta[:N_domain, N_domain:2*N_domain] =  jnp.reshape(val, (N_domain, N_domain))
        Theta[N_domain:2*N_domain, :N_domain] =  jnp.transpose(jnp.reshape(val, (N_domain, N_domain)))
        
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_Delta_y_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta[2*N_domain:3*N_domain, 2*N_domain:3*N_domain] = jnp.reshape(val, (N_domain, N_domain))
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta[2*N_domain:3*N_domain, :N_domain] = jnp.reshape(val, (N_domain, N_domain))
        Theta[:N_domain, 2*N_domain:3*N_domain] = jnp.transpose(jnp.reshape(val, (N_domain, N_domain)))
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),jnp.transpose(XXdd0).flatten(),jnp.transpose(XXdd1).flatten())
        Theta[2*N_domain:3*N_domain, N_domain:2*N_domain] = jnp.reshape(val, (N_domain, N_domain))
        Theta[N_domain:2*N_domain, 2*N_domain:3*N_domain] = jnp.transpose(jnp.reshape(val, (N_domain, N_domain)))
        
        # interior+boundary v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXdbdb0.flatten(),XXdbdb1.flatten(),jnp.transpose(XXdbdb0).flatten(),jnp.transpose(XXdbdb1).flatten())
        Theta[3*N_domain:, 3*N_domain:] =  jnp.reshape(val, (N_domain+N_boundary, N_domain+N_boundary))

        # interior v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta[:N_domain, 3*N_domain:] = jnp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta[3*N_domain:, :N_domain] = jnp.transpose(jnp.reshape(val, (N_domain, N_domain+N_boundary)))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta[N_domain:2*N_domain, 3*N_domain:] = jnp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta[3*N_domain:, N_domain:2*N_domain] = jnp.transpose(jnp.reshape(val, (N_domain, N_domain+N_boundary)))
        
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta[2*N_domain:3*N_domain, 3*N_domain:] = jnp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta[3*N_domain:, 2*N_domain:3*N_domain] = jnp.transpose(jnp.reshape(val, (N_domain, N_domain+N_boundary)))
        return Theta
    
    elif eqn == 'Darcy_flow2d':
        Theta_a = onp.zeros((3*N_domain, 3*N_domain))
        Theta_u = onp.zeros((4*N_domain + N_boundary, 4*N_domain + N_boundary))
        
        # Construct kernel matrix Theta_u
        # interior v.s. interior 
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta_u[0:N_domain, 0:N_domain] =  onp.reshape(val, (N_domain, N_domain))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta_u[N_domain:2*N_domain, N_domain:2*N_domain] =  onp.reshape(val, (N_domain, N_domain))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta_u[N_domain:2*N_domain, 0:N_domain] =  onp.reshape(val, (N_domain, N_domain))
        Theta_u[0:N_domain, N_domain:2*N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain)))
        
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_Delta_y_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta_u[2*N_domain:3*N_domain, 2*N_domain:3*N_domain] =  onp.reshape(val, (N_domain, N_domain))
        
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta_u[2*N_domain:3*N_domain, 0:N_domain] =  onp.reshape(val, (N_domain, N_domain))
        Theta_u[0:N_domain, 2*N_domain:3*N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain)))
        
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta_u[2*N_domain:3*N_domain, N_domain:2*N_domain] =  onp.reshape(val, (N_domain, N_domain))
        Theta_u[N_domain:2*N_domain, 2*N_domain:3*N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain)))
        
        # interior+boundary v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXdbdb0.flatten(),XXdbdb1.flatten(),onp.transpose(XXdbdb0).flatten(),onp.transpose(XXdbdb1).flatten())
        Theta_u[3*N_domain:, 3*N_domain:] =  onp.reshape(val, (N_domain+N_boundary, N_domain+N_boundary))

        # interior v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.D_x1_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta_u[0:N_domain, 3*N_domain:] = onp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta_u[3*N_domain:, 0:N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain+N_boundary)))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_x2_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta_u[N_domain:2*N_domain, 3*N_domain:] = onp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta_u[3*N_domain:, N_domain:2*N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain+N_boundary)))
        
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_kappa(x1, x2, y1, y2, kernel_parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta_u[2*N_domain:3*N_domain, 3*N_domain:] = onp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta_u[3*N_domain:, 2*N_domain:3*N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain+N_boundary)))
        
        # Construct kernel matrix Theta_a
        # interior v.s. interior 
        Theta_a[0:2*N_domain, 0:2*N_domain] = Theta_u[0:2*N_domain, 0:2*N_domain]
        Theta_a[2*N_domain:3*N_domain, 2*N_domain:3*N_domain] = Theta_u[3*N_domain:4*N_domain, 3*N_domain:4*N_domain]
        Theta_a[0:2*N_domain, 2*N_domain:3*N_domain] = Theta_u[0:2*N_domain, 3*N_domain:4*N_domain]
        Theta_a[2*N_domain:3*N_domain, 0:2*N_domain] = Theta_u[3*N_domain:4*N_domain, 0:2*N_domain]
        return Theta_u, Theta_a


def construct_Theta_test(X_test, X_domain, X_boundary, eqn = 'Nonlinear_elliptic', kernel = 'Gaussian', kernel_parameter = 0.2):
    N_test = X_test.shape[0]
    N_domain = X_domain.shape[0]
    N_boundary = X_boundary.shape[0]

    # auxiliary variables to make things readable
    # X_test coordinates
    Xt0=X_test[:,0]
    Xt1=X_test[:,1]
    
    # interior points coordinates
    Xd0=X_domain[:N_domain, 0]
    Xd1=X_domain[:N_domain, 1]
    
    # interior + boundary points coordinates
    Xdb0=jnp.concatenate([Xd0, X_boundary[:N_boundary, 0]])
    Xdb1=jnp.concatenate([Xd1, X_boundary[:N_boundary, 1]])
    
    # test v.s. interior
    XXtd0=jnp.transpose(jnp.tile(Xt0,(N_domain,1)))
    XXtd1=jnp.transpose(jnp.tile(Xt1,(N_domain,1)))
    XXtd0_2=jnp.tile(Xd0,(N_test,1))
    XXtd1_2=jnp.tile(Xd1,(N_test,1))
    
    # test v.s. interior + boundary
    XXtdb0=jnp.transpose(jnp.tile(Xt0,(N_domain+N_boundary,1)))
    XXtdb1=jnp.transpose(jnp.tile(Xt1,(N_domain+N_boundary,1)))
    XXtdb0_2=jnp.tile(Xdb0,(N_test,1))
    XXtdb1_2=jnp.tile(Xdb1,(N_test,1))
    
    if kernel == 'Gaussian':
        K = Gaussian_kernel()
    elif kernel == 'anisotropic_Gaussian':
        K = Anisotropic_Gaussian_kernel()
        
    # constructing Theta matrix
    if eqn == 'Nonlinear_elliptic':
        Theta_test = onp.zeros((N_test, 2*N_domain + N_boundary))
        val = vmap(lambda x1,x2,y1,y2: K.Delta_y_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_test[:,:N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXtdb0.flatten(),XXtdb1.flatten(),XXtdb0_2.flatten(),XXtdb1_2.flatten())
        Theta_test[:, N_domain:] = onp.reshape(val, (N_test, N_domain+N_boundary))
        return Theta_test
    elif eqn == 'Burgers':
        Theta_test = onp.zeros((N_test, 4*N_domain + N_boundary))
        # constructing Theta matrix
        val = vmap(lambda x1, x2, y1, y2: K.D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_test[:,:N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1, x2, y1, y2: K.D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_test[:,N_domain:2*N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1, x2, y1, y2: K.DD_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_test[:,2*N_domain:3*N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1, x2, y1, y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXtdb0.flatten(),XXtdb1.flatten(),XXtdb0_2.flatten(),XXtdb1_2.flatten())
        Theta_test[:,3*N_domain:4*N_domain+N_boundary] = onp.reshape(val, (N_test, N_domain+N_boundary))
        return Theta_test
    elif eqn == 'Eikonal':
        Theta_test = onp.zeros((N_test, 4*N_domain + N_boundary))
        val = vmap(lambda x1,x2,y1,y2: K.D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_test[:,:N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_test[:, N_domain:2*N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.Delta_y_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_test[:, 2*N_domain:3*N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXtdb0.flatten(),XXtdb1.flatten(),XXtdb0_2.flatten(),XXtdb1_2.flatten())
        Theta_test[:, 3*N_domain:] = onp.reshape(val, (N_test, N_domain+N_boundary))
        return Theta_test
    elif eqn == 'Darcy_flow2d':
        Theta_a_test = onp.zeros((N_test, 3*N_domain))
        Theta_u_test = onp.zeros((N_test, 4*N_domain+N_boundary))
        
         # constructing Theta_a_test matrix
        val = vmap(lambda x1,x2,y1,y2: K.D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_a_test[:,:N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_a_test[:,N_domain:2*N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_a_test[:,2*N_domain:] = onp.reshape(val, (N_test, N_domain))
        
        # constructing Theta_u_test matrix
        val = vmap(lambda x1,x2,y1,y2: K.D_y1_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_u_test[:,:N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.D_y2_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_u_test[:,N_domain:2*N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.Delta_y_kappa(x1, x2, y1, y2, kernel_parameter))(XXtd0.flatten(),XXtd1.flatten(),XXtd0_2.flatten(),XXtd1_2.flatten())
        Theta_u_test[:,2*N_domain:3*N_domain] = onp.reshape(val, (N_test, N_domain))
        
        val = vmap(lambda x1,x2,y1,y2: K.kappa(x1, x2, y1, y2, kernel_parameter))(XXtdb0.flatten(),XXtdb1.flatten(),XXtdb0_2.flatten(),XXtdb1_2.flatten())
        Theta_u_test[:,3*N_domain:] = onp.reshape(val, (N_test, N_domain+N_boundary))
        return Theta_u_test, Theta_a_test