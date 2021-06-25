# JAX
import jax.numpy as jnp
from jax import vmap

# numpy
import numpy as onp
from kernel import Gaussian_kernel

def Gram_matrix_assembly(X_domain, X_boundary, eqn = 'Nonlinear_elliptic', kernel = 'Gaussian', parameter = 0.2):
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
    
    if eqn == 'Nonlinear_elliptic':
        if kernel == 'Gaussian':
            K = Gaussian_kernel(parameter)
        Theta = onp.zeros((2*N_domain + N_boundary, 2*N_domain + N_boundary))
        # Construct kernel matrix
        # interior v.s. interior (Laplace)
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_y_kappa(x1, x2, y1, y2, parameter))(XXdd0.flatten(),XXdd1.flatten(),onp.transpose(XXdd0).flatten(),onp.transpose(XXdd1).flatten())
        Theta[0:N_domain, 0:N_domain] =  onp.reshape(val, (N_domain, N_domain))
        
        # interior+boundary v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.kappa(x1, x2, y1, y2, parameter))(XXdbdb0.flatten(),XXdbdb1.flatten(),onp.transpose(XXdbdb0).flatten(),onp.transpose(XXdbdb1).flatten())
        Theta[N_domain:, N_domain:] =  onp.reshape(val, (N_domain+N_boundary, N_domain+N_boundary))

        # interior v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: K.Delta_x_kappa(x1, x2, y1, y2, parameter))(XXddb0.flatten(),XXddb1.flatten(),XXddb0_2.flatten(),XXddb1_2.flatten())
        Theta[:N_domain, N_domain:] = onp.reshape(val, (N_domain, N_domain+N_boundary))
        Theta[N_domain:, :N_domain] = onp.transpose(onp.reshape(val, (N_domain, N_domain+N_boundary)))
        
    return Theta