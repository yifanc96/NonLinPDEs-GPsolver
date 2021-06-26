import jax.numpy as jnp
from jax import grad

class Gaussian_kernel(object):
    def __init__(self):
        pass
    
    def kappa(self, x1,x2,y1,y2,sigma):
        return jnp.exp(-(1/(2*sigma**2))*( (x1- y1)**2 + (x2 - y2)**2))
    
    def D_x1_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,0)(x1, x2, y1, y2, sigma)
        return val
    
    def D_x2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,1)(x1, x2, y1, y2, sigma)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,1),1)(x1, x2, y1, y2, sigma)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,2)(x1, x2, y1, y2, sigma)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,3)(x1, x2, y1, y2, sigma)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,3),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,0),2)(x1, x2, y1, y2, sigma)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,0),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(grad(self.kappa,0),3),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,1),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(grad(self.kappa,1),3),3)(x1, x2, y1, y2, sigma)
        return val

    def DD_x2_DD_y2_kappa(self,x1, x2, y1, y2, sigma):
        val = grad(grad(grad(grad(self.kappa,1),1),3),3)(x1, x2, y1, y2, sigma)
        return val
    
    def Delta_x_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2, sigma)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2, sigma)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2, sigma)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2, sigma)
        return val

    def Delta_x_y_kappa(self,x1, x2, y1, y2, sigma):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2, sigma)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2, sigma)
        return val
    
class Anisotropic_Gaussian_kernel(object):
    def __init__(self):
        pass
    
    def kappa(self, x1,x2,y1,y2,sigma):
        scale_t = sigma[0]
        scale_x = sigma[1]
        r = ((x1-y1)/scale_t)**2+((x2-y2)/scale_x)**2
        return jnp.exp(-r)
    
    def D_x1_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,0)(x1, x2, y1, y2, sigma)
        return val
    
    def D_x2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,1)(x1, x2, y1, y2, sigma)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,1),1)(x1, x2, y1, y2, sigma)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,2)(x1, x2, y1, y2, sigma)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(self.kappa,3)(x1, x2, y1, y2, sigma)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,3),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,0),2)(x1, x2, y1, y2, sigma)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,0),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(grad(self.kappa,0),3),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa,1),3)(x1, x2, y1, y2, sigma)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(grad(self.kappa,1),3),3)(x1, x2, y1, y2, sigma)
        return val

    def DD_x2_DD_y2_kappa(self,x1, x2, y1, y2, sigma):
        val = grad(grad(grad(grad(self.kappa,1),1),3),3)(x1, x2, y1, y2, sigma)
        return val
    
    def Delta_x_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2, sigma)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2, sigma)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2, sigma):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2, sigma)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2, sigma)
        return val

    def Delta_x_y_kappa(self,x1, x2, y1, y2, sigma):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2, sigma)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2, sigma)
        return val
            
        