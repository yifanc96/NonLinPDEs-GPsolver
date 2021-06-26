import numpy as onp
from numpy import random 


def sampled_pts_rdm(N_domain, N_boundary, time_dependent = False):
    if time_dependent == False:
        #[0,1]*[0,1]
        # interior nodes
        X_domain = random.uniform(0.0, 1.0, (N_domain, 2))
        N_boundary_per_bd = int(N_boundary/4)
        X_boundary = onp.zeros((N_boundary_per_bd*4, 2))
        
        # bottom face
        X_boundary[0:N_boundary_per_bd, 0] = random.uniform(0.0, 1.0, N_boundary_per_bd)
        # right face
        X_boundary[N_boundary_per_bd:2*N_boundary_per_bd, 0] += 1
        X_boundary[N_boundary_per_bd:2*N_boundary_per_bd, 1] = random.uniform(0.0, 1.0, N_boundary_per_bd)
        # top face
        X_boundary[2*N_boundary_per_bd:3*N_boundary_per_bd, 0] = random.uniform(0.0, 1.0, N_boundary_per_bd)
        X_boundary[2*N_boundary_per_bd:3*N_boundary_per_bd, 1] += 1
        # left face
        X_boundary[3*N_boundary_per_bd:4*N_boundary_per_bd, 1] = random.uniform(0.0, 1.0, N_boundary_per_bd)
    else:
        #[0,1]*[-1,1]
        # interior nodes
        X_domain = random.uniform(0, 1.0, (N_domain, 2))
        X_domain[:,1] = 2*X_domain[:,1]-1
        N_boundary_per_bd = int(N_boundary/3)
        X_boundary = onp.zeros((N_boundary_per_bd*3, 2))

        # bottom face
        X_boundary[0:N_boundary_per_bd, 1] = random.uniform(-1.0, 1.0, N_boundary_per_bd)
        # right face
        X_boundary[N_boundary_per_bd:2*N_boundary_per_bd, 0] = random.uniform(0.0, 1.0, N_boundary_per_bd)
        X_boundary[N_boundary_per_bd:2*N_boundary_per_bd, 1] = 1
        # left face
        X_boundary[2*N_boundary_per_bd:, 0] = random.uniform(0.0, 1.0, N_boundary_per_bd)
        X_boundary[2*N_boundary_per_bd:, 1] = -1
        
    return X_domain, X_boundary

def sampled_pts_grid(N_domain, N_boundary, time_dependent = False):
    if time_dependent == False:
        N_pts = int(onp.sqrt(N_domain+N_boundary))-2
        xx= onp.linspace(0, 1, N_pts+2)
        yy = onp.linspace(0, 1, N_pts+2)
        XX, YY = onp.meshgrid(xx, yy)

        XX_int = XX[1:N_pts+1, 1:N_pts+1]
        YY_int = YY[1:N_pts+1, 1:N_pts+1]

        # vectorized (x,y) coordinates
        XXv_int = onp.array(XX_int.flatten())
        YYv_int = onp.array(YY_int.flatten())

        XXv_int = onp.expand_dims(XXv_int, axis=1) 
        YYv_int = onp.expand_dims(YYv_int, axis=1) 
        
        XXv_bd = onp.concatenate((XX[0,0:N_pts+1], XX[N_pts+1,0:N_pts+1], XX[0:N_pts+1,0], XX[0:N_pts+1,N_pts+1]), axis = 0)
        YYv_bd = onp.concatenate((YY[0,0:N_pts+1], YY[N_pts+1,0:N_pts+1], YY[0:N_pts+1,0], YY[0:N_pts+1,N_pts+1]), axis = 0)

        XXv_bd = onp.expand_dims(XXv_bd, axis=1) 
        YYv_bd = onp.expand_dims(YYv_bd, axis=1) 
        
        X_domain = onp.concatenate((XXv_int, YYv_int), axis=1)
        X_boundary = onp.concatenate((XXv_bd, YYv_bd), axis=1)
    else:
        N_pts = int(onp.sqrt(N_domain+N_boundary))-2
        xx= onp.linspace(0, 1, N_pts+2)
        yy = onp.linspace(-1, 1, N_pts+2)
        XX, YY = onp.meshgrid(xx, yy)

        XX_int = XX[1:N_pts+1, 1:N_pts+2]
        YY_int = YY[1:N_pts+1, 1:N_pts+2]

        # vectorized (x,y) coordinates
        XXv_int = onp.array(XX_int.flatten())
        YYv_int = onp.array(YY_int.flatten())

        XXv_int = onp.expand_dims(XXv_int, axis=1) 
        YYv_int = onp.expand_dims(YYv_int, axis=1) 
        
        XXv_bd = onp.concatenate((XX[0,1:N_pts+2], XX[N_pts+1,1:N_pts+2], XX[0:N_pts+2,0]), axis = 0)
        YYv_bd = onp.concatenate((YY[0,1:N_pts+2], YY[N_pts+1,1:N_pts+2], YY[0:N_pts+2,0]), axis = 0)

        XXv_bd = onp.expand_dims(XXv_bd, axis=1) 
        YYv_bd = onp.expand_dims(YYv_bd, axis=1) 
        X_domain = onp.concatenate((XXv_int, YYv_int), axis=1)
        X_boundary = onp.concatenate((XXv_bd, YYv_bd), axis=1)
    return X_domain, X_boundary