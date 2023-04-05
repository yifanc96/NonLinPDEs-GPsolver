# NonlinearPDEs_GPsolver
Code for the paper [Solving and Learning Nonlinear PDEs with Gaussian Processes](https://arxiv.org/abs/2103.12959)

Packages required: [JAX](https://github.com/google/jax)
- [JAX](https://github.com/google/jax) is used for automatic differentiation and vectorization of the constriction of Gram matrices, and the Gauss-Newton iteration. It can be avoid if the users supplement the derivatives manually.
- Note: Codes may plot some figures along with its execution. Please make sure *latex* is supported in the machine environment, otherwise you may need to change the plot configuration in the code manually.
- Note: the algorithm in the paper https://arxiv.org/abs/2103.12959 can be optimized further to greatly improve the efficiency. Please follow up with our recent progress.
- The repo https://github.com/yifanc96/PDEs-GP-KoleskySolver presents a near linear complexity algorithm for the GP-PDE solver

### For demonstration of use in solving PDEs and inverse problems covered in the paper 
run the following in your terminal
- `python main_NonLinElliptic2d.py --kernel Gaussian --kernel_parameter 0.2 --nugget 1e-13 --N_domain 900 --N_boundary 124 --GNsteps 4 --show_figure True`
  
- `python main_Burgers1d.py --kernel anisotropic_Gaussian --kernel_parameter 0.3 0.05 --nugget 1e-5 --N_domain 1000 --N_boundary 200 --GNsteps 8 --show_figure True`
  
- `python main_Eikonal2d.py --kernel Gaussian --kernel_parameter 0.2 --nugget 1e-5 --N_domain 1000 --N_boundary 200 --GNsteps 8 --show_figure True`
  
- `python main_DarcyFlow2d.py --kernel Gaussian --kernel_parameter 0.2 --nugget 1e-8 --N_domain 400 --N_boundary 100 --N_data 60 --noise_level 0.001 --GNsteps 8 --show_figure True`


### The architecture of this code
In the `src` folder:
- `PDEs.py` contains PDEs classes with built-in methods that are used to run our algorithm, while `InverseProblems.py `contains the corresponding part for inverse problems
- `sample_points.py` provides several ways of sampling collocation points; users can also provide their own setting of collocation points
- `kernels.py` provides a collection of kernel functions and their derivatives
- `Gram_matrice.py` constructs the kernel matrix used in training and testing stages for our GP based method
- `solver.py` contains a high level class that integrate the above three files to run the algorithms for any PDEs and inverse problems

In addition, folder `reference_solver` contains several classical solvers for these PDEs, which are used for comparison purposes