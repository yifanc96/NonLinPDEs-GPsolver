# NonlinearPDEs_GPsolver
Code for the paper [Solving and Learning Nonlinear PDEs with Gaussian Processes](https://arxiv.org/abs/2103.12959)

Packages required: JAX, munch
- JAX is used for automatic differentiation and vectorization of the constriction of Gram matrices, and the Gauss-Newton iteration.
- Note: Codes may plot some figures along with its execution. Please make sure latex is supported in the machine environment, otherwise you may need to change the plot configuration in the code manually.

**For demonstration of use in solving PDEs and inverse problems covered in the paper**: run or modify the configuration dictionary in the following files
- `main_demonstration_Elliptic.py`
- `main_demonstration_Burgers.py`
- `main_demonstration_Eikonal.py`
- `main_demonstration_Darcy_flow2d.py`

**For error curve analysis regarding the number of collocation points**: run or modify the configuration dictionary in the following
- `main_elliptic_err_curve`
- `main_Burgers_err_curve`
- `main_Eikonal_err_curve`

**The architecture of this code**:
- `PDEs.py` contains PDEs classes with built-in methods that are used to run our algorithm, while InverseProblems.py contains the corresponding part for inverse problems
- `sample_points.py` provides several ways of sampling collocation points; users can also provide their own setting of collocation points
- `kernels.py` provides a collection of kernel functions and their derivatives
- `Gram_matrice.py` constructs the kernel matrix used in training and testing stages for our GP based method
- `solver.py` contains a high level class that integrate the above three files to run the algorithms for any PDEs and inverse problems
- Folder standard_solver contains several classical solvers for these PDEs, which are used for comparison purposes