# Physics-Informed Neural Network (PINN) for the Poisson Equation

A neural network that solves the Poisson equation by building the PDE
itself into the loss function, instead of learning only from labeled data.

## Problem
Solving: -∇²u(x) = f(x) on the domain [domain], with boundary
conditions [Dirichlet / Neumann: describe here].

## Approach
- Network: [e.g., fully connected, 4 hidden layers, 50 neurons, tanh]
- Loss = PDE residual loss + boundary condition loss
- Derivatives computed with automatic differentiation
- Optimizer: [Adam / L-BFGS], [number] epochs

## Results
- Error vs. exact solution: [L2 error]
- Plots: predicted vs. exact solution (add a screenshot below)

![Result](results.png)

## How to run
1. Clone the repo: `git clone https://github.com/prggr8/PINN.git`
2. Install: `pip install torch numpy matplotlib` (or TensorFlow, depending on your notebook)
3. Open `PINN_poisson_eqn.ipynb` in Jupyter and run all cells

## Tools
Python, [PyTorch/TensorFlow], NumPy, Matplotlib# PINN
