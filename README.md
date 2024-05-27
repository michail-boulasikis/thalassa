# Thalassa

Thalassa (from the Greek word Θάλασσα, meaning sea) is a Python package that helps generate explicit numerical solvers for Partial Differential Equations (PDEs) implemented in PyTorch. The solvers use the subset of PyTorch that can run on both CPUs, GPUs and AI hardware such as Google Cloud TPUs. Thalassa uses the method of auxiliary features to capture the non-linear and non-constant parts of the equation. Examples of equations that Thalassa can implement solvers for include:
- The transport equation (u_t + a u_x = 0) which is a linear PDE with a constant coefficient.
- The non-constant coefficient transport equation (u_t + a(x) u_x = 0) which is a linear PDE with a non-constant coefficient.
- Burgers' equation (u_t + u u_x = 0) which is a non-linear PDE.
- The non-linear advection-diffusion equation (u_t + u u_x = c u_xx) which is a non-linear PDE with second order spatial derivatives.
- 2D versions of Burgers' equation and the advection-diffusion equation.
- The Kolmogorov-Petrovsky-Piskunov (KPP) equation (u_t = u_xx + exp(u)) which has a non-linear term in u.
- Equations with derivatives in non-linear functions, such as (u_t = u_xx + exp(u_x)).

Thalassa can not yet solve the Navier-Stokes equation because of the pressure term. This is a future goal for the project.

## Changes from PDE

There are a few changes from the original PDE solver. The main changes are:
- Use of PyTorch instead of TensorFlow. This is because PyTorch is more flexible and widely used in academia than TensorFlow. This might create some issues when running the code on Google TPUs.
- Use of SymPy in the PDE front-end. Interfacing with SymPy is better than creating a custom language for describing PDE problems.

## Acknowledgement

This work is part of ELLIIT project C05, GPAI (General Purpose AI Computing).