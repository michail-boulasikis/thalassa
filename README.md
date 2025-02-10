# Thalassa

Thalassa is a Python package that helps generate explicit numerical solvers for Partial Differential Equations (PDEs)
implemented in PyTorch. 
Its goal is to explore using AI hardware to efficiently run traditional scientific computing algorithms.
For that reason, the solver use a subset of PyTorch that can run on both CPUs, GPUs and AI hardware such
as Google Cloud TPUs.

## Requirements

Thalassa requires `python 3.10` and above. For a list of required libraries, see `requirements.txt`

## Example

This example shows how to generate a tensor-based solver for Burgers' equation $\partial_tu + u\partial_xu = \alpha\partial_{xx}u$,
with $u = u(x,t)$ and $\alpha = 0.001$.

We first import sympy and Thalassa. Thalassa uses sympy as its symbolic manipulation engine.
We also generate the sympy symbols needed to describe our PDE.

```python
from sympy import Eq, Function, symbols
from sympy import Derivative as D
import thalassa

t, x = symbols('t x')
u = Function('u')(t, x)
```

Thalassa provides a PDESystem class which can be used to describe a PDE system. We construct that as follows:

```python
alpha = 0.001
pde = thalassa.PDESystem(
    # The PDE(s) in a list
    [Eq(D(u, t) + u * D(u, x), alpha * D(u, (x, 2)))],
    # The unknown(s) of the PDE(s) in a list
    [u],
    # The parameters of the unknown functions (time first)
    [t, x]                                              
)
```

Thalassa requires a discretization method which maps the derivatives to finite differences. The library provides a
helper function to create simple forward/backward/central difference methods of arbitrary order. Time derivatives are
approximated by Thalassa to guarantee that the resulting method is explicit. Here, we just supply discretization
methods for the 1st and 2nd spatial derivative.

```python
# Discretization step
dx = 0.01 
# Discretizations in a list
disc = [
    thalassa.fdm_simple_partial_derivative(D(u, (x, 1)), dx, method='central'),
    thalassa.fdm_simple_partial_derivative(D(u, (x, 2)), dx, method='central')
]
```

Thalassa can be invoked as follows:
```python
dt = 0.001
with open('burgers.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', output='plot',
                                sol_hypercube=[4, int(1 / dx)], dt=dt, loop_iterations=250)
    output_file.write(code)
```

This generates a file named `burgers.py` which implements a solver for the PDE in PyTorch. The used options do the following:
- `target='pytorch'` specifies that the solver will be described as a PyTorch network. Other targets are WIP.
- `ics='external'` specifies that the initial conditions should be given as a command line argument to the solver.
- `output` controls the product of the solver. In this case, the output will be a `plot` of the solution at the end time.
- `sol_hypercube` specifies the hypercube one step of the solver will compute.
- `dt` specifies the discretization step in time.
- `loop_iterations` specifies how many time steps the solver will compute before stopping.

The elapsed time after the end of the simulation is $\Delta t \times H_t \times I$, where $H_t$ is the value of the first
dimension in the solution hypercube and $I$ is the number of `loop_iterations`.

### Namesake

Thalassa comes from the greek word Θάλασσα, which means sea.

## Acknowledgement

This work is part of ELLIIT project C05, GPAI (General Purpose AI Computing).