import sympy
from sympy import Eq, Function, symbols
from sympy import Derivative as D
import numpy as np
import thalassa

t, x = symbols('t x')
u = Function('u')(t, x)
tensor_type = np.float64

pde = thalassa.PDESystem(
    [Eq(D(u, t) + u * D(u, x), 0)],
    [u],
    [t, x]
)

disc = [
    thalassa.fdm_simple_partial_derivative(D(u, x), dx := 0.005, method='central'),
]

dt = 0.0001
xs = np.linspace(0, 1, int(1 / dx), dtype=tensor_type)
u_initial = np.exp(-100 * (xs - 0.5) ** 2)

np.save('u_burgers.npy', u_initial)

with open('burgers.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', output='plot',
                                sol_hypercube=[10, int(1 / dx)], dt=dt, loop_iterations=60)
    output_file.write(code)
