import sympy
from sympy import Eq, Function, symbols
from sympy import Derivative as D
import numpy as np
import thalassa


t, x = symbols('t x')
u = Function('u')(t, x)
m1 = Function('m1')(x)
# g = Function('g')(t, x)
tensor_type = np.float64
pde = thalassa.PDESystem(
    [Eq(
        # D(u, t) + sympy.exp(u) * D(u, x), 0.005 * D(u, (x, 2))
        # D(u, t) + u * D(u, x), 0.005 * D(u, (x, 2))
        # D(u, t) + u * u * D(u, x), 0.005 * D(u, (x, 2))
        D(u, t) + u * D(u, x), 0.005 * D(u, (x, 2))
    )],
    [u],
    [t, x]
)

disc = [
    thalassa.fdm_simple_partial_derivative(D(u, (x, 1)), dx := 0.01, method='central'),
    thalassa.fdm_simple_partial_derivative(D(u, (x, 2)), dx, method='central')
]

dt = 0.0001
xs = np.linspace(0, 1, int(1 / dx), dtype=tensor_type)
u_initial = np.exp(-100 * (xs - 0.5) ** 2)

np.save('u_burgers.npy', u_initial)

with open('burgers.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', output='plot',
                                sol_hypercube=[4, int(1 / dx)], dt=dt,
                                funcs={m1: 'external'}, loop_iterations=250)
    output_file.write(code)















"""
t, x, y = symbols('t x y')
u = Function('u')(t, x, y)
v = Function('v')(t, x, y)

pde = thalassa.PDESystem(
    [
        Eq(D(u, t) + u * D(u, x) + v * D(u, y), 0),
        Eq(D(v, t) + v * D(v, y) + u * D(v, x), 0)
    ],
    [u, v],
    [t, x, y]
)

disc = [
    thalassa.fdm_simple_partial_derivative(D(u, x), dx := 0.01, method='central'),
    thalassa.fdm_simple_partial_derivative(D(u, y), dy := 0.01, method='central'),
    thalassa.fdm_simple_partial_derivative(D(v, x), dx, method='central'),
    thalassa.fdm_simple_partial_derivative(D(v, y), dy, method='central'),
]
np.save(f'2d_sin_2.npy', np.sin(
    2 * np.pi * np.linspace(0, 1, int(1 / dx)))[:, None] * np.sin(2 * np.pi * np.linspace(0, 1, int(1 / dy))))
with open('burgers.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', output='plot',
                                sol_hypercube=[12, int(1 / dx), int(1 / dy)], dt=0.001, loop_iterations=20)
    output_file.write(code)
"""