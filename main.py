from sympy import Eq, Function, symbols
from sympy import Derivative as D
import numpy as np
import thalassa

t, x = symbols('t x')
u = Function('u')(t, x)

pde = thalassa.PDESystem(
    [Eq(
        D(u, (t, 2)) - D(u, (x, 2)), 0
    )],
    [u],
    [t, x]
)

disc = [
    thalassa.fdm_simple_partial_derivative(D(u, (x, 2)), dx := 0.01, method='central')
]
np.save('1d_expsq.npy', np.exp(-(np.linspace(0, 1, int(1 / dx)) - 0.5) ** 2 * 100))
np.save('1d_sin2pi.npy', np.sin(2 * np.pi * np.linspace(0, 1, int(1 / dx))))
np.save('1d_zeros.npy', np.zeros(int(1 / dx)))

with open('burgers.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', plot=True,
                                sol_hypercube=[10000, int(1 / dx)], dt=0.00001)
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
    thalassa.fdm_simple_partial_derivative(D(u, t), dt := 0.001, method='forward'),
    thalassa.fdm_simple_partial_derivative(D(u, x), dx := 0.01, method='central'),
    thalassa.fdm_simple_partial_derivative(D(u, y), dy := 0.01, method='central'),
    thalassa.fdm_simple_partial_derivative(D(v, t), dt, method='forward'),
    thalassa.fdm_simple_partial_derivative(D(v, x), dx, method='central'),
    thalassa.fdm_simple_partial_derivative(D(v, y), dy, method='central'),
]
np.save(f'2d_sin_2.npy', np.sin(
    2 * np.pi * np.linspace(0, 1, int(1 / dx)))[:, None] * np.sin(2 * np.pi * np.linspace(0, 1, int(1 / dy))))
with open('burgers.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', plot=True,
                                sol_hypercube=[150, int(1 / dx), int(1 / dy)])
    output_file.write(code)
"""
