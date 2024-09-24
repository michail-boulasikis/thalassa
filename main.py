import sympy
from sympy import Eq, Function, symbols
from sympy import Derivative as D
import numpy as np
import thalassa

t, x = symbols('t x')
u = Function('u')(t, x)
m1 = Function('m1')(x)
#g = Function('g')(t, x)
tensor_type = np.float64
pde = thalassa.PDESystem(
    [Eq(
        D(u, (t, 2)) + D(u, (x, 2)), 0
    )],
    [u],
    [t, x]
)

disc = [
    thalassa.fdm_simple_partial_derivative(D(u, (x, 2)), dx := 0.02, method='central')
]
dt = 0.0001
xs = np.linspace(0, 1, int(1 / dx), dtype=tensor_type)
#ts = np.linspace(0, 1, int(1 / dt), dtype=tensor_type)
u_initial = np.exp(-80 * (xs - 0.5)**2)
#Ts, Xs = np.meshgrid(ts, xs, indexing='ij')
#g_forcing = np.pad(0.01 * np.sin(2 * np.pi * Xs) * np.exp(-200*Ts), [(1, 1), (1, 1)], mode='constant', constant_values=0.)

np.save('u_burgers.npy', u_initial)
#np.save('g_burgers.npy', g_forcing)

with open('burgers.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', output='plot',
                                sol_hypercube=[1000, int(1 / dx)], dt=dt,
                                funcs={})
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
