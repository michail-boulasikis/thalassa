from thalassa import pde_compile
import thalassa
from sympy import Eq
from sympy import Derivative as D
import sympy
import os
import numpy as np

t, x, y = sympy.symbols('t x y')
u, v, p = sympy.Function('u'), sympy.Function('v'), sympy.Function('p')
f, g, h1, h2 = sympy.Function('f'), sympy.Function('g'), sympy.Function('h1'), sympy.Function('h2')

solve_2d = True

Nt = 200
Nx = 800
dx = 1.0 / (Nx - 1)
dt = 1.0 / (Nt - 1)
ts = np.linspace(0, 1, Nt)
xs = np.linspace(0, 1, Nx)
xs = xs[1:-1]
ts = ts[1:-1]

probs = [
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + 0.1 * D(u(t, x), x), 0)], [u(t, x)], [t, x], name='T-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) - f(x) * D(u(t, x), x), 0)], [u(t, x)], [t, x], name='T-NC-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + 0.1 * D(u(t, x), x), g(t, x))], [u(t, x)], [t, x], name='T-F-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + u(t, x) * D(u(t, x), x), 0)], [u(t, x)], [t, x], name='T-B1-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + 0.1 * D(u(t, x) * u(t, x) + u(t, x), x), 0)], [u(t, x)], [t, x],
                                  name='T-B2-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem(
            [Eq(D(u(t, x), t) + 0.1 * u(t, x) * D(u(t, x), x), -f(x) * D(u(t, x), (x, 2)) + g(t, x))],
            [u(t, x)], [t, x], name='T-B3-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), dx, method='central')

        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t), D(u(t, x), (x, 2)))], [u(t, x)],
                                  [t, x], name='H-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) - 0.05 * D(u(t, x), x, x), 0.1 * sympy.exp(u(t, x)))], [u(t, x)],
                                  [t, x], name='H-KPP1-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) - 0.05 * D(u(t, x), x, x), 0.1 * D(u(t, x), x) * D(u(t, x), x))],
                                  [u(t, x)],
                                  [t, x], name='H-KPP2-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(sympy.sqrt(D(u(t, x), t)) - 0.001 * D(u(t, x), x, x), u(t, x))],
                                  [u(t, x)],
                                  [t, x], name='H-KPP3-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), dx, method='central')
        ]
    },
    {
        # Wave equation in 2D
        "pde": thalassa.PDESystem(
            [Eq(D(u(t, x, y), (t, 2)) - 0.1 * D(u(t, x, y), x, x) - 0.1 * D(u(t, x, y), y, y), 0)], [u(t, x, y)],
            [t, x, y], name='W-2D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (x, 2)), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (y, 2)), dx, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([
            Eq(D(u(t, x, y), t) + u(t, x, y) * 0.1 * D(u(t, x, y), x) + 0.1 * v(t, x, y) * D(u(t, x, y), y), 0),
            Eq(D(v(t, x, y), t) + v(t, x, y) * 0.1 * D(v(t, x, y), y) + 0.1 * u(t, x, y) * D(v(t, x, y), x), 0)
        ], [u(t, x, y), v(t, x, y)], [t, x, y], name='W-B1-2D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), x), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), y), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), x), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), y), dx, method='central'),
        ]
    },
    {
        "pde": thalassa.PDESystem([
            Eq(D(u(t, x, y), t) + 0.1 * u(t, x, y) * D(u(t, x, y), x) + 0.1 * v(t, x, y) * D(u(t, x, y), y),
               0.02 * (D(u(t, x, y), (x, 2)) + D(u(t, x, y), (y, 2))) + h1(t, x, y)),
            Eq(D(v(t, x, y), t) + 0.1 * v(t, x, y) * D(v(t, x, y), y) + 0.1 * u(t, x, y) * D(v(t, x, y), x),
               0.02 * (D(v(t, x, y), (x, 2)) + D(v(t, x, y), (y, 2))) + h2(t, x, y))
        ], [u(t, x, y), v(t, x, y)], [t, x, y], name='W-B2-2D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), x), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), y), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (x, 2)), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (y, 2)), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), x), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), y), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (x, 2)), dx, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (y, 2)), dx, method='central'),
        ]
    }
]

""",
    {
        "name": "navier-stokes-2d",
        "pde": thalassa.PDESystem([
            Eq(D(u(t, x, y), t) + u(t, x, y) * D(u(t, x, y), x) + v(t, x, y) * D(u(t, x, y), y),
               -D(p(t, x, y), x) + 0.01 * (D(u(t, x, y), (x, 2)) + D(u(t, x, y), (y, 2)))),
            Eq(D(v(t, x, y), t) + v(t, x, y) * D(v(t, x, y), y) + u(t, x, y) * D(v(t, x, y), x),
               -D(p(t, x, y), y) + 0.01 * (D(v(t, x, y), (x, 2)) + D(v(t, x, y), (y, 2)))),
            Eq(D(p(t, x, y), (x, 2)) + D(p(t, x, y), (y, 2)),
               -(D(u(t, x, y), x) ** 2 + 2 * D(u(t, x, y), y)) * D(v(t, x, y), x) + D(v(t, x, y), (y, 2)) ** 2)
        ], [u(t, x, y), v(t, x, y), p(t, x, y)], [t, x, y]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (y, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (y, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(p(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(p(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(p(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(p(t, x, y), (y, 2)), 0.05, method='central'),
        ]
    }
    """


def generate_ics(dir_path):
    np.save(f'{dir_path}/x_expsquare.npy', np.exp(-(xs - 0.5) ** 2 / 0.01))
    np.save(f'{dir_path}/x_zeros.npy', np.zeros_like(xs))
    Xs, Ts = np.meshgrid(xs, ts)
    np.save(f'{dir_path}/xt_expsquare_decayed.npy', np.exp(-(Xs - 0.5) ** 2 / 0.02) * np.exp(-Ts / 0.2))
    if solve_2d:
        Xs, Ys = np.meshgrid(xs, xs)
        np.save(f'{dir_path}/xy_sin.npy', np.sin(np.pi * Xs) * np.sin(np.pi * Ys)),
        np.save(f'{dir_path}/xy_sin_2.npy',
                np.sin(4 * np.pi * Xs) * np.sin(4 * np.pi * Ys))
        np.save(f'{dir_path}/xy_expsquare.npy', np.exp(-((Xs - 0.5) ** 2) / 0.05) * np.exp(-((Ys - 0.5) ** 2) / 0.05))
        np.save(f'{dir_path}/xy_zeros.npy', np.zeros_like(Xs))


if __name__ == '__main__':
    problem_dir = "generated-code"
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    data_dir = "generated-data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    generate_ics(problem_dir)
    for prob in probs:
        file_name = f"{problem_dir}/{prob['pde'].name.replace('-', '_')}.py"
        sol_hypercube = [Nt, Nx, Nx] if '2D' in prob['pde'].name else [Nt, Nx]
        if not solve_2d and '2D' in prob['pde'].name:
            continue
        print(f"Generating {file_name}")
        with open(file_name, 'w') as prob_generated_code:
            code = pde_compile(prob["pde"], prob["disc"], funcs={
                f(x): -0.05 - 0.02 * sympy.sin(sympy.pi * x),
                g(t, x): 0.05 * sympy.sin(8 * sympy.pi * x) * sympy.sin(2 * sympy.pi * t),
                h1(t, x, y): 'external',
                h2(t, x, y): 'external'
            }, sol_hypercube=sol_hypercube, output='external', ics='external', target='pytorch', loop_iterations=1,
                               dt=dt, timeit=True, device='gpu', num_acc='f32')
            prob_generated_code.write(code)
