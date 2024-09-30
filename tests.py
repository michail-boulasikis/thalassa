from thalassa import pde_compile
import thalassa
from sympy import Eq
from sympy import Derivative as D
import sympy
import os
import time

t, x, y = sympy.symbols('t x y')
u, v, p = sympy.Function('u'), sympy.Function('v'), sympy.Function('p')
f, g, h = sympy.Function('f'), sympy.Function('g'), sympy.Function('h')

probs = [
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + 2.0 * D(u(t, x), x), 0)], [u(t, x)], [t, x], name='T-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + f(x) * D(u(t, x), x), 0)], [u(t, x)], [t, x], name='T-NC-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + 2.0 * D(u(t, x), x), g(t, x))], [u(t, x)], [t, x], name='T-F-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + u(t, x) * D(u(t, x), x), 0)], [u(t, x)], [t, x], name='T-B1-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + D(u(t, x)*u(t, x) + u(t, x), x), 0)], [u(t, x)], [t, x], name='T-B2-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + u(t, x) * D(u(t, x), x), f(x) * D(u(t, x), (x, 2)) + g(t, x))], [u(t, x)], [t, x], name='T-B3-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')

        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t), D(u(t, x), (x, 2)))], [u(t, x)],
                                  [t, x], name='H-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) - 0.05 * D(u(t, x), x, x), 0.5 * sympy.exp(u(t, x)))], [u(t, x)],
                                  [t, x], name='H-KPP1-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) - 0.05 * D(u(t, x), x, x), 0.5 * sympy.sin(D(u(t, x), x)))],
                                  [u(t, x)],
                                  [t, x], name='H-KPP2-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([Eq(sympy.exp(D(u(t, x), t)) - 0.05 * D(u(t, x), x, x), 0)],
                                  [u(t, x)],
                                  [t, x], name='H-KPP3-1D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')
        ]
    },
    {
        # Wave equation in 2D
        "pde": thalassa.PDESystem([Eq(D(u(t, x, y), t) - 0.05 * D(u(t, x, y), x, x) - 0.05 * D(u(t, x, y), y, y), 0)], [u(t, x, y)],
                                  [t, x, y], name='W-2D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (y, 2)), 0.05, method='central')
        ]
    },
    {
        "pde": thalassa.PDESystem([
            Eq(D(u(t, x, y), t) + u(t, x, y) * D(u(t, x, y), x) + v(t, x, y) * D(u(t, x, y), y), 0),
            Eq(D(v(t, x, y), t) + v(t, x, y) * D(v(t, x, y), y) + u(t, x, y) * D(v(t, x, y), x), 0)
        ], [u(t, x, y), v(t, x, y)], [t, x, y], name='W-B1-2D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), y), 0.05, method='central'),
        ]
    },
    {
        "pde": thalassa.PDESystem([
            Eq(D(u(t, x, y), t) + u(t, x, y) * D(u(t, x, y), x) + v(t, x, y) * D(u(t, x, y), y),
               0.02 * (D(u(t, x, y), (x, 2)) + D(u(t, x, y), (y, 2))) + f(t, x, y)),
            Eq(D(v(t, x, y), t) + v(t, x, y) * D(v(t, x, y), y) + u(t, x, y) * D(v(t, x, y), x),
               0.02 * (D(v(t, x, y), (x, 2)) + D(v(t, x, y), (y, 2))) + h(t, x, y))
        ], [u(t, x, y), v(t, x, y)], [t, x, y], name='W-B2-2D'),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (y, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (y, 2)), 0.05, method='central'),
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


def generate_ics(dir_path, cube_sz=65):
    import numpy as np
    np.save(f'{dir_path}/1d_sin.npy', np.sin(np.pi * np.linspace(0, 1, cube_sz))),
    np.save(f'{dir_path}/1d_sin_2.npy', np.sin(2 * np.pi * np.linspace(0, 1, cube_sz))),
    np.save(f'{dir_path}/2d_sin.npy',
            np.sin(np.pi * np.linspace(0, 1, cube_sz))[:, None] * np.sin(np.pi * np.linspace(0, 1, cube_sz)))
    np.save(f'{dir_path}/2d_sin_2.npy',
            np.sin(4 * np.pi * np.linspace(0, 1, cube_sz))[:, None] * np.sin(4 * np.pi * np.linspace(0, 1, cube_sz)))
    np.save(f'{dir_path}/1d_expsquare.npy', np.exp(-((np.linspace(0, 1, cube_sz) - 0.5) ** 2) / 0.02))
    np.save(f'{dir_path}/2d_expsquare.npy', np.exp(-((np.linspace(0, 1, cube_sz) - 0.5) ** 2) / 0.02)[:, None] * np.exp(
        -((np.linspace(0, 1, cube_sz) - 0.5) ** 2) / 0.01))


if __name__ == '__main__':
    problem_dir = "generated-code"
    prob_sz = 128
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    generate_ics(problem_dir, prob_sz)
    for prob in probs:
        file_name = f"{problem_dir}/{prob['pde'].name.replace('-', '_')}.py"
        sol_hypercube = [prob_sz // 5, prob_sz, prob_sz] if '2d' in prob['pde'].name else [prob_sz // 5, prob_sz]
        print(f"Generating {file_name}")
        with open(file_name, 'w') as prob_generated_code:
            code = pde_compile(prob["pde"], prob["disc"], funcs={
                    'f': sympy.sin(sympy.pi * x),
                    'g': 0.05 * sympy.sin(8 * sympy.pi * x) * sympy.sin(2 * sympy.pi * t)
                }, sol_hypercube=sol_hypercube, output='plot', ics='external', target='pytorch', loop_iterations=5,
                            dt=0.0002, timeit=True)
            prob_generated_code.write(code)
