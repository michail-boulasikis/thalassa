from thalassa import pde_compile, fdm_simple_partial_derivative
import thalassa
from sympy import Eq
from sympy import Derivative as D
import sympy
import sys
import argparse

t, x, y = sympy.symbols('t x y')
u, v = sympy.Function('u'), sympy.Function('v')
f, g, h = sympy.Function('f'), sympy.Function('g'), sympy.Function('h')

probs = [
    {
        "name": "transport",
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + 2.0 * D(u(t, x), x), 0)], [u(t, x)], [t, x]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "name": "transport-non-const",
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + f(x) * D(u(t, x), x), 0)], [u(t, x)], [t, x]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "name": "transport-forcing",
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + 2.0 * D(u(t, x), x), g(t, x))], [u(t, x)], [t, x]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "name": "burgers-1d",
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + u(t, x) * D(u(t, x), x), 0)], [u(t, x)], [t, x]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central')
        ]
    },
    {
        "name": "burgers-1d-diffusion",
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) + u(t, x) * D(u(t, x), x), 0.2 * D(u(t, x), (x, 2)))], [u(t, x)], [t, x]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')
        ]
    },
    {
        "name": "kolmogorov–petrovskii–piskunov",
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) - D(u(t, x), x, x), 0.01 * sympy.exp(u(t, x)))], [u(t, x)], [t, x]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')
        ]
    },
    {
        "name": "kpp-deriv-exp",
        "pde": thalassa.PDESystem([Eq(D(u(t, x), t) - D(u(t, x), x, x), 0.01 * sympy.exp(D(u(t, x), x)))], [u(t, x)], [t, x]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x), (x, 2)), 0.05, method='central')
        ]
    },
    {
        "name": "burgers-2d",
        "pde": thalassa.PDESystem([
            Eq(D(u(t, x, y), t) + u(t, x, y) * D(u(t, x, y), x) + v(t, x, y) * D(u(t, x, y), y), 0),
            Eq(D(v(t, x, y), t) + v(t, x, y) * D(v(t, x, y), y) + u(t, x, y) * D(v(t, x, y), x), 0)
        ], [u(t, x, y), v(t, x, y)], [t, x, y]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), y), 0.05, method='central'),
        ]
    },
    {
        "name": "burgers-2d-diffusion",
        "pde":  thalassa.PDESystem([
            Eq(D(u(t, x, y), t) + u(t, x, y) * D(u(t, x, y), x) + v(t, x, y) * D(u(t, x, y), y), 0.02 * (D(u(t, x, y), (x, 2)) + D(u(t, x, y), (y, 2)))),
            Eq(D(v(t, x, y), t) + v(t, x, y) * D(v(t, x, y), y) + u(t, x, y) * D(v(t, x, y), x), 0.02 * (D(v(t, x, y), (x, 2)) + D(v(t, x, y), (y, 2))))
        ], [u(t, x, y), v(t, x, y)], [t, x, y]),
        "disc": [
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(u(t, x, y), (y, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), t), 0.001, method='forward'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), x), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), y), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (x, 2)), 0.05, method='central'),
            thalassa.fdm_simple_partial_derivative(D(v(t, x, y), (y, 2)), 0.05, method='central'),
        ]
    }
]

if __name__ == '__main__':
    problem_dir = "generated-code"
    for prob in probs:
        file_name = f"{problem_dir}/{prob['name'].replace('-', '_')}.py"
        with open(file_name, 'w') as prob_generated_code:
            prob_generated_code.write(
                pde_compile(prob["pde"], prob["disc"], funcs={
                    'f': sympy.sin(sympy.pi * x),
                    'g': sympy.sin(8 * sympy.pi * x) * sympy.sin(2 * sympy.pi * t)
                })
            )
