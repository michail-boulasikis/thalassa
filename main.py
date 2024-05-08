from lib.sym_pde import StencilCompiler
import sympy
import pprint
import sys
import argparse

discretization_2d = {
    # du/dx ~= 0.005 * (u(t, x + dx, y) - u(t, x - dx, y))
    # dx = dy = dt = 0.01
    "x": {
        "coeffs": [0.005, -0.005],
        "indices": [[0, 1, 0], [0, -1, 0]]
    },
    "y": {
        "coeffs": [0.005, -0.005],
        "indices": [[0, 0, 1], [0, 0, -1]]
    },
    "t": {
        "coeffs": [0.01, -0.01],
        "indices": [[1, 0, 0], [0, 0, 0]]
    },
    "xx": {
        "coeffs": [0.0001, -0.0002, 0.0001],
        "indices": [[0, 1, 0], [0, 0, 0], [0, -1, 0]]
    },
    "yy": {
        "coeffs": [0.0001, -0.0002, 0.0001],
        "indices": [[0, 0, 1], [0, 0, 0], [0, 0, -1]]
    }
}

discretization_1d = {
    # du/dx ~= 0.005 * (u(t, x + dx) - u(t, x))
    # dx = dt = 0.01
    "x": {
        "coeffs": [0.005, -0.005],
        "indices": [[0, 1], [0, -1]]
    },
    "t": {
        "coeffs": [0.01, -0.01],
        "indices": [[1, 0], [0, 0]]
    }
}

discretization_1d_second = {
    # du/dx ~= 0.005 * (u(t, x + dx) - u(t, x))
    # dx = dt = 0.01
    "x": {
        "coeffs": [0.005, -0.005],
        "indices": [[0, 1], [0, -1]]
    },
    "t": {
        "coeffs": [0.01, -0.01],
        "indices": [[1, 0], [0, 0]]
    },
    "xx": {
        "coeffs": [0.0001, -0.0002, 0.0001],
        "indices": [[0, 1], [0, 0], [0, -1]]
    }
}

discretization_1d_second_time = {
    # du/dx ~= 0.005 * (u(t, x + dx) - u(t, x))
    # dx = dt = 0.01
    "x": {
        "coeffs": [0.005, -0.005],
        "indices": [[0, 1], [0, -1]]
    },
    "t": {
        "coeffs": [0.01, -0.01],
        "indices": [[1, 0], [0, 0]]
    },
    "xx": {
        "coeffs": [0.0001, -0.0002, 0.0001],
        "indices": [[0, 1], [0, 0], [0, -1]]
    },
    "tt": {
        "coeffs": [0.0001, -0.0002, 0.0001],
        "indices": [[1, 0], [0, 0], [-1, 0]]
    }
}


free_symbols = list(map(sympy.Symbol, ["t", "x", "y"]))
t, x, y = free_symbols
known_functions = map(sympy.Function, ["f", "g", "h"])
f, g, h = known_functions
unknowns = list(map(sympy.Function, ["u", "v"]))
u, v = unknowns

problems = {
    # Tests linear equations
    # u_t + u_x = 0
    "linear-const-coeff": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t) + sympy.Derivative(u(t, x), x), 0)],
        "unknowns": [u.name],
        "known_functions": [],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d
    },
    # Tests linear equations with non-constant coefficients
    # u_t + f(x) * u_x = 0
    "linear-non-const-coeff": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t) + f(x) * sympy.Derivative(u(t, x), x), 0)],
        "unknowns": [u.name],
        "known_functions": [f.name],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d
    },
    # Tests equations with forcing terms
    # u_t + u_x = g(t, x)
    "linear-with-forcing": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t) + sympy.Derivative(u(t, x), x), g(t, x))],
        "unknowns": [u.name],
        "known_functions": [g.name],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d
    },
    # Tests non-linear equations with eventual discontinuities
    # u_t + u * u_x = 0
    "burgers-1d-shockwave": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t) + u(t, x) * sympy.Derivative(u(t, x), x), 0)],
        "unknowns": [u.name],
        "known_functions": [],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d
    },
    # Tests non-linear equations
    # u_t + u * u_x = 0.2 * u_xx
    "burgers-1d-diffusion": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t) + u(t, x) * sympy.Derivative(u(t, x), x),
                         0.2 * sympy.Derivative(u(t, x), x, x))],
        "unknowns": [u.name],
        "known_functions": [],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d_second
    },
    # Tests systems of equations
    # u_t + u * u_x + v * u_y = 0.2 * u_xx + 0.2 * u_yy
    # v_t + u * v_x + v * v_y = 0.2 * v_xx + 0.2 * v_yy
    "burgers-2d-diffusion": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x, y), t) + u(t, x, y) * sympy.Derivative(u(t, x, y), x)
                         + v(t, x, y) * sympy.Derivative(u(t, x, y), y),
                0.2 * sympy.Derivative(u(t, x, y), x, x) + 0.2 * sympy.Derivative(u(t, x, y), y, y)
            ),
            sympy.Eq(sympy.Derivative(v(t, x, y), t) + u(t, x, y) * sympy.Derivative(v(t, x, y), x)
                         + v(t, x, y) * sympy.Derivative(v(t, x, y), y),
                0.2 * sympy.Derivative(v(t, x, y), x, x) + 0.2 * sympy.Derivative(v(t, x, y), y, y)
            )
        ],
        "unknowns": [u.name, v.name],
        "known_functions": [],
        "free_symbols": [t.name, x.name, y.name],
        "discretization": discretization_2d
    },
    # Tests equations with functional non-linearity in u
    # u_t - u_x = 0.01 * exp(u)
    "kolmogorov–petrovskii–piskunov": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t) - sympy.Derivative(u(t, x), x, x), 0.01 * sympy.exp(u(t, x)))],
        "unknowns": [u.name],
        "known_functions": [],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d_second
    },
    # Tests 2nd order linear equations
    # u_tt - u_xx = 0
    "wave-equation": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t, t) - sympy.Derivative(u(t, x), x, x), 0)],
        "unknowns": [u.name],
        "known_functions": [],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d_second_time
    },
    # Tests equations with functional non-linearity in derivatives of u
    # u_t - u_x = 0.01 * exp(u_x)
    "kpp-deriv-functional-nonlinearity": {
        "eqs": [sympy.Eq(sympy.Derivative(u(t, x), t) - sympy.Derivative(u(t, x), x, x),
                         0.01 * sympy.exp(sympy.Derivative(u(t, x), x)))],
        "unknowns": [u.name],
        "known_functions": [],
        "free_symbols": [t.name, x.name],
        "discretization": discretization_1d_second
    }
}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--problem", type=str, default=None)
    args = args.parse_args()
    if not args.problem:
        for problem_name, problem in problems.items():
            print(f"Compiling {problem_name}...")
            compiler = StencilCompiler(problem["eqs"], problem["unknowns"],
                                       problem["known_functions"], problem["free_symbols"], log=True)
            kernel, instr = compiler.compile(problem["discretization"], focus="memory")
            pprint.pp(kernel)
            pprint.pp(instr)
            print(f"Compiled {problem_name} successfully!")
            print("\n\n")
    else:
        problem = problems[args.problem]
        print(f"Compiling {args.problem}...")
        compiler = StencilCompiler(problem["eqs"], problem["unknowns"],
                                   problem["known_functions"], problem["free_symbols"], log=False)
        kernel, instr = compiler.compile(problem["discretization"], focus="depth")
        pprint.pp(kernel)
        pprint.pp(instr)
        print(f"Compiled {args.problem} successfully!")
