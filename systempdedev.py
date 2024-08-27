import sympy

import thalassa
from sympy import Eq, Function, symbols
from sympy import Derivative as D

t, x = symbols('t x')
u = Function('u')(t, x)
v = Function('v')(t, x)

pde = thalassa.PDESystem(
    [Eq(D(u, t, 3) - D(u, x, 2), 0)],
    [u],
    [t, x]
)


def sys_first_order(sys):
    # List of (time_diff_order, unknown) elements
    derivs = [elem for elems in
              list(map(lambda x: [(sum(z == sys.free_symbols[0] for z in y.variables), y.expr) for y in x],
                       [eq.atoms(sympy.Derivative) for eq in sys.equations])) for elem in elems]
    # Grouping the tuples by unknown
    derivs = [list(filter(lambda x: x[1] == unk, derivs)) for unk in sys.unknowns]
    # Finding the maximum order of each unknown
    max_order = [max(d, key=lambda x: x[0]) for d in derivs]
    # Filtering the unknowns which have order < 2
    max_order = list(filter(lambda x: x[0] >= 2, max_order))
    print(max_order)
    # Introduce new variables
    new_vars = [[e[1], *[sympy.Function(f'{e[1].name}_{i + 1}')(*sys.free_symbols) for i in range(e[0] - 1)]] for e in max_order]
    new_eqs = [sympy.Eq(var, sympy.Derivative(nv[idx-1], sys.free_symbols[0])) for nv in new_vars for idx, var in enumerate(nv) if idx > 0]
    print(new_eqs)

sys_first_order(pde)
