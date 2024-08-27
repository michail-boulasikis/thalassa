import sympy

import thalassa
from sympy import Eq, Function, symbols
from sympy import Derivative as D

t, x = symbols('t x')
u = Function('u')(t, x)
v = Function('v')(t, x)

pde = thalassa.PDESystem(
    [Eq(D(u, t), D(u, x, x))],
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
    max_order = {e[1]: e[0] for e in max_order}
    # Introduce new variables
    new_vars = [[k, *[sympy.Function(f'{k.name}_{i + 1}')(*sys.free_symbols) for i in range(v - 1)]] for k, v in
                max_order.items()]
    # If we have only first order derivatives, return the system as is
    if not new_vars:
        return sys
    new_vars = {e[0]: e for e in new_vars}
    new_eqs = [sympy.Eq(sympy.Derivative(nv[idx - 1], sys.free_symbols[0]), var) for _, nv in new_vars.items() for
               idx, var in enumerate(nv) if idx > 0]
    # Equation substitutions with new variables
    for i, eq in enumerate(sys.equations):
        # Find the time derivatives in the equation
        substitutions = {}
        eq_time_derivs = list(filter(
            lambda x: sys.free_symbols[0] in x.variables
                      and sum(z == sys.free_symbols[0] for z in x.variables) >= 1,
            eq.atoms(sympy.Derivative)))
        for d in eq_time_derivs:
            # Special case when the time derivative is of maximum order (replaced with derivative of last new variable)
            if sum(z == sys.free_symbols[0] for z in d.variables) == max_order[d.expr]:
                new_var = sympy.Derivative(new_vars[d.expr][-1], sys.free_symbols[0])
            # General case (replaced with new variable)
            else:
                new_var = new_vars[d.expr][sum(z == sys.free_symbols[0] for z in d.variables)]
            substitutions[d] = new_var
        sys.equations[i] = eq.subs(substitutions)
    # Add new variables to the system in order (u, u_1, u_2, ... , v, v_1, v_2, ...)
    # This is necessary because the order of unknowns is the order of supplied initial conditions
    sys.unknowns = [[k, *v[1:]] for k, v in new_vars.items()]
    # Flatten the list
    sys.unknowns = [item for sublist in sys.unknowns for item in sublist]
    # Add new equations to the system
    sys.equations.extend(new_eqs)
    return sys


def sys_first_order_disc(sys, disc):
    pass


s = sys_first_order(pde)
print(f"{s.unknowns=}\n{s.equations=}")
