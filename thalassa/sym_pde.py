import sympy
import numpy as np
import functools


class PDESystem(object):
    def __init__(self, equations, unknowns, free_symbols):
        if not equations or not unknowns or not free_symbols:
            raise ValueError("Equations, unknowns, and free symbols must be provided")
        if not all(isinstance(eq, sympy.Eq) for eq in equations):
            raise ValueError("Equations must be sympy equations")
        self.equations = equations
        if not all(isinstance(u, sympy.Function) for u in unknowns):
            raise ValueError("Unknowns must be sympy functions")
        if not all(isinstance(x, sympy.Symbol) for x in free_symbols):
            raise ValueError("Free symbols (i.e., variables) must be sympy symbols (e.g., t, x, y)")
        self.unknowns = unknowns
        self.free_symbols = free_symbols
        self.free_functions = [f for eq in equations for f in eq.atoms(sympy.Function) if f not in unknowns]

    def __str__(self):
        return f"PDESystem({self.equations}, {self.unknowns}, {self.free_symbols})"
