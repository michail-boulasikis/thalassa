import sympy
import numpy as np

import functools


class StencilCompiler(object):

    def __init__(self, eq, methods=None, unknown="u"):
        # 1: Check if the equation is a string and convert it to a sympy expression
        if type(eq) == str:
            eq = sympy.sympify(eq)
        self.eq = eq
        # 2: Find all symbols in the equation
        self.eq_symbols = {s.name: s for s in eq.atoms(sympy.Symbol, sympy.Function)}
        # 3: Find the unknown function in the equation
        self.unknown = self.eq_symbols.get(unknown)
        # 4: Error handling
        if not self.unknown:
            raise ValueError(f"Unknown function '{unknown}' not found in equation '{eq}'")
        # Check if the unknown is a function
        if not isinstance(self.unknown, sympy.Function):
            raise ValueError(f"Unknown function '{unknown}' is not a function in equation '{eq}'")
        # 5: Assign the discretization methods
        self.methods = methods

    def compile(self, methods=None):
        # 1: Check if new discretization methods are provided
        if methods:
            self.methods = methods
        if not self.methods:
            raise ValueError(f"No discretization methods provided for discretizing equation '{self.eq}'")
        # 2: Compile the stencil for the equation
        # 2.1: Discretize the equation
        ubase, idx = self._discretize()
        # 2.2: Solve for the next time step (if possible) (expression for the next time step is stored in self.eq)
        self._solve_for_next_timestep(ubase, idx)
        # 2.3: Perform optimizations to reduce number of auxiliary features
        self._common_subexpression_elimination()
        # 2.4: Compute the stencil and the instructions for computing the auxiliary features
        return self._compute_stencil(ubase, idx)

    def _discretize(self):
        # 1: Find the minimum and maximum indices for each variable
        #    (these indices determine the offset in the tensors we are looking at when doing the convolution)
        indices = []
        for _, data in self.methods.items():
            indices.extend(data["indices"])
        minimums = np.min(indices, axis=0)
        maximums = np.max(indices, axis=0)
        # Here we create the indexed base and the indexed variables for the unknown function
        # In sympy an indexed expression is as follows: Indexed(IndexedBase, Index1, Index2, ...)
        # 2: Create the indexed base for the unknown function
        ubase = sympy.IndexedBase(self.unknown.name)
        # 3: Create the index variables for the unknown function
        indices = [sympy.Idx(f"i_{i}", (m, M)) for i, (m, M) in enumerate(zip(minimums, maximums))]
        # 4: Replace the derivatives with the FDM expressions
        for method, data in self.methods.items():
            # Create the derivative's FDM expression
            deriv = sympy.sympify(f"Derivative({self.unknown}, {','.join(method)})")
            # Replace the derivative with the FDM expression
            fdm_expr = sum(data["coeffs"][i] * ubase[idx] for i, idx in enumerate(data["indices"]))
            # Substitution happens here
            self.eq = self.eq.subs(deriv, fdm_expr)
        # 5: Replace the unknown function with the indexed base
        zero_index = [0 for _ in range(len(indices))]
        self.eq = self.eq.subs(self.unknown, ubase[zero_index])
        return ubase, indices

    def _solve_for_next_timestep(self, ubase, indices):
        # 1: Define the index of the next time step
        idx_next_timestep = [1, *[0 for _ in range(len(indices) - 1)]]
        # 2: Try to solve the equation for the next time step
        self.eq = sympy.solve(self.eq, ubase[idx_next_timestep])
        # 3: Error handling
        if not self.eq:
            raise ValueError(
                f"Failed to solve equation '{self.eq}' for the next time step, method might be implicit or equation"
                f" might be too complex to solve analytically.")
        if len(self.eq) > 1:
            raise ValueError(f"Multiple solutions found for the next time step: {self.eq}")
        # 4: Remove the list and keep the expression
        #    (sympy.solve returns a list of solutions, but we only expect one solution)
        #    From that point on, self.eq holds an expression for the next time step, not the equation (refactor?)
        self.eq = self.eq[0]

    def _common_subexpression_elimination(self):
        # TODO: write optimizations here
        self.eq = self.eq.expand()

    def _compute_stencil(self, ubase, indices):
        # 1: Classify the terms in the equation, depending on whether they end up as coefficients or auxiliary features
        term_classes = self._classify_terms()
        # 2: Compute the stencil shape
        stencil = StencilCompiler._compute_stencil_shape(term_classes, indices)
        # 3: Collect the coefficients and auxiliary features for the stencil.
        #    This step also generates the instructions for computing the auxiliary features
        stencil, instr = StencilCompiler._collect_coefficients(term_classes, indices, stencil)
        return stencil, instr

    def _classify_terms(self):
        term_classes = []
        for term in sympy.Add.make_args(self.eq):
            # Extract the constant factor from the term, used as a coefficient in the stencil
            c, term = StencilCompiler._extract_constant(term)
            # Classify the term as a coefficient or an auxiliary feature
            term_classes.append((c, term, StencilCompiler._classify_term(term)))
        return term_classes

    @staticmethod
    def _extract_constant(term):
        # 1: Find all the constant factors in the term
        consts = [_x for _x in term.as_ordered_factors() if _x.is_constant() and not isinstance(_x, sympy.Indexed)]
        # 2: Multiply them together
        #    Note, the or statement is used to provide a default value in case the list is empty
        c = functools.reduce(lambda x1, x2: x1 * x2, consts or [1.0])
        # 3: Simplify the term by dividing it by the constant factor
        term = sympy.nsimplify(term / c)
        return c, term

    @staticmethod
    def _classify_term(term):
        # 1. Find all the u-terms in the expression
        us = term.atoms(sympy.Indexed)
        if len(us) != 1:
            # If there are 0 or multiple u-terms, the term is an auxiliary feature
            # The case of 0 u-terms corresponds to a forcing term in the PDE
            return "auxiliary"
        u = us.pop()
        # 2. Differentiate the term with respect to the u-term to determine if it is of the form c*u.
        #    If it is, the term will contribute to a coefficient, otherwise it will be described as an auxiliary feature
        return "coefficient" if term.diff(u).is_constant() else "auxiliary"

    @staticmethod
    def _compute_stencil_shape(term_classes, indices):
        # 1. Count the number of auxiliary features
        n_aux = sum(1 for _, _, c in term_classes if c == "auxiliary")
        # 2. Compute the stencil shape
        n_stencil = np.array([i.upper - i.lower + 1 for i in indices])
        n_stencil = np.roll(n_stencil, -1)
        # 3. Add the number of auxiliary features to the last dimension of the stencil, "behind" the time steps
        n_stencil[-1] += n_aux - 1
        return np.zeros(shape=n_stencil)

    @staticmethod
    def _collect_coefficients(term_classes, indices, stencil):
        aux_count = indices[0].upper - indices[0].lower
        instr = {}
        for const, term, termclass in term_classes:
            # Construct the stencil and the instructions for computing the auxiliary features
            if termclass == "coefficient":
                stencil, instr = StencilCompiler._handle_coefficient_term(const, term, indices, stencil, instr)
            elif termclass == "auxiliary":
                stencil, instr, aux_count = StencilCompiler._handle_auxiliary_term(const, term, indices, stencil, instr,
                                                                                   aux_count)
        return stencil, instr

    @staticmethod
    def _handle_coefficient_term(const, term, indices, stencil, instr):
        _u = term.atoms(sympy.Indexed).pop()
        stencil_idx = [int(_u.indices[n]) - idx.lower for n, idx in enumerate(indices)]
        stencil_idx = np.roll(stencil_idx, -1)
        instr[stencil_idx[-1]] = term
        stencil[tuple(stencil_idx)] += const
        return stencil, instr

    @staticmethod
    def _handle_auxiliary_term(const, term, indices, stencil, instr, aux_count):
        stencil_idx = [int((idx.upper - idx.lower) / 2) for idx in indices]
        stencil_idx = np.roll(stencil_idx, -1)
        stencil_idx[-1] += aux_count
        instr[stencil_idx[-1]] = term
        stencil[tuple(stencil_idx)] += const
        aux_count += 1
        return stencil, instr, aux_count

