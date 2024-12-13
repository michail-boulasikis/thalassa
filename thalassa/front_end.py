import functools
import pprint

import time
import sympy
import itertools
import copy
import numpy as np

import thalassa
from abc import ABC, abstractmethod


class PDESystem(object):
    def __init__(self, equations, unknowns, free_symbols, name=None):
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
        self.free_functions = [f for eq in equations for f in eq.atoms(sympy.Function) if
                               (f not in unknowns and isinstance(f.__class__, sympy.core.function.UndefinedFunction))]
        if name is None:
            self.name = "PDESystem"
        else:
            self.name = name

    def __str__(self):
        return f"PDESystem({self.equations})"

    def __repr__(self):
        return self.__str__()


class FrontEndPhase(ABC):
    def __init__(self, name, description, frontend, enabled=True):
        self.name = name
        self.description = description
        self.frontend = frontend
        self.enabled = enabled

    def __str__(self):
        return f"FrontEnd phase {self.name} ({self.description})"

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SimplifyDerivativesPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("simplify_derivatives", "Simplify derivatives in the PDE system so that "
                                                 "only derivatives of the form D(u, ...) remain. "
                                                 "An example replacement is D(u*u, x) -> 2*u*D(u, x).",
                         frontend, enabled)

    def __call__(self):
        if self.enabled:
            for i, eq in enumerate(self.frontend.pde_system.equations):
                subs_dict = {}
                for d in eq.atoms(sympy.Derivative):
                    subs_dict[d] = d.doit()
                self.frontend.pde_system.equations[i] = eq.subs(subs_dict)


class SubstituteFreeFunctionsPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True, funcs=None):
        super().__init__("substitute_free_functions",
                         "Substitute free functions in the PDE system with the provided functions.",
                         frontend, enabled)
        self.funcs = funcs

    def __call__(self):
        if self.enabled and self.funcs:
            for func, expr in self.funcs.items():
                # If the expression is not 'external', we assume it is a sympy expression and substitute it in the PDE
                if expr != "external":
                    self.frontend.pde_system.equations = [eq.subs(func, expr) for eq in
                                                          self.frontend.pde_system.equations]
                    # Remove the function from the list of free functions
                    self.frontend.pde_system.free_functions = [f for f in self.frontend.pde_system.free_functions if
                                                               f != func]


class TurnIntoFirstOrderSystemPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True, dt=0.01):
        super().__init__("turn_into_first_order_system",
                         "Turn the PDE system into a first order PDE system in time by introducing new variables",
                         frontend, enabled)
        self.dt = dt

    def __call__(self):
        # Set of all derivatives in the PDE system
        derivs = set.union(*[eq.atoms(sympy.Derivative) for eq in self.frontend.pde_system.equations])
        # Finding the order of each derivative in time
        orders = [sum(z == self.frontend.pde_system.free_symbols[0] for z in d.variables) for d in derivs]
        # Pairing the order with the derivative's expression
        derivs = list(zip(orders, [d.expr for d in derivs]))
        # Grouping the tuples by unknown
        derivs = [list(filter(lambda x: x[1] == unk, derivs)) for unk in self.frontend.pde_system.unknowns]
        # Finding the maximum order of each unknown
        max_order = [max(d, key=lambda x: x[0]) for d in derivs]
        # Filtering the unknowns which have order < 2 (no need to introduce new variables)
        max_order = list(filter(lambda x: x[0] >= 2, max_order))
        # Constructing a dictionary of maximum orders
        max_order = {e[1]: e[0] for e in max_order}
        # Constructing the time discretizations for all already existing time derivatives
        time_discs = [
            thalassa.fdm_simple_partial_derivative(sympy.Derivative(d[0][1], self.frontend.pde_system.free_symbols[0]),
                                                   self.dt,
                                                   method='forward')
            for d in derivs]
        self.frontend.discretizations.extend(time_discs)
        # Introduce new variables ([u, u_1, u_2, ...], [v, v_1, v_2, ...], ...)
        # with u and v already existing unknowns, which we use to construct the dictionary of new variables
        new_vars = []
        for k, v in max_order.items():
            new_var_list = [k]
            for i in range(v - 1):
                new_var_list.append(sympy.Function(f'{k.name}_{i + 1}')(*self.frontend.pde_system.free_symbols))
            new_vars.append(new_var_list)
        new_vars = {e[0]: e for e in new_vars}
        # If we have only first order derivatives, return as is
        if not new_vars:
            return
        # Constructing the new equations which represent the time derivatives of the new variables
        new_eqs = []
        for _, nv in new_vars.items():
            for idx, var in enumerate(nv[1:], start=1):
                new_eqs.append(sympy.Eq(sympy.Derivative(nv[idx - 1], self.frontend.pde_system.free_symbols[0]), var))
        # Equation substitutions with new variables
        for i, eq in enumerate(self.frontend.pde_system.equations):
            substitutions = {}
            # Find the time derivatives in the equation
            eq_time_derivs = []
            for derivative in eq.atoms(sympy.Derivative):
                # If time is in the function's free symbols and the derivative is at least of order 1 in time,
                # then it is a time derivative
                if (self.frontend.pde_system.free_symbols[0] in derivative.variables and
                        sum(var == self.frontend.pde_system.free_symbols[0] for var in derivative.variables) >= 1):
                    eq_time_derivs.append(derivative)
            # Substitute the time derivatives with the new variables
            for d in eq_time_derivs:
                # Special case when the time derivative is of maximum order
                # (replaced with derivative of last new variable)
                if sum(z == self.frontend.pde_system.free_symbols[0] for z in d.variables) == max_order[d.expr]:
                    new_var = sympy.Derivative(new_vars[d.expr][-1], self.frontend.pde_system.free_symbols[0])
                # General case (replaced with new variable)
                else:
                    new_var = new_vars[d.expr][sum(z == self.frontend.pde_system.free_symbols[0] for z in d.variables)]
                substitutions[d] = new_var
            self.frontend.pde_system.equations[i] = eq.subs(substitutions)
        # Constructing the time discretizations for new variables
        time_discs = []
        for _, v in new_vars.items():
            time_discs.extend([thalassa.fdm_simple_partial_derivative(
                sympy.Derivative(k, self.frontend.pde_system.free_symbols[0]), self.dt, method='forward') for k in
                v[1:]])
        # Add new variables to the system in order (u, u_1, u_2, ... , v, v_1, v_2, ...)
        # Order matters, because the order of unknowns is the order of supplied initial conditions
        # in the resultant solver
        self.frontend.pde_system.unknowns = [[k, *v[1:]] for k, v in new_vars.items()]
        # Flatten the list
        self.frontend.pde_system.unknowns = [item for sublist in self.frontend.pde_system.unknowns for item in sublist]
        # Add new equations to the system
        self.frontend.pde_system.equations.extend(new_eqs)
        # Add new discretizations to the system
        self.frontend.discretizations.extend(time_discs)


class VerifyDiscretizationsPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("verify_discretizations",
                         "Verifies that the provided discretizations are correct, "
                         "that all derivatives in the PDE system are covered by a discretization "
                         "and that there are no duplicate discretizations.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        deriv_exprs_disc = [d.derivative_expr for d in self.frontend.discretizations]
        if not all(d.args[0] in self.frontend.pde_system.unknowns for d in deriv_exprs_disc):
            raise ValueError("Discretizations must be based on unknown functions in the PDE system")
        # Make sure that every derivative expression in the PDE system is covered by a discretization
        deriv_exprs_eqs = [deriv_expr for eq in self.frontend.pde_system.equations for deriv_expr in
                           eq.atoms(sympy.Derivative)]
        if not all(deriv_expr in deriv_exprs_disc for deriv_expr in deriv_exprs_eqs):
            raise ValueError("Not all derivative expressions in the PDE system are covered by a discretization")
        # Warn if a discretization is not used in the PDE system
        if not all(deriv_expr in deriv_exprs_eqs for deriv_expr in deriv_exprs_disc):
            print("\033[93mWarning:\033[0m not all discretizations are used in the PDE system. "
                  "Unused discretizations will be ignored.")
            # Filter out unused discretizations
            self.frontend.discretizations = [d for d in self.frontend.discretizations if
                                             d.derivative_expr in deriv_exprs_eqs]
        # Make sure that there is no overlap between discretizations
        for d1, d2 in itertools.combinations(self.frontend.discretizations, 2):
            if d1.derivative_expr == d2.derivative_expr:
                raise ValueError("Discretizations must not overlap.")


class DiscretizePDESystemPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("discretize_pde_system",
                         "Discretizes the PDE system using the provided discretizations.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        # Sort the discretizations by the order of the derivative expression, this is important, otherwise sympy
        # will not be able to substitute the derivatives correctly
        self.frontend.discretizations.sort(key=lambda d: sum(d.order), reverse=True)
        # Discretize the PDE system
        for i, eq in enumerate(self.frontend.pde_system.equations):
            for d in self.frontend.discretizations:
                eq = eq.subs(d.derivative_expr, d.fdm_expr)
            self.frontend.pde_system.equations[i] = eq
        # We also need to discretize any unknowns in the PDE system that are not part of a derivative
        for i, eq in enumerate(self.frontend.pde_system.equations):
            for unk in self.frontend.pde_system.unknowns:
                unk_d = sympy.Indexed(unk.name, *[0] * len(unk.args))
                eq = eq.subs(unk, unk_d)
            self.frontend.pde_system.equations[i] = eq


class SolveForNextTimestepPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("solve_for_next_timestep",
                         "Solves the discretized PDE system for the next timestep.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        # NOTE: This works for all maximum time indices, even though in practice we only have 0 and 1
        # Find the largest time index for each unknown by searching all equations for indexed unknowns
        indexed_vars = []
        for eq in self.frontend.pde_system.equations:
            for idxd in eq.atoms(sympy.Indexed):
                if idxd.base.name in map(lambda x: x.name, self.frontend.pde_system.unknowns):
                    indexed_vars.append(idxd)

        # Determine the maximum time index for each unknown
        max_time_idx = {}
        for u in map(lambda x: x.name, self.frontend.pde_system.unknowns):
            max_time_idx[u] = max((int(idxd.indices[0]) for idxd in indexed_vars if idxd.base.name == u), default=0)

        # Create the variables we need to solve for
        next_time_step_vars = []
        for u in self.frontend.pde_system.unknowns:
            indices = [max_time_idx[u.name]] + [0] * (len(u.args) - 1)
            next_time_step_vars.append(sympy.Indexed(u.name, *indices))

        # Solve for the unknowns' next time step
        solutions = sympy.solve(self.frontend.pde_system.equations, next_time_step_vars, dict=True)
        # Check if we have a unique solution
        if not solutions or len(solutions) > 1:
            raise ValueError(f"The PDE system has {0 if not solutions else len(solutions)} solutions. "
                             f"It might be that the discretization is implicit "
                             f"or that the PDE is too complex for thalassa.")
        self.frontend.solved_pdes = solutions[0]


def _get_hypercube_range(term1, global_imin, global_imax):
    indexed_vars = list(term1.atoms(sympy.Indexed))
    if not indexed_vars:
        return np.array([0] * len(global_imin)), np.array([0] * len(global_imin))
    imin_vars = np.zeros(len(indexed_vars[0].indices[1:]), dtype=int) + 100
    imax_vars = np.zeros(len(indexed_vars[0].indices[1:]), dtype=int) - 100
    for idx in indexed_vars:
        imin_vars = np.minimum(imin_vars, [int(i) for i in idx.indices[1:]])
        imax_vars = np.maximum(imax_vars, [int(i) for i in idx.indices[1:]])
    return global_imin - imin_vars, global_imax - imax_vars


def _center_term(term):
    _, t = term
    idxd = list(t.atoms(sympy.Indexed))
    if not idxd:
        return term
    # Group by base
    idxd_dict = {}
    for i in idxd:
        if i.base not in idxd_dict:
            idxd_dict[i.base] = []
        idxd_dict[i.base].append(i)
    # If there is any base with indices all equal to 0, return the term as is
    if any(all(ii == 0 for ii in i.indices[1:]) for i in idxd):
        return term
    # If for any of the bases, for every indexed variable with index i there is an indexed variable with index -i
    # that means the term is centered, so return the term as is
    for _, v in idxd_dict.items():
        if all(any(all(ii == -jj for ii, jj in zip(i.indices[1:], j.indices[1:])) for j in v) for i in v):
            return term
    # If the term is not centered, we center it
    # We find the index with the smallest manhattan distance from the origin
    # and shift all indices of that variable by that amount
    min_dist = 100
    min_idxd = None
    for i in idxd:
        dist = sum(abs(j) for j in i.indices[1:])
        if dist < min_dist:
            min_dist = dist
            min_idxd = i
    substitutions = {}
    for i in idxd:
        new_idxd = sympy.Indexed(i.base,
                                 *([i.indices[0]] + [j - k for j, k in zip(i.indices[1:], min_idxd.indices[1:])]))
        substitutions[i] = new_idxd
    return term[0], term[1].subs(substitutions)


def _delete_duplicates(terms):
    unique_terms = []
    for term in terms:
        if not any(_fast_equals(term[1], unique_term[1]) for unique_term in unique_terms):
            unique_terms.append(term)
    return unique_terms


def _fast_equals(a, b):
    # If the free symbols are different, the expressions are not equal
    if a.free_symbols != b.free_symbols:
        return False
    # If the expressions are symbolic functions, we compare them directly
    if isinstance(a, sympy.Function) and isinstance(b, sympy.Function):
        return a == b
    # We compare the expressions by substituting random numbers for the free symbols
    # Substitute all functions in the expressions with random numbers
    fs = [x for x in a.atoms(sympy.Function) if isinstance(x.__class__, sympy.core.function.UndefinedFunction)]
    for f in fs:
        n = np.random.random()
        a = a.subs(f, n)
        b = b.subs(f, n)
    vars = list(a.free_symbols)
    # If the expressions are not equal for two random substitutions, they are not equal
    for i in range(2):
        vars_n = np.random.random(len(vars))
        n_a = complex(a.subs({v: n for v, n in zip(vars, vars_n)}))
        n_b = complex(b.subs({v: n for v, n in zip(vars, vars_n)}))
        if not np.allclose(n_a, n_b):
            return False
    # If we haven't returned by now, the expressions need to be compared symbolically to be sure
    diff = a - b
    return sympy.simplify(diff) == 0


class CreateAuxiliaryFeaturesPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True, max_features=4):
        super().__init__("create_auxiliary_features",
                         "Creates auxiliary features for the PDE system. "
                         "This phase is optional and takes a maximum number of features as an argument. "
                         "By default, the maximum number of features is 4.",
                         frontend, enabled)
        self.max_features = max_features

    def __call__(self):
        if not self.enabled:
            return
        if self.frontend.solved_pdes:
            # First we remove any free numbers (e.g. 0.322984)
            # from the solved PDEs and add them as biases to the IR
            self.frontend.ir['biases'] = {}
            for k, v in self.frontend.solved_pdes.items():
                free_numbers = v.as_coeff_Add()
                self.frontend.ir['biases'][k] = free_numbers[0]
                self.frontend.solved_pdes[k] = free_numbers[1]
            # We expand the expressions and get all terms. Then we factor out the constants from each term
            # and sort the terms first by the number of indexed variables in each term and
            # then by the absolute value of the sum of the indices in the indexed variables
            # and finally by the number of atoms in the term.
            # This is done to bring more "general" terms closer to the front,
            # as well as terms which are more "centered".
            funcs = [f for eq in self.frontend.solved_pdes.values() for f in eq.atoms(sympy.Function)]
            unk_eq_terms = [x.as_coeff_Mul() for k, v in self.frontend.solved_pdes.items() for x in
                            sympy.Add.make_args(v.expand().collect(funcs))]
            pure_terms_centered = [_center_term(t) for t in unk_eq_terms]
            pure_terms_centered = _delete_duplicates(pure_terms_centered)
            pure_terms_idxd = [(c, t, len(list(t.atoms(sympy.Indexed)))) for c, t in pure_terms_centered]
            pure_terms_centered = [(c, t) for c, t, *_ in sorted(pure_terms_idxd, key=lambda x: x[2])]
            # We form a 2D array of matches between the terms. If two terms are equal up to a shift in the indices,
            # we store the shift. If a term is not equal to any other term, we store None.
            # The search space for the shifts is determined by the minimum and maximum indices
            # of the indexed variables in the term as well as the limits of the discretization.
            # This step is expected to be the most computationally expensive step, especially if the number of terms
            # is large.
            matches = [None] * len(unk_eq_terms) * len(pure_terms_centered)
            imin, imax = _find_iminmax(self.frontend.discretizations)
            imin = imin[1:]
            imax = imax[1:]
            for i in range(len(unk_eq_terms)):
                for j in range(len(pure_terms_centered)):
                    hypercube_range_min, hypercube_range_max = _get_hypercube_range(unk_eq_terms[i][1], imin, imax)
                    search_range = [r + 1 for r in hypercube_range_max - hypercube_range_min]
                    for nidx in np.ndindex(*search_range):
                        shft = nidx + hypercube_range_min
                        t_idxd = list(unk_eq_terms[i][1].atoms(sympy.Indexed))
                        substitutions = {}
                        for idxd in t_idxd:
                            new_idxd = sympy.Indexed(idxd.base, *(
                                    [idxd.indices[0]] + [i + j for i, j in zip(idxd.indices[1:], shft)]))
                            substitutions[idxd] = new_idxd
                        new_expr = unk_eq_terms[i][1].subs(substitutions, simultaneous=True)
                        if _fast_equals(new_expr, pure_terms_centered[j][1]):
                            matches[i * len(pure_terms_centered) + j] = -shft
                            break
            # Turning the matches into a 2D array
            matches = [matches[i:i + len(pure_terms_centered)] for i in
                       range(0, len(matches), len(pure_terms_centered))]
            matches = [next(item for item in enumerate(m) if item[1] is not None) for m in matches]
            matches = [(pure_terms_centered[i], j) for i, j in matches]
            # We now create the features. We start by creating a dictionary of features for each unknown.
            # We then iterate over the terms and match them with other terms. If a term is not matched with any
            # other term, we create a feature for it and loop over the array of terms to match it.
            # If a term is matched with other terms, we continue.
            feature_list = list({f for (_, f), _ in matches})
            subs_dict = {}
            expr_match_pairs = list(zip(list(range(len(unk_eq_terms))), unk_eq_terms, matches))
            for j, (coeff_term, term_expr), ((pure_coeff, pure_term), offset) in expr_match_pairs:
                feature_base = sympy.IndexedBase(f"__FEATURE_{feature_list.index(pure_term)}")
                offsets = offset.tolist()
                subs_dict[term_expr] = feature_base[[0] + offsets]
            # Needed, otherwise substitution will not work
            # for k, v in self.frontend.solved_pdes.items():
            #     self.frontend.solved_pdes[k] = sympy.nsimplify(v)
            # We substitute the features in the solved PDEs

            self.frontend.ir = {
                'biases': self.frontend.ir.get('biases', {}),
                'features': {sympy.IndexedBase(f"__FEATURE_{i}"): [i, v] for i, v in enumerate(feature_list)},
                'eqs': {k: v.expand().collect(funcs).subs(subs_dict, simultaneous=True) for k, v in self.frontend.solved_pdes.items()},
            }


def _find_iminmax(disc):
    imin = np.zeros(len(disc[0].order), dtype=int) + 100
    imax = np.zeros(len(disc[0].order), dtype=int) - 100
    for d in disc:
        imin = np.minimum(imin, d.imin)
        imax = np.maximum(imax, d.imax)
    return imin, imax


class ComputeStencilCoefficients(FrontEndPhase):
    def __init__(self, frontend, enabled):
        super().__init__("compute_stencil_coefficients",
                         "Computes the coefficients for the stencil.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        if self.frontend.ir:
            imin, imax = _find_iminmax(self.frontend.discretizations)
            kernel_slice_shape = [imax[i] - imin[i] + 1 for i in range(len(imin))]
            kernel_slice_shape = kernel_slice_shape[1:]
            n_total_features = len(self.frontend.ir['features'])
            n_unknowns = len(self.frontend.pde_system.unknowns)
            stencil = np.zeros(shape=(n_unknowns, n_total_features, *kernel_slice_shape), dtype=np.float64)
            for unk_idx, (unk, expr) in enumerate(self.frontend.ir['eqs'].items()):
                ts = sympy.Add.make_args(expr.expand())
                for t in ts:
                    multiplicative_constant = t.as_coeff_Mul()[0] if len(t.as_coeff_Mul()) > 1 else 1
                    feature = t.atoms(sympy.Indexed).pop()
                    feature_number = self.frontend.ir['features'][feature.base][0]
                    feature_idx = (unk_idx, feature_number) + tuple(
                        [i - j for i, j in zip(feature.indices[1:], imin[1:])])
                    stencil[feature_idx] += multiplicative_constant
            self.frontend.ir['stencil'] = stencil
            del self.frontend.ir['eqs']


class PrepareIRPhase(FrontEndPhase):
    def __init__(self, frontend, enabled):
        super().__init__("prepare_ir",
                         "Prepares the intermediate representation for the back end.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        if self.frontend.ir:
            imin, imax = _find_iminmax(self.frontend.discretizations)
            imin = imin[1:]
            imax = imax[1:]
            self.frontend.ir = {
                'pdes': self.frontend.pde_system,
                'features': self.frontend.ir['features'],
                'stencil': self.frontend.ir['stencil'],
                'biases': self.frontend.ir['biases'],
                'hypercube': (imax - imin).tolist(),
            }


class PDEFrontEnd(object):

    def __init__(self, pde_system, discretizations, **kwargs):
        self.pde_system = copy.deepcopy(pde_system)
        self.pde_system_unchanged = copy.deepcopy(pde_system)
        if not all(isinstance(d, thalassa.discretization.FDApproximation) for d in discretizations):
            raise ValueError("Discretizations must be FDApproximation objects")
        if not discretizations:
            raise ValueError("At least one discretization must be provided")
        self.discretizations = discretizations
        self.funcs = kwargs.get('funcs', None)
        self.solved_pdes = None
        self.shift_time_steps = False
        self.kwargs = kwargs
        self.ir = {}
        self.phases = [
            SimplifyDerivativesPhase(self, True),
            SubstituteFreeFunctionsPhase(self, True, funcs=self.funcs),
            TurnIntoFirstOrderSystemPhase(self, True, kwargs.get('dt', 0.01)),
            VerifyDiscretizationsPhase(self, True),
            DiscretizePDESystemPhase(self, True),
            SolveForNextTimestepPhase(self, True),
            CreateAuxiliaryFeaturesPhase(self, True, kwargs.get('max_features', 3)),
            ComputeStencilCoefficients(self, True),
            PrepareIRPhase(self, True)
        ]

    def execute(self):
        timing = {}
        for phase in self.phases:
            t_start = time.time()
            phase()
            t_end = time.time()
            timing[phase.name] = t_end - t_start
        timing['total'] = sum(timing.values())
        if self.kwargs.get('timeit', False):
            import pandas as pd
            timing['num_terms'] = len(sympy.Add.make_args(sum(self.solved_pdes.values())))
            pd.DataFrame(timing, index=[0]).to_csv("generated-data/" + self.ir['pdes'].name + '.timing.frontend.csv')
        return self.ir

    def __str__(self):
        phase_str = '\n'.join([str(p) for p in self.phases])
        return f"PDEFrontEnd with phases:\n{phase_str}\n...and PDE system {self.pde_system}"
