import functools
import pprint

import sympy
import itertools
import copy

import thalassa
from abc import ABC, abstractmethod


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
            self.frontend.pde_system.equations = [eq.simplify() for eq in self.frontend.pde_system.equations]


class TurnIntoFirstOrderSystemPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True, dt=0.01):
        super().__init__("turn_into_first_order_system",
                         "Turn the PDE system into a first order PDE system in time by introducing new variables",
                         frontend, enabled)
        self.dt = dt

    def __call__(self):
        derivs = [elem for elems in
                  list(map(lambda x: [(sum(z == self.frontend.pde_system.free_symbols[0] for z in y.variables), y.expr)
                                      for y in x],
                           [eq.atoms(sympy.Derivative) for eq in self.frontend.pde_system.equations])) for elem in
                  elems]
        # Grouping the tuples by unknown
        derivs = [list(filter(lambda x: x[1] == unk, derivs)) for unk in self.frontend.pde_system.unknowns]
        # Finding the maximum order of each unknown
        max_order = [max(d, key=lambda x: x[0]) for d in derivs]
        # Filtering the unknowns which have order < 2
        max_order = list(filter(lambda x: x[0] >= 2, max_order))
        max_order = {e[1]: e[0] for e in max_order}
        # Introduce new variables
        new_vars = [
            [k, *[sympy.Function(f'{k.name}_{i + 1}')(*self.frontend.pde_system.free_symbols) for i in range(v - 1)]]
            for k, v in
            max_order.items()]
        # If we have only first order derivatives, return as is
        if not new_vars:
            return
        new_vars = {e[0]: e for e in new_vars}
        new_eqs = [sympy.Eq(sympy.Derivative(nv[idx - 1], self.frontend.pde_system.free_symbols[0]), var) for _, nv in
                   new_vars.items() for
                   idx, var in enumerate(nv) if idx > 0]
        # Equation substitutions with new variables
        for i, eq in enumerate(self.frontend.pde_system.equations):
            # Find the time derivatives in the equation
            substitutions = {}
            eq_time_derivs = list(filter(
                lambda x: self.frontend.pde_system.free_symbols[0] in x.variables
                          and sum(z == self.frontend.pde_system.free_symbols[0] for z in x.variables) >= 1,
                eq.atoms(sympy.Derivative)))
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
        # Constructing the time discretizations for all variables
        time_discs = [
            thalassa.fdm_simple_partial_derivative(sympy.Derivative(k, self.frontend.pde_system.free_symbols[0]), self.dt,
                                          method='forward')
            for k in new_vars.keys()]
        for _, v in new_vars.items():
            time_discs.extend([thalassa.fdm_simple_partial_derivative(
                sympy.Derivative(k, self.frontend.pde_system.free_symbols[0]), self.dt, method='forward') for k in
                               v[1:]])
        # Add new variables to the system in order (u, u_1, u_2, ... , v, v_1, v_2, ...)
        # This is necessary because the order of unknowns is the order of supplied initial conditions
        self.frontend.pde_system.unknowns = [[k, *v[1:]] for k, v in new_vars.items()]
        # Flatten the list
        self.frontend.pde_system.unknowns = [item for sublist in self.frontend.pde_system.unknowns for item in sublist]
        # Add new equations to the system
        self.frontend.pde_system.equations.extend(new_eqs)
        print(f"{self.frontend.pde_system.equations=}")
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
        # Solve the PDE system for the next timestep
        # Find the largest time index for each unknown by searching all equations for indexed unknowns and taking the
        # maximum time index for each unknown
        indexed_vars = [idxd for eq in self.frontend.pde_system.equations for idxd in eq.atoms(sympy.Indexed)
                        if idxd.base.name in map(lambda x: x.name, self.frontend.pde_system.unknowns)]
        max_time_idx = {u: max([int(idxd.indices[0]) for idxd in indexed_vars if idxd.base.name == u], default=0)
                        for u in map(lambda x: x.name, self.frontend.pde_system.unknowns)}
        if any(max_time_idx[u.name] > 1 for u in self.frontend.pde_system.unknowns):
            print("\033[93mWarning:\033[0m Next time step has an index greater than 1.")
        # Create the variables we need to solve for
        next_time_step_vars = [sympy.Indexed(u.name, *([max_time_idx[u.name]] + [0] * (len(u.args) - 1))) for u in
                               self.frontend.pde_system.unknowns]
        solutions = sympy.solve(self.frontend.pde_system.equations, next_time_step_vars, dict=True)
        # Check if we have a unique solution
        if not solutions or len(solutions) > 1:
            raise ValueError(f"The PDE system has {0 if not solutions else len(solutions)} solutions. "
                             f"It might be that the discretization is implicit "
                             f"or that the PDE is too complex for thalassa.")
        self.frontend.solved_pdes = solutions[0]
        print(f"{self.frontend.solved_pdes=}")


class RuleBasedExpressionOptimizationPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("rule_based_expression_optimization",
                         "Optimizes the expressions for the next time steps using a set of rules, "
                         "which is expected to reduce their computational overhead. "
                         "Example applied rule: exp(x)*exp(y) -> exp(x + y). "
                         "This phase is optional.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        if self.frontend.solved_pdes:
            self.frontend.solved_pdes = {k: v.simplify() for k, v in self.frontend.solved_pdes.items()}


class SplitConstNonConstPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("split_const_non_const",
                         "Splits the expressions for the next time steps into constant and non-constant parts. "
                         "Constant parts are going to be handled with convolution. "
                         "Non-constant parts will become auxiliary features. "
                         "If this phase is disabled, all expressions will be treated as non-constant.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            self.frontend.ir = {k: (0, v) for k, v in self.frontend.solved_pdes.items()}
            return
        if self.frontend.solved_pdes:
            # Generate an intermediate representation (IR) of the PDE system
            parts = {}
            for k, v in self.frontend.solved_pdes.items():
                expr_terms = sympy.Add.make_args(v.expand())
                expr_terms_no_indices = [
                    term.replace(lambda e: isinstance(e, sympy.Indexed), lambda e: sympy.Dummy(k.base.name))
                    for term in expr_terms]
                non_const_terms = [sympy.Poly(e).total_degree() > 1 or len(e.atoms(sympy.Function)) > 0 for e in
                                   expr_terms_no_indices]
                parts[k] = [
                    sum([t for i, t in enumerate(expr_terms) if not non_const_terms[i]]),
                    sum([t for i, t in enumerate(expr_terms) if non_const_terms[i]]) if any(
                        nl for nl in non_const_terms) else 0
                ]
            self.frontend.ir = parts


class SplitAuxiliaryFeaturesPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True, split_strategy='memory'):
        super().__init__("split_auxiliary_features",
                         "Splits the auxiliary features of the PDE system, "
                         "which are the non-constant parts of the expressions for the next time steps. "
                         "This phase is optional, and takes a split strategy as an argument. "
                         "By default, the strategy is 'memory', which means that the auxiliary features are "
                         "kept as they are. "
                         "If the strategy is 'depth', the auxiliary features are split as much as possible.",
                         frontend, enabled)
        self.split_strategy = split_strategy

    def __call__(self):
        if not self.enabled:
            return
        if self.frontend.ir:
            match self.split_strategy:
                case 'memory':
                    pass
                case 'depth':
                    for k, v in self.frontend.ir.items():
                        self.frontend.ir[k] = (v[0], [t.simplify() for t in sympy.Add.make_args(v[1])])
                case _:
                    raise ValueError("Invalid optimization strategy. "
                                     "Valid strategies are 'memory' and 'depth'.")


def _pull_most_common_factor(expr):
    consts = [
        [c for c in t.as_ordered_factors() if c.is_constant() and not isinstance(c, (sympy.Indexed, sympy.Function))]
        for t in sympy.Add.make_args(expr)]
    consts = [functools.reduce(lambda x, y: x * y, cs or [1.0]) for cs in consts]
    c = max(set(consts), key=consts.count)
    return c


class FactorAuxiliaryFeaturesPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("factor_auxiliary_features",
                         "Pulls the most common factor out of the auxiliary features and normalizes the features. "
                         "This saves some multiplications during the convolution. "
                         "If there is no constant part in the expressions for the next time step, this phase is a NOP. "
                         "This phase is optional.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        if self.frontend.ir:
            for k, v in self.frontend.ir.items():
                aux_features = v[1]
                if aux_features:
                    if type(aux_features) is list:
                        for i, t in enumerate(aux_features):
                            c = _pull_most_common_factor(t.expand())
                            aux_features[i] = [c, sympy.nsimplify(t / c)]
                    else:
                        c = _pull_most_common_factor(v[1].expand())
                        aux_features = [[c, sympy.nsimplify(v[1] / c)]]
                    self.frontend.ir[k] = [v[0], aux_features]


class PrepareOutputsPhase(FrontEndPhase):
    def __init__(self, frontend, enabled=True):
        super().__init__("prepare_outputs", "Prepares and creates output products.",
                         frontend, enabled)

    def __call__(self):
        if not self.enabled:
            return
        if self.frontend.ir:
            outputs = {'pdes': self.frontend.pde_system, 'disc': self.frontend.discretizations, 'instr': {
                k.base.name: {
                    'conv': v[0],
                    'aux': [{
                        'c': aux[0],
                        'expr': aux[1]
                    } for aux in v[1]] if type(v[1]) is list else {}
                } for k, v in self.frontend.ir.items()
            }}
            orders = [
                (u.name, 1.0, sympy.Indexed(u.name, *([i] + [0] * len(u.args[1:]))))
                for u in self.frontend.pde_system.unknowns
                for i in list(range(0, -max([d.order[0] for d in outputs['disc'] if d.base_func.name == u.name]), -1))
            ]
            orders.extend([
                (k, aux['c'], aux['expr']) for k, v in outputs['instr'].items() for aux in v['aux']
            ])
            outputs['order'] = orders
            self.frontend.ir = outputs


class PDEFrontEnd(object):

    def __init__(self, pde_system, discretizations, **kwargs):
        self.pde_system = copy.deepcopy(pde_system)
        self.pde_system_unchanged = copy.deepcopy(pde_system)
        if not all(isinstance(d, thalassa.discretization.FDApproximation) for d in discretizations):
            raise ValueError("Discretizations must be FDApproximation objects")
        if not discretizations:
            raise ValueError("At least one discretization must be provided")
        self.discretizations = discretizations
        self.solved_pdes = None
        self.shift_time_steps = False
        self.ir = None
        self.phases = [
            SimplifyDerivativesPhase(self, True),
            TurnIntoFirstOrderSystemPhase(self, True, kwargs.get('dt', 0.01)),
            VerifyDiscretizationsPhase(self, True),
            DiscretizePDESystemPhase(self, True),
            SolveForNextTimestepPhase(self, True),
            RuleBasedExpressionOptimizationPhase(self, True),
            SplitConstNonConstPhase(self, kwargs.get('use_conv', True)),
            SplitAuxiliaryFeaturesPhase(self, True, kwargs.get('aux_split_strategy', 'memory')),
            FactorAuxiliaryFeaturesPhase(self, kwargs.get('pull_factors', True)),
            PrepareOutputsPhase(self, True)
        ]

    def execute(self):
        [phase() for phase in self.phases]
        return self.ir

    def __str__(self):
        phase_str = '\n'.join([str(p) for p in self.phases])
        return f"PDEFrontEnd with phases:\n{phase_str}\n...and PDE system {self.pde_system}"
