import sympy
import numpy as np
import functools


class StencilCompiler(object):

    _stencil = None
    _instructions = None

    def __init__(self, equations, unknowns, known_functions, free_symbols, log=False, target="pytorch"):
        if not equations or not unknowns or not free_symbols:
            raise ValueError("Equations, unknowns, and free symbols must be provided")
        if not isinstance(equations[0], sympy.Eq):
            raise ValueError("Equations must be sympy equations")
        self.equations = equations
        # Check whether arrays contain functions or symbols
        if not all(isinstance(x, (sympy.Function, str)) for x in unknowns) \
                and not all(isinstance(x, (sympy.Function, str)) for x in known_functions) \
                and not all(isinstance(x, (sympy.Symbol, str)) for x in free_symbols):
            raise ValueError(
                "Unknowns, known functions, and free symbols must be sympy functions or symbols or strings")
        # We turn the unknowns, known functions, and free symbols into strings
        # otherwise it's hard to actually use them since the equation does not contain the same object references
        self.unknowns = [str(x) if not isinstance(x, str) else x for x in unknowns]
        self.known_functions = [str(x) if not isinstance(x, str) else x for x in known_functions]
        self.free_symbols = [str(x) if not isinstance(x, str) else x for x in free_symbols]
        # Check if the number of equations is more or equal to the number of unknowns
        if len(self.equations) < len(self.unknowns):
            raise ValueError("The number of equations must be greater or equal to the number of unknowns")
        self.log = log
        if target.lower() not in ["pytorch", "tf", "mlir"]:
            raise ValueError("Target must be one of 'pytorch', 'tf', 'mlir'")
        self.target = target.lower()

    def compile(self, discretization, focus="depth"):
        if self.log:
            print(f"\033[96mEquations:\033[0m {self.equations}")
        # 1. Discretization of the unknowns in the equations.
        #    We replace the derivatives of the unknowns with the supplied FDMs.
        #    We also compute the discretization indices of each dimension,
        #    which will be used to compute the stencil shape.
        d_equations = self._discretize_unknowns(discretization)
        if self.log:
            print(f"\033[96mDiscretized equations:\033[0m {d_equations}")
        d_ind = self._compute_fdm_indices(discretization)
        if self.log:
            print(f"\033[96mDiscretization indices:\033[0m {d_ind} with "
                  f"min = {' '.join([str(i.lower) for i in d_ind])} and max = {' '.join([str(i.upper) for i in d_ind])}")
        # 2. Solve for the next time step
        #    We solve the discretized equations for the next time step.
        #    This will give us the expressions for the unknowns at the next time step.
        s_exprs = self._solve_for_next_timestep(d_equations)
        if self.log:
            print(f"\033[96mSolved for next time step:\033[0m {s_exprs}")
        # 3. Separate the linear and non-const terms in the equations
        s_dict = self._separate_linear_non_linear(s_exprs)
        if self.log:
            print(f"\033[96mSeparated linear and non-const terms:\033[0m {s_dict}")
        # 4. Optimize the non-const terms by grouping them together in a more efficient way
        s_dict = self._optimize(s_dict, focus=focus)
        if self.log:
            print(f"\033[96mOptimized non-const terms:\033[0m {s_dict}")
        # 5. Compute the stencil shape and instructions for computing the stencil
        stencil, instructions = self._compute_stencil(s_dict, d_ind)
        if self.log:
            print(f"\033[96mComputed Stencil:\033[0m {stencil}")
            print(f"\033[96mComputed Instructions:\033[0m {instructions}")
        self._stencil = stencil
        self._instructions = instructions
        return stencil, instructions

    def _discretize_unknowns(self, discretization):
        d_eqs = []
        for eq in self.equations:
            deriv_exprs = []
            for deriv, method in discretization.items():
                # We create a list of sympy expressions for the derivatives of the unknowns, depending on the FDMs
                sympy_dvs = [sympy.sympify(f"Derivative({uf}({','.join(self.free_symbols)}), {','.join(deriv)})") for uf
                             in self.unknowns]
                sympy_dvs_order = [sympy_dv.derivative_count for sympy_dv in sympy_dvs]
                # Helper list of tuples of the form (sympy derivative, unknown, order, fdm expression)
                deriv_exprs.extend(zip(sympy_dvs, self.unknowns, sympy_dvs_order,
                                       [sum(c * sympy.Indexed(uf, *idx) for c, idx in
                                            zip(method["coeffs"], method["indices"]) if c != 0)
                                        for uf in self.unknowns]))
            # We sort the derivatives by their order, so we can substitute the highest order derivatives first
            deriv_exprs.sort(key=lambda x: x[2], reverse=True)
            # Substitution happens here
            for deriv_expr, _, _, fdm_expr in deriv_exprs:
                eq = eq.subs(deriv_expr, fdm_expr)
            # We also substitute the 0th order derivatives with indexed unknowns (e.g. u -> u[0,0])
            for unk in self.unknowns:
                zero_idx = [0 for _ in range(len(self.free_symbols))]
                eq = eq.subs(sympy.sympify(f"{unk}({','.join(self.free_symbols)})"), sympy.Indexed(unk, *zero_idx))
            d_eqs.append(eq)
        return d_eqs

    def _compute_fdm_indices(self, discretization):
        d_ind = []
        # Collect all indices from the discretization methods
        for _, method in discretization.items():
            d_ind.extend(method["indices"])
        m = np.min(d_ind, axis=0)
        M = np.max(d_ind, axis=0)
        # We create a list of sympy indices for each dimension - this is used to compute the stencil shape and make it
        # easier to compute the stencil
        return [sympy.Idx(f"i_{i}", [m[i], M[i]]) for i in range(len(self.free_symbols))]

    def _solve_for_next_timestep(self, d_equations):
        next_time_step_index = [1] + [0 for _ in range(len(self.free_symbols) - 1)]
        next_time_step_vars = [sympy.Indexed(unk, *next_time_step_index) for unk in self.unknowns]
        sols = sympy.solve(d_equations, next_time_step_vars, dict=True)
        if not sols:
            raise ValueError("Failed to solve for the next time step")
        if len(sols) > 1:
            raise ValueError("Multiple solutions found for the next time step, which is not supported")
        return sols[0]

    def _separate_linear_non_linear(self, s_exprs):
        parts = {}
        for u_var, s_expr in s_exprs.items():
            s_expr = s_expr.expand()
            expr_terms = sympy.Add.make_args(s_expr)
            # We replace the indexed terms with a dummy variable, so we can check if the term is linear
            expr_terms_no_ind = [term.replace(lambda expr: isinstance(expr, sympy.Indexed),
                                              lambda expr: sympy.Dummy(u_var.base.name))
                                 for term in expr_terms]
            # A term is non-linear if it has a total degree greater than 1 (as a polynomial in the unknowns)
            # or if it contains a function
            non_linear_terms = [sympy.Poly(e).total_degree() > 1 or len(e.atoms(sympy.Function)) > 0 for e in
                                expr_terms_no_ind]
            non_linear_sum = sum([t for i, t in enumerate(expr_terms) if non_linear_terms[i]])
            linear_sum = sum([t for i, t in enumerate(expr_terms) if not non_linear_terms[i]])
            parts[u_var] = {
                "linear": linear_sum,
                "non-const": [non_linear_sum] if any(non_linear_terms) else []
            }
        return parts

    def _optimize(self, split_exprs, focus="depth"):
        # If we focus on memory, we just handle the whole non-const sum as a complicated auxiliary feature
        if focus == "memory":
            pass
        elif focus == "depth":
            # If we focus on depth, we try to factor the non-const sum
            for u_var, features in split_exprs.items():
                if not features["non-const"]:
                    continue
                factored = sympy.Add.make_args(features["non-const"][0])
                split_exprs[u_var]["non-const"] = list(factored)
        # After optimization, we find the most common coefficient and normalize the auxiliary features
        for u_var, features in split_exprs.items():
            if not features["non-const"]:
                continue
            cs = [[c for c in t.as_ordered_factors() if
                   c.is_constant() and not isinstance(c, (sympy.Indexed, sympy.Function))]
                  for t in sympy.Add.make_args(features["non-const"][0])]
            cs = [functools.reduce(lambda x1, x2: x1 * x2, c or [1.0]) for c in cs]
            c = max(set(cs), key=cs.count)
            split_exprs[u_var]["non-const"] = [(c, sympy.nsimplify(aux / c)) for aux in features["non-const"]]
        return split_exprs

    def _compute_stencil(self, feature_maps, d_ind):
        stencil_shape = self._compute_stencil_shape(feature_maps, d_ind)
        stencil = np.zeros(stencil_shape)
        instructions = {}

        aux_feature_index_begin = self._calculate_aux_feature_index_begin(d_ind)

        for u_var, features in feature_maps.items():
            var_idx = self.unknowns.index(u_var.base.name)
            self._process_linear_features(features, var_idx, d_ind, stencil, instructions)
            aux_feature_index_begin = self._process_non_const_features(features, var_idx, d_ind, stencil, instructions,
                                                                       aux_feature_index_begin)

        return stencil, instructions

    def _calculate_aux_feature_index_begin(self, d_ind):
        # Note: Again we assume that every supplied PDE has the same order in time. This is not necessarily true.
        # TODO: A more general solution is needed.
        return (d_ind[0].upper - d_ind[0].lower) * len(self.unknowns)

    def _process_linear_features(self, features, var_idx, d_ind, stencil, instructions):
        linear_feature = features["linear"]
        for term in sympy.Add.make_args(linear_feature):
            c, term = self._normalize_term(term)
            stencil_idx = self._calculate_stencil_idx(term, var_idx, d_ind)
            stencil[tuple(stencil_idx)] += c
            spatial_idx_dummy = [stencil_idx[-1] + d_ind[0].lower] + [0 for _ in range(len(self.free_symbols) - 1)]
            instructions[stencil_idx[-1]] = term.base[spatial_idx_dummy]

    def _normalize_term(self, term):
        c = [t for t in term.as_ordered_factors() if t.is_constant() and not isinstance(t, sympy.Indexed)]
        c = functools.reduce(lambda x1, x2: x1 * x2, c or [1.0])
        term = sympy.nsimplify(term / c)
        return c, term

    def _calculate_stencil_idx(self, term, var_idx, d_ind):
        stencil_idx = [var_idx] + [int(idx) - int(d_ind[i].lower) for i, idx in enumerate(term.indices)]
        stencil_idx[1] = stencil_idx[1] * len(self.unknowns) + var_idx
        stencil_idx.append(stencil_idx[1])
        del stencil_idx[1]
        return stencil_idx

    def _process_non_const_features(self, features, var_idx, d_ind, stencil, instructions, aux_feature_index_begin):
        for i, (c, aux) in enumerate(features["non-const"]):
            # Note: We assume that the auxiliary features are computed in a centered way
            # otherwise we need to be smarter about how we compute the stencil indices
            stencil_idx = [var_idx] + [(idx.upper - idx.lower) // 2 for idx in d_ind]
            stencil_idx[1] = aux_feature_index_begin
            aux_feature_index_begin += 1
            stencil_idx.append(stencil_idx[1])
            del stencil_idx[1]
            stencil[tuple(stencil_idx)] += c
            instructions[stencil_idx[-1]] = aux
        return aux_feature_index_begin

    def _compute_stencil_shape(self, feature_maps, d_ind):
        stencil_shape = [len(self.unknowns)] + [i.upper - i.lower + 1 for i in d_ind]
        # NOTE: Here we assume that every supplied PDE has the same order in time. This is not necessarily true.
        # A more general solution is needed.
        stencil_shape[1] = (stencil_shape[1] - 1) * len(self.unknowns)
        stencil_shape[1] += sum([len(features["non-const"]) for features in feature_maps.values()])
        stencil_shape.append(stencil_shape[1])
        del stencil_shape[1]
        return stencil_shape
