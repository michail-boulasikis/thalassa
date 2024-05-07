import sympy
import numpy as np
import functools

import pprint


class StencilCompiler(object):
    def __init__(self, equations, unknowns, known_functions, free_symbols, log=False):
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

    def compile(self, discretization, focus="depth"):
        # 1. Discretize the equations
        #    This step replaces the derivatives with the FDM expressions given in the discretization dictionary
        d_equations = self._discretize(discretization)
        # 1.1 Compute the kernel shape given the discretization (this is easier done here than later)
        # 2. Solve for the next time step where possible
        #    This step tries to solve the equations for the next time step. If it fails this could mean three things:
        #    A. The method is implicit and the equation cannot be solved analytically
        #    B. The equations are too complex to solve for the next time step.
        #    C. The equations have more than one solution for the next time step. This is not supported.
        d_s_exprs = self._solve_for_next_timestep(d_equations)
        # 3. Perform optimizations (i.e., eliminate as many auxiliary features as possible)
        #    This step reduces the number of auxiliary features by performing factoring
        #    and common subexpression elimination.
        feature_maps = self._optimize(d_s_exprs, focus)
        # 4. Compute the stencil and the instructions how to compute the auxiliary features
        stencil, instructions = self._compute_stencil(feature_maps)
        return stencil, instructions

    def _discretize(self, discretization):
        d_eqs = []
        for eq in self.equations:
            deriv_exprs = []
            for deriv, method in discretization.items():
                sympy_dvs = [sympy.sympify(f"Derivative({uf}({','.join(self.free_symbols)}), {','.join(deriv)})") for uf
                             in self.unknowns]
                sympy_dvs_order = [sympy_dv.derivative_count for sympy_dv in sympy_dvs]
                deriv_exprs.extend(zip(sympy_dvs, self.unknowns, sympy_dvs_order,
                                       [sum(c * sympy.Indexed(uf, *idx) for c, idx in
                                            zip(method["coeffs"], method["indices"]) if c != 0)
                                        for uf in self.unknowns]))
            deriv_exprs.sort(key=lambda x: x[2], reverse=True)
            for deriv_expr, _, _, fdm_expr in deriv_exprs:
                eq = eq.subs(deriv_expr, fdm_expr)
            for unk in self.unknowns:
                zero_idx = [0 for _ in range(len(self.free_symbols))]
                eq = eq.subs(sympy.sympify(f"{unk}({','.join(self.free_symbols)})"), sympy.Indexed(unk, *zero_idx))
            d_eqs.append(eq)
        return d_eqs

    def _solve_for_next_timestep(self, d_equations):
        next_time_step_index = [1] + [0 for _ in range(len(self.free_symbols) - 1)]
        next_time_step_vars = [sympy.Indexed(unk, *next_time_step_index) for unk in self.unknowns]
        sols = sympy.solve(d_equations, next_time_step_vars, dict=True)
        if not sols:
            raise ValueError("Failed to solve for the next time step")
        if len(sols) > 1:
            raise ValueError("Multiple solutions found for the next time step, which is not supported")
        return sols[0]

    def _optimize(self, d_s_exprs, focus="depth"):
        feature_maps = {}
        # No matter the optimization, we always need to group the terms in linear and non-linear.
        for u_var, s_expr in d_s_exprs.items():
            expr_terms = sympy.Add.make_args(s_expr)
            expr_terms_no_ind = [term.replace(lambda expr: isinstance(expr, sympy.Indexed),
                                              lambda expr: sympy.Symbol(expr.base.name))
                                 for term in expr_terms]

            non_linear_terms = [sympy.Poly(e).total_degree() > 1 or len(e.atoms(sympy.Function)) > 0 for e in
                                expr_terms_no_ind]
            non_linear_sum = sum([t for i, t in enumerate(expr_terms) if non_linear_terms[i]])
            linear_sum = sum([t for i, t in enumerate(expr_terms) if not non_linear_terms[i]])
            feature_maps[u_var] = {
                "linear": linear_sum,
                "auxiliary_features": [non_linear_sum] if any(non_linear_terms) else []
            }
        # If we focus on memory, we just handle the whole non-linear sum as a complicated auxiliary feature
        if focus == "memory":
            pass
        elif focus == "depth":
            # If we focus on depth, we try to factor the non-linear sum
            for u_var, features in feature_maps.items():
                if not features["auxiliary_features"]:
                    continue
                factored = sympy.Add.make_args(features["auxiliary_features"][0])
                feature_maps[u_var]["auxiliary_features"] = list(factored)
        return feature_maps

    def _compute_stencil(self, feature_maps):
        # 1. Compute the stencil shape
        stencil_shape, mM = self._compute_stencil_shape(feature_maps)
        # 2. Collect the coefficients and instructions for computing the auxiliary features
        stencil = np.zeros(stencil_shape)
        instructions = {}
        # 2.1 Handle the linear terms
        aux_feature_index_begin = -1
        for u_var, features in feature_maps.items():
            linear_feature = features["linear"]
            u_var_idx = self.unknowns.index(u_var.base.name)
            for term in sympy.Add.make_args(linear_feature):
                c = [t for t in term.as_ordered_factors() if t.is_constant() and not isinstance(t, sympy.Indexed)]
                c = functools.reduce(lambda x1, x2: x1 * x2, c or [1.0])
                term = sympy.nsimplify(term / c)
                u = term.atoms(sympy.Indexed).pop()
                stencil_idx = np.array([int(i) for i in u.indices])
                stencil_idx[0] = stencil_idx[0] * len(self.unknowns) + u_var_idx
                if stencil_idx[0] > aux_feature_index_begin:
                    aux_feature_index_begin = stencil_idx[0]
                stencil_idx -= np.array([v[0] for v in mM[u_var.base]])
                stencil_idx = np.roll(stencil_idx, -1)
                stencil[tuple(stencil_idx)] += c
                spatial_idx_dummy = [stencil_idx[-1]] + [0 for _ in range(len(self.free_symbols) - 1)]
                instructions[stencil_idx[-1]] = term.base[spatial_idx_dummy]
        # 2.2 Handle the auxiliary features
        for u_var, features in feature_maps.items():
            aux_features = features["auxiliary_features"]
            if not aux_features:
                continue
            for i, aux in enumerate(aux_features):
                # 2.2.1 Extract coefficient which appears most often in the terms of the auxiliary feature
                cs = [[c for c in t.as_ordered_factors()
                      if c.is_constant() and not isinstance(c, (sympy.Indexed, sympy.Function))]
                      for t in sympy.Add.make_args(aux)]
                cs = [functools.reduce(lambda x1, x2: x1 * x2, c or [1.0]) for c in cs]
                c = max(set(cs), key=cs.count)
                # 2.2.2 Normalize the auxiliary feature by dividing it by the coefficient
                aux = sympy.nsimplify(aux / c)
                # 2.2.3 Compute the stencil index for the auxiliary feature
                stencil_idx = np.array([int((v[1] - v[0]) / 2) for v in mM[u_var.base]])
                stencil_idx = np.roll(stencil_idx, -1)
                stencil_idx[-1] += aux_feature_index_begin + i + 1
                aux_feature_index_begin = stencil_idx[-1]
                stencil[tuple(stencil_idx)] += c
                instructions[stencil_idx[-1]] = aux
        return stencil, instructions

    def _compute_stencil_shape(self, feature_maps):
        # All of this is necessary because my planning is shit and I have no proper indices.
        indices = []
        for _, components in feature_maps.items():
            indices.extend(components["linear"].atoms(sympy.Indexed))
            for aux in components["auxiliary_features"]:
                indices.extend(aux.atoms(sympy.Indexed))
        n_aux_features = sum([len(features["auxiliary_features"]) for features in feature_maps.values()])
        indices = list(set(indices))
        i_dict = {}
        for i in indices:
            if i.base not in i_dict:
                i_dict[i.base] = []
            i_dict[i.base].append([int(j) for j in i.indices])
        # Find min and max of each element in i_dict
        min_max = {k: [[min([j[i] for j in v]), max(j[i] for j in v)] for i in range(len(v[0]))]
                   for k, v in i_dict.items()}
        stencil_shape = [[v[1] - v[0] + 1 for v in j] for j in min_max.values()]
        stencil_shape = np.max(np.array(stencil_shape), axis=0)
        # Note, does not necessarily work well for all problems, investigate further
        stencil_shape[0] *= len(self.unknowns)
        stencil_shape[0] += n_aux_features
        stencil_shape = np.roll(stencil_shape, -1)
        return stencil_shape, min_max
