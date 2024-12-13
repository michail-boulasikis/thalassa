from functools import cached_property
from abc import ABC, abstractmethod
import sympy
import numpy as np
import scipy


class FDApproximation(ABC):

    @cached_property
    @abstractmethod
    def _weights(self):
        pass

    @property
    @abstractmethod
    def derivative_expr(self):
        pass

    @property
    @abstractmethod
    def fdm_expr(self):
        pass


class FDApproximationTaylor(FDApproximation):
    def __init__(self, base_func, order, imin, imax, dx):
        if not isinstance(base_func, sympy.Function):
            raise ValueError("Base function for the FDM must be a sympy function")
        self.base_func = base_func
        if len(order) != len(self.base_func.args):
            raise ValueError("Derivative must be described as an order for each variable in the base function")
        if not all(isinstance(i, int) for i in order):
            raise ValueError("Derivative orders must be integers")
        self.order = order
        if len(imin) != len(self.base_func.args) or len(imax) != len(self.base_func.args):
            raise ValueError("imin and imax must be provided for each variable in the base function")
        if not all(isinstance(i, int) for i in imin) or not all(isinstance(i, int) for i in imax):
            raise ValueError("imin and imax must be integers")
        if not all(i <= j for i, j in zip(imin, imax)):
            raise ValueError("imin must be less than or equal to imax")
        if not all(i - j + 1 >= k for i, j, k in zip(imax, imin, order)):
            raise ValueError("imax - imin + 1 must be large enough to support the requested derivative order")
        self.imin = imin
        self.imax = imax
        if not all(isinstance(i, float) for i in dx):
            raise ValueError("Step sizes must be floats")
        if len(dx) != len(self.base_func.args):
            raise ValueError("Step sizes must be provided for each variable in the base function")
        if not all(dx > 0 for dx in dx):
            raise ValueError("Step sizes must be positive")
        if not all(dx <= 1 for dx in dx):
            raise ValueError("Step sizes must be less than 1")
        self.dx = dx

    @cached_property
    def weights(self):
        c_vectors = []
        taylor_coeff = scipy.special.factorial(np.sum(self.order)) / np.prod(np.power(self.dx, self.order))
        for var, (start, stop, order) in enumerate(zip(self.imin, self.imax, self.order)):
            if order > 0:
                W = np.zeros((stop - start + 1, stop - start + 1), dtype=np.float64)
                for i in range(start, stop + 1):
                    for j in range(start, stop + 1):
                        W[j - start, i - start] = i ** (j - start)
                e = np.zeros(stop - start + 1, dtype=np.float64)
                e[order] = 1.0
                c_vectors.append(np.linalg.solve(W, e))
            else:
                c_vectors.append(np.ones(stop - start + 1, dtype=np.float64))
        c_tensor = c_vectors[0]
        for c in c_vectors[1:]:
            c_tensor = np.multiply.outer(c_tensor, c)
        c_tensor *= taylor_coeff
        return c_tensor

    @property
    def derivative_expr(self):
        return sympy.Derivative(self.base_func,
                                *[(self.base_func.args[i], order) for i, order in enumerate(self.order) if order > 0])

    @property
    def fdm_expr(self):
        return sum(self.weights[i] * sympy.Indexed(self.base_func.name, *(np.array(i) + self.imin)) for i in
                   np.ndindex(*self.weights.shape))

    def __str__(self):
        return f"{self.derivative_expr} ==> {self.fdm_expr}"

    def __repr__(self):
        return self.__str__()


def fdm_simple_partial_derivative(deriv: sympy.Derivative, dx: float, method='forward', order=None):
    if len(deriv.variable_count) != 1:
        raise ValueError("Supplied derivative in the FDM helper is either a mixed derivative or wrongly typed. "
                         "If you want to express the FDM of a mixed derivative use 'fdm_mixed_partial_derivative'.")
    variable, deriv_order = deriv.variable_count[0]
    if order is None:
        order = int(deriv_order)
    if order < deriv_order:
        raise ValueError("Supplied order is smaller than the derivative order in the FDM helper")
    func = deriv.args[0]
    var_idx = func.args.index(variable)
    imin = [0] * len(func.args)
    imax = [0] * len(func.args)
    order_array = [0] * len(func.args)
    dx_array = [1.0] * len(func.args)
    match method:
        case 'forward':
            imax[var_idx] = order
        case 'backward':
            imin[var_idx] = - order
        case 'central':
            imax[var_idx] = (order + 1) // 2
            imin[var_idx] = - ((order + 1) // 2)
        case _:
            raise ValueError(f"Unknown FDM method supplied in helper function: {method}")
    dx_array[var_idx] = dx
    order_array[var_idx] = order
    return FDApproximationTaylor(func, order_array, imin, imax, dx_array)
