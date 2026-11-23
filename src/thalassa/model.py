"""Immutable public data model for PDE problems and compiler options."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar, cast

import sympy as sp
from sympy.core.function import AppliedUndef

from .dsl import erase_time_levels
from .exceptions import ModelValidationError
from .schemes import ExpressionRule
from .stencils import FiniteDifferenceRule

_T = TypeVar("_T")


def _items(value: _T | Sequence[_T], item_type: type[_T]) -> tuple[_T, ...]:
    """Normalize one item or a sequence without exposing mutable collections."""

    if isinstance(value, item_type):
        return (value,)
    return tuple(cast(Sequence[_T], value))


@dataclass(frozen=True)
class Axis:
    symbol: sp.Symbol
    lower: sp.Expr | float
    upper: sp.Expr | float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, sp.Symbol):
            raise ModelValidationError("Axis.symbol must be a SymPy Symbol")
        lower = sp.sympify(self.lower)
        upper = sp.sympify(self.upper)
        if lower.is_real is False or upper.is_real is False:
            raise ModelValidationError(f"Axis {self.symbol} bounds must be real")
        difference = sp.simplify(upper - lower)
        if difference.is_positive is not True:
            raise ModelValidationError(
                f"Axis {self.symbol} requires a finite upper bound greater than its lower bound"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def length(self) -> sp.Expr:
        return sp.simplify(self.upper - self.lower)


@dataclass(frozen=True, init=False)
class CartesianDomain:
    axes: tuple[Axis, ...]

    def __init__(self, *axes: Axis | Sequence[Axis]) -> None:
        if len(axes) == 1 and not isinstance(axes[0], Axis):
            normalized = tuple(axes[0])
        else:
            normalized = tuple(axes)  # type: ignore[arg-type]
        if not 1 <= len(normalized) <= 3:
            raise ModelValidationError("CartesianDomain supports one to three axes")
        if not all(isinstance(axis, Axis) for axis in normalized):
            raise ModelValidationError("CartesianDomain entries must be Axis objects")
        symbols = tuple(axis.symbol for axis in normalized)
        if len(set(symbols)) != len(symbols):
            raise ModelValidationError("Domain axis symbols must be unique")
        object.__setattr__(self, "axes", normalized)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def symbols(self) -> tuple[sp.Symbol, ...]:
        return tuple(axis.symbol for axis in self.axes)


@dataclass(frozen=True)
class UniformGrid:
    shape: tuple[int, ...]

    def __init__(self, shape: int | Sequence[int]) -> None:
        normalized = (shape,) if isinstance(shape, int) else tuple(shape)
        if not normalized or not all(isinstance(n, int) and not isinstance(n, bool) for n in normalized):
            raise ModelValidationError("UniformGrid.shape must contain integer cell counts")
        if any(n <= 0 for n in normalized):
            raise ModelValidationError("UniformGrid cell counts must be positive")
        object.__setattr__(self, "shape", normalized)

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class UniformDirichlet:
    value: float = 0.0

    def __post_init__(self) -> None:
        value = sp.sympify(self.value)
        if value.is_real is not True or value.free_symbols:
            raise ModelValidationError("Dirichlet boundary values must be real spatially uniform scalars")
        object.__setattr__(self, "value", float(value))


@dataclass(frozen=True)
class HomogeneousNeumann:
    pass


@dataclass(frozen=True)
class Periodic:
    pass


BoundaryCondition = UniformDirichlet | HomogeneousNeumann | Periodic


@dataclass(frozen=True)
class PDEEquation:
    target: sp.Function
    equation: sp.Equality

    def __post_init__(self) -> None:
        if not isinstance(self.target, AppliedUndef):
            raise ModelValidationError("PDEEquation.target must be an applied undefined SymPy function")
        if not isinstance(self.equation, sp.Equality):
            raise ModelValidationError("PDEEquation.equation must be a SymPy Eq")

    @property
    def residual(self) -> sp.Expr:
        return sp.Add(self.equation.lhs, -self.equation.rhs, evaluate=True)

    @property
    def continuous_equation(self) -> sp.Equality:
        """Return the continuous equation obtained by erasing time levels."""

        equation = erase_time_levels(self.equation)
        if not isinstance(equation, sp.Equality):  # pragma: no cover - guaranteed by construction
            raise ModelValidationError("Time-level erasure did not preserve the equation")
        return equation

    @property
    def continuous_residual(self) -> sp.Expr:
        """Return the continuous residual obtained by erasing time levels."""

        return erase_time_levels(self.residual)


@dataclass(frozen=True, init=False)
class TwoLevelDiscretization:
    """Immutable time step and exact spatial discretization rules."""

    dt: sp.Expr
    differences: tuple[FiniteDifferenceRule, ...]
    expression_rules: tuple[ExpressionRule, ...]

    def __init__(
        self,
        dt: float | sp.Expr,
        differences: FiniteDifferenceRule | Sequence[FiniteDifferenceRule] = (),
        *,
        expression_rules: ExpressionRule | Sequence[ExpressionRule] = (),
    ) -> None:
        value = sp.sympify(dt)
        if value.free_symbols or value.is_real is not True or value.is_positive is not True:
            raise ModelValidationError(
                "TwoLevelDiscretization.dt must be a positive compile-time constant"
            )
        normalized = _items(differences, FiniteDifferenceRule)
        if not all(isinstance(rule, FiniteDifferenceRule) for rule in normalized):
            raise ModelValidationError(
                "TwoLevelDiscretization.differences must contain finite-difference rules"
            )
        normalized_expressions = _items(expression_rules, ExpressionRule)
        if not all(isinstance(rule, ExpressionRule) for rule in normalized_expressions):
            raise ModelValidationError(
                "TwoLevelDiscretization.expression_rules must contain expression rules"
            )
        object.__setattr__(self, "dt", value)
        object.__setattr__(self, "differences", normalized)
        object.__setattr__(self, "expression_rules", normalized_expressions)


@dataclass(frozen=True)
class ScalarParameter:
    symbol: sp.Symbol

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, sp.Symbol):
            raise ModelValidationError("ScalarParameter requires a SymPy Symbol")

    @property
    def name(self) -> str:
        return self.symbol.name


@dataclass(frozen=True)
class CoefficientField:
    field: sp.Function | sp.FunctionClass
    boundary: BoundaryCondition = HomogeneousNeumann()

    def __post_init__(self) -> None:
        if isinstance(self.field, sp.FunctionClass):
            valid = issubclass(self.field, AppliedUndef)
        else:
            valid = isinstance(self.field, AppliedUndef)
        if not valid:
            raise ModelValidationError("CoefficientField requires an undefined SymPy function or applied function")
        if not isinstance(
            self.boundary,
            (UniformDirichlet, HomogeneousNeumann, Periodic),
        ):
            raise ModelValidationError(
                "CoefficientField.boundary must be a supported boundary condition"
            )

    @property
    def function(self) -> sp.FunctionClass:
        return self.field.func if isinstance(self.field, sp.Function) else self.field

    @property
    def name(self) -> str:
        return self.function.__name__


@dataclass(frozen=True, init=False)
class PDEProblem:
    equations: tuple[PDEEquation, ...]
    time: sp.Symbol
    domain: CartesianDomain
    boundaries: Mapping[sp.Function, BoundaryCondition]
    parameters: tuple[ScalarParameter, ...]
    coefficient_fields: tuple[CoefficientField, ...]
    name: str
    discretization: TwoLevelDiscretization

    def __init__(
        self,
        equations: PDEEquation | Sequence[PDEEquation],
        time: sp.Symbol,
        domain: CartesianDomain,
        boundaries: Mapping[sp.Function, BoundaryCondition],
        parameters: ScalarParameter | Sequence[ScalarParameter] = (),
        coefficient_fields: CoefficientField | Sequence[CoefficientField] = (),
        name: str = "PDEProblem",
        *,
        discretization: TwoLevelDiscretization,
    ) -> None:
        object.__setattr__(self, "equations", _items(equations, PDEEquation))
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "boundaries", MappingProxyType(dict(boundaries)))
        object.__setattr__(self, "parameters", _items(parameters, ScalarParameter))
        object.__setattr__(self, "coefficient_fields", _items(coefficient_fields, CoefficientField))
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "discretization", discretization)

    @property
    def fields(self) -> tuple[sp.Function, ...]:
        return tuple(equation.target for equation in self.equations)


@dataclass(frozen=True)
class KrylovOptions:
    max_iterations: int = 200
    rtol_float16: float = 1e-3
    rtol_bfloat16: float = 1e-2
    rtol_float32: float = 1e-5
    rtol_float64: float = 1e-8
    atol: float = 0.0

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ModelValidationError("Krylov max_iterations must be positive")
        if min(
            self.rtol_float16,
            self.rtol_bfloat16,
            self.rtol_float32,
            self.rtol_float64,
            self.atol,
        ) < 0:
            raise ModelValidationError("Krylov tolerances must be nonnegative")


@dataclass(frozen=True)
class NewtonOptions:
    max_iterations: int = 20
    max_backtracks: int = 8
    backtrack_factor: float = 0.5
    armijo_constant: float = 1e-4
    atol_float32: float = 1e-6
    atol_float64: float = 1e-10

    def __post_init__(self) -> None:
        if self.max_iterations <= 0 or self.max_backtracks <= 0:
            raise ModelValidationError("Newton iteration limits must be positive")
        if not 0 < self.backtrack_factor < 1:
            raise ModelValidationError("Newton backtrack_factor must be between zero and one")
        if not 0 < self.armijo_constant < 1:
            raise ModelValidationError("Newton armijo_constant must be between zero and one")


@dataclass(frozen=True)
class CompileOptions:
    krylov: KrylovOptions = field(default_factory=KrylovOptions)
    newton: NewtonOptions = field(default_factory=NewtonOptions)
    simplify: bool = True


__all__ = [
    "Axis",
    "BoundaryCondition",
    "CartesianDomain",
    "CoefficientField",
    "CompileOptions",
    "HomogeneousNeumann",
    "KrylovOptions",
    "NewtonOptions",
    "PDEEquation",
    "PDEProblem",
    "Periodic",
    "ScalarParameter",
    "TwoLevelDiscretization",
    "UniformDirichlet",
    "UniformGrid",
]
