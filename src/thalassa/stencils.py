"""Exact finite-difference stencils and built-in rule constructors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import sympy as sp

from .exceptions import DiscretizationError


def _normal_offset(offset: Sequence[int]) -> tuple[int, ...]:
    result = tuple(offset)
    if not result or not all(isinstance(value, int) and not isinstance(value, bool) for value in result):
        raise DiscretizationError("Stencil offsets must be nonempty integer tuples")
    return result


@dataclass(frozen=True, init=False)
class Stencil:
    """A lexicographically normalized exact mapping from offsets to weights."""

    entries: tuple[tuple[tuple[int, ...], sp.Expr], ...]

    def __init__(
        self,
        coefficients: Mapping[Sequence[int], object]
        | Iterable[tuple[Sequence[int], object]],
    ) -> None:
        items = coefficients.items() if isinstance(coefficients, Mapping) else coefficients
        combined: dict[tuple[int, ...], sp.Expr] = {}
        ndim: int | None = None
        for raw_offset, raw_coefficient in items:
            offset = _normal_offset(raw_offset)
            if ndim is None:
                ndim = len(offset)
            if len(offset) != ndim:
                raise DiscretizationError("All stencil offsets must have the same dimensionality")
            coefficient = sp.simplify(sp.sympify(raw_coefficient))
            if coefficient.is_finite is False:
                raise DiscretizationError("Stencil coefficients must be finite")
            combined[offset] = sp.simplify(combined.get(offset, sp.S.Zero) + coefficient)
        normalized = tuple(
            (offset, coefficient)
            for offset, coefficient in sorted(combined.items())
            if coefficient != 0
        )
        if not normalized:
            normalized = (((0,) * (ndim or 1), sp.S.Zero),)
        object.__setattr__(self, "entries", normalized)

    @property
    def coefficients(self) -> Mapping[tuple[int, ...], sp.Expr]:
        return dict(self.entries)

    @property
    def offsets(self) -> tuple[tuple[int, ...], ...]:
        return tuple(offset for offset, _ in self.entries)

    @property
    def weights(self) -> tuple[sp.Expr, ...]:
        return tuple(coefficient for _, coefficient in self.entries)

    @property
    def ndim(self) -> int:
        return len(self.entries[0][0])

    @property
    def radius(self) -> tuple[int, ...]:
        return tuple(max(abs(offset[axis]) for offset, _ in self.entries) for axis in range(self.ndim))

    def moment(self, powers: Sequence[int]) -> sp.Expr:
        powers = tuple(powers)
        if len(powers) != self.ndim:
            raise DiscretizationError("Moment powers must match stencil dimensionality")
        return sp.simplify(
            sum(
                coefficient
                * sp.prod(
                    sp.Integer(index) ** power
                    for index, power in zip(offset, powers, strict=True)
                )
                for offset, coefficient in self.entries
            )
        )

    def __add__(self, other: Stencil) -> Stencil:
        if not isinstance(other, Stencil) or self.ndim != other.ndim:
            return NotImplemented
        return Stencil((*self.entries, *other.entries))

    def __sub__(self, other: Stencil) -> Stencil:
        if not isinstance(other, Stencil) or self.ndim != other.ndim:
            return NotImplemented
        return Stencil((*self.entries, *((offset, -coefficient) for offset, coefficient in other.entries)))

    def __mul__(self, scalar: object) -> Stencil:
        if isinstance(scalar, Stencil):
            if self.ndim != scalar.ndim:
                return NotImplemented
            return Stencil(
                (
                    tuple(
                        left + right
                        for left, right in zip(left_offset, right_offset, strict=True)
                    ),
                    left_coefficient * right_coefficient,
                )
                for left_offset, left_coefficient in self.entries
                for right_offset, right_coefficient in scalar.entries
            )
        scalar = sp.sympify(scalar)
        return Stencil((offset, coefficient * scalar) for offset, coefficient in self.entries)

    def __rmul__(self, scalar: object) -> Stencil:
        return self * scalar

    def __neg__(self) -> Stencil:
        return self * -1


@dataclass(frozen=True)
class FiniteDifferenceRule:
    derivative: sp.Derivative
    stencil: Stencil
    accuracy: int | None = None
    scale_by_spacing: bool = False
    local_axes: tuple[sp.Symbol, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.derivative, sp.Derivative):
            raise DiscretizationError("FiniteDifferenceRule.derivative must be a SymPy Derivative")
        if not isinstance(self.stencil, Stencil):
            raise DiscretizationError("FiniteDifferenceRule.stencil must be a Stencil")
        if self.accuracy is not None and self.accuracy <= 0:
            raise DiscretizationError("Finite-difference accuracy must be positive")
        if self.local_axes and len(self.local_axes) != self.stencil.ndim:
            raise DiscretizationError("Rule local_axes must match stencil dimensionality")


def _points(kind: str, derivative_order: int, accuracy: int) -> tuple[int, ...]:
    if derivative_order <= 0:
        raise DiscretizationError("A spatial finite-difference rule needs positive derivative order")
    if accuracy <= 0:
        raise DiscretizationError("accuracy denotes a positive truncation order")
    if kind == "centered":
        radius = max(1, (derivative_order + 1) // 2)
        while True:
            point_count = 2 * radius + 1
            achieved = point_count - derivative_order
            if achieved % 2 == 1:
                achieved += 1
            if achieved >= accuracy:
                break
            radius += 1
        return tuple(range(-radius, radius + 1))
    count = derivative_order + accuracy
    if kind == "forward":
        return tuple(range(0, count))
    if kind == "backward":
        return tuple(range(-(count - 1), 1))
    raise DiscretizationError(f"Unknown finite-difference kind {kind!r}")


def _one_dimensional_weights(order: int, points: Sequence[int]) -> tuple[sp.Expr, ...]:
    return tuple(sp.simplify(weight) for weight in sp.finite_diff_weights(order, list(points), 0)[order][-1])


def _built_in(derivative: sp.Derivative, accuracy: int, kind: str) -> FiniteDifferenceRule:
    if not isinstance(derivative, sp.Derivative):
        raise DiscretizationError("Finite-difference helpers require a SymPy Derivative")
    local_axes: list[sp.Symbol] = []
    orders: list[int] = []
    for variable, count in derivative.variable_count:
        if not isinstance(variable, sp.Symbol):
            raise DiscretizationError("Finite-difference variables must be symbols")
        local_axes.append(variable)
        orders.append(int(count))
    point_sets = [_points(kind, order, accuracy) for order in orders]
    weight_sets = [
        _one_dimensional_weights(order, points)
        for order, points in zip(orders, point_sets, strict=True)
    ]
    entries: list[tuple[tuple[int, ...], sp.Expr]] = [((), sp.S.One)]
    for points, weights in zip(point_sets, weight_sets, strict=True):
        entries = [
            (prefix + (point,), coefficient * weight)
            for prefix, coefficient in entries
            for point, weight in zip(points, weights, strict=True)
        ]
    return FiniteDifferenceRule(
        derivative=derivative,
        stencil=Stencil(entries),
        accuracy=accuracy,
        scale_by_spacing=True,
        local_axes=tuple(local_axes),
    )


def centered(derivative: sp.Derivative, accuracy: int = 2) -> FiniteDifferenceRule:
    return _built_in(derivative, accuracy, "centered")


def forward(derivative: sp.Derivative, accuracy: int = 1) -> FiniteDifferenceRule:
    return _built_in(derivative, accuracy, "forward")


def backward(derivative: sp.Derivative, accuracy: int = 1) -> FiniteDifferenceRule:
    return _built_in(derivative, accuracy, "backward")


__all__ = ["FiniteDifferenceRule", "Stencil", "backward", "centered", "forward"]
