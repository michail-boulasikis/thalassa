"""Whole-expression discretization rules and built-in composite schemes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from .exceptions import DiscretizationError
from .stencils import FiniteDifferenceRule, centered


@dataclass(frozen=True, init=False)
class ExpressionRule:
    """Replace one exact symbolic expression with a discretization recipe.

    Derivative rules bundled with the recipe are local to the approximation.
    The surrounding ``old`` or ``new`` annotation is inherited from the matched
    target occurrence.
    """

    target: sp.Expr
    approximation: sp.Expr
    differences: tuple[FiniteDifferenceRule, ...]

    def __init__(
        self,
        target: sp.Expr,
        approximation: sp.Expr,
        differences: FiniteDifferenceRule | Sequence[FiniteDifferenceRule] = (),
    ) -> None:
        normalized_target = sp.sympify(target)
        normalized_approximation = sp.sympify(approximation)
        if not isinstance(normalized_target, sp.Expr):
            raise DiscretizationError("ExpressionRule.target must be a SymPy expression")
        if not isinstance(normalized_approximation, sp.Expr):
            raise DiscretizationError(
                "ExpressionRule.approximation must be a SymPy expression"
            )
        if isinstance(differences, FiniteDifferenceRule):
            normalized_differences = (differences,)
        else:
            normalized_differences = tuple(differences)
        if not all(
            isinstance(rule, FiniteDifferenceRule)
            for rule in normalized_differences
        ):
            raise DiscretizationError(
                "ExpressionRule.differences must contain finite-difference rules"
            )
        object.__setattr__(self, "target", normalized_target)
        object.__setattr__(self, "approximation", normalized_approximation)
        object.__setattr__(self, "differences", normalized_differences)


def arakawa_jacobian(
    left: sp.Expr,
    right: sp.Expr,
    *,
    axes: tuple[sp.Symbol, sp.Symbol],
) -> ExpressionRule:
    """Return the classic second-order Arakawa discretization of ``J(left, right)``.

    The sign convention is ``J(a, b) = a_x b_y - a_y b_x``.  The rule is the
    equal-weight average of the advective and two conservative centered forms.
    """

    normalized_axes = tuple(axes)
    if (
        len(normalized_axes) != 2
        or not all(isinstance(axis, sp.Symbol) for axis in normalized_axes)
        or normalized_axes[0] == normalized_axes[1]
    ):
        raise DiscretizationError(
            "Arakawa Jacobian axes must be two distinct SymPy symbols"
        )
    x, y = normalized_axes
    a = sp.sympify(left)
    b = sp.sympify(right)
    if not isinstance(a, sp.Expr) or not isinstance(b, sp.Expr):
        raise DiscretizationError(
            "Arakawa Jacobian operands must be SymPy expressions"
        )

    a_x = sp.Derivative(a, x, evaluate=False)
    a_y = sp.Derivative(a, y, evaluate=False)
    b_x = sp.Derivative(b, x, evaluate=False)
    b_y = sp.Derivative(b, y, evaluate=False)
    target = a_x * b_y - a_y * b_x
    if target == 0:
        raise DiscretizationError(
            "Arakawa Jacobian operands must define a nonzero symbolic target"
        )
    advective = target
    first_conservative = (
        sp.Derivative(a * b_y, x, evaluate=False)
        - sp.Derivative(a * b_x, y, evaluate=False)
    )
    second_conservative = (
        sp.Derivative(b * a_x, y, evaluate=False)
        - sp.Derivative(b * a_y, x, evaluate=False)
    )
    approximation = sp.Add(
        advective,
        first_conservative,
        second_conservative,
    ) / sp.Integer(3)
    differences = tuple(
        centered(derivative, accuracy=2)
        for derivative in sorted(
            approximation.atoms(sp.Derivative),
            key=sp.default_sort_key,
        )
    )
    return ExpressionRule(target, approximation, differences)


__all__ = ["ExpressionRule", "arakawa_jacobian"]
