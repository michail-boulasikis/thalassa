"""Two-level temporal annotation DSL."""

from __future__ import annotations

import sympy as sp


class Old(sp.Function):
    nargs = 1


class New(sp.Function):
    nargs = 1


def old(expr: sp.Expr) -> sp.Expr:
    """Evaluate all temporal quantities in *expr* at step n."""

    return Old(sp.sympify(expr))


def new(expr: sp.Expr) -> sp.Expr:
    """Evaluate all temporal quantities in *expr* at step n+1."""

    return New(sp.sympify(expr))


def erase_time_levels(expr: sp.Expr) -> sp.Expr:
    """Project an annotated expression back to its continuous form."""

    result = sp.sympify(expr)
    if isinstance(result, sp.Equality):
        return sp.Eq(
            erase_time_levels(result.lhs),
            erase_time_levels(result.rhs),
            evaluate=False,
        )
    previous: sp.Expr | None = None
    while previous != result:
        previous = result
        result = result.replace(
            lambda item: isinstance(item, (Old, New)),
            lambda item: item.args[0],
        )
    return result


__all__ = [
    "New",
    "Old",
    "erase_time_levels",
    "new",
    "old",
]
