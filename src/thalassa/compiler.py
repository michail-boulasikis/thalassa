"""Pure compiler passes from the symbolic DSL to :mod:`thalassa.ir`."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import sympy as sp
from sympy.core.function import AppliedUndef

from .dsl import New, Old
from .exceptions import (
    CanonicalizationError,
    CompilationError,
    DiscretizationError,
    ModelValidationError,
)
from .ir import (
    PROGRAM_IR_VERSION,
    BlockKind,
    BoundaryIR,
    Call,
    CoefficientRef,
    CoordinateRef,
    DiscreteEquationIR,
    ExpressionIR,
    GridRef,
    Literal,
    NaryOp,
    OperatorProperties,
    Power,
    ProgramIR,
    PropertyState,
    ScalarRef,
    ScheduleLayerIR,
    SolveBlockIR,
    StencilApply,
    TemporalTreatment,
    TimeLevel,
    TimeRef,
    _expression_children,
)
from .model import (
    BoundaryCondition,
    CompileOptions,
    HomogeneousNeumann,
    PDEProblem,
    Periodic,
    TwoLevelDiscretization,
    UniformDirichlet,
    UniformGrid,
)
from .schemes import ExpressionRule
from .stencils import FiniteDifferenceRule, Stencil

_SUPPORTED_FUNCTIONS = {
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.asinh: "asinh",
    sp.acosh: "acosh",
    sp.atanh: "atanh",
    sp.exp: "exp",
    sp.log: "log",
    sp.Abs: "abs",
    sp.sign: "sign",
    sp.floor: "floor",
    sp.ceiling: "ceil",
    sp.Min: "minimum",
    sp.Max: "maximum",
}


_ResolvedDifference = tuple[tuple[tuple[int, ...], sp.Expr], ...]


@dataclass(frozen=True)
class _ResolvedExpressionRule:
    target: sp.Expr
    approximation: sp.Expr
    differences: Mapping[sp.Derivative, _ResolvedDifference]


@dataclass(frozen=True)
class VerifiedModel:
    problem: PDEProblem
    grid: UniformGrid
    options: CompileOptions
    field_indices: Mapping[sp.Function, int]
    resolved_rules: Mapping[sp.Derivative, _ResolvedDifference]
    resolved_expression_rules: Mapping[sp.Expr, _ResolvedExpressionRule]


@dataclass(frozen=True)
class _SymbolicEquation:
    target: int
    residual: sp.Expr
    execution_residual: ExpressionIR
    continuous_residual: str
    annotated_residual: sp.Expr


@dataclass(frozen=True)
class DiscreteSystem:
    verified: VerifiedModel
    equations: tuple[_SymbolicEquation, ...]
    atoms: Mapping[sp.Symbol, ExpressionIR]


@dataclass(frozen=True)
class CanonicalSystem:
    discrete: DiscreteSystem
    equations: tuple[_SymbolicEquation, ...]
    dependencies: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ScheduledSystem:
    canonical: CanonicalSystem
    layers: tuple[tuple[tuple[int, ...], ...], ...]


def _stable_expressions(expressions: Iterable[sp.Basic]) -> tuple[sp.Basic, ...]:
    return tuple(sorted(expressions, key=sp.default_sort_key))


def _coefficient_function_names(problem: PDEProblem) -> dict[sp.FunctionClass, str]:
    return {coefficient.function: coefficient.name for coefficient in problem.coefficient_fields}


def _time_order(derivative: sp.Derivative, time: sp.Symbol) -> int:
    return sum(
        int(count)
        for variable, count in derivative.variable_count
        if variable == time
    )


def _validate_time_annotations(
    expression: sp.Expr,
    problem: PDEProblem,
    target_functions: set[sp.FunctionClass],
    coefficient_functions: set[sp.FunctionClass],
) -> None:
    def visit(node: sp.Basic, levelled: bool = False) -> None:
        if isinstance(node, (Old, New)):
            time_derivatives = tuple(
                derivative
                for derivative in node.args[0].atoms(sp.Derivative)
                if _time_order(derivative, problem.time)
            )
            if time_derivatives:
                raise ModelValidationError(
                    f"Time derivative {time_derivatives[0]} cannot be wrapped in old/new"
                )
            visit(node.args[0], True)
            return
        if isinstance(node, sp.Derivative):
            time_order = _time_order(node, problem.time)
            if time_order > 1:
                raise ModelValidationError(
                    f"Higher-order time derivative {node} is not supported; introduce state explicitly"
                )
            if time_order:
                if levelled or node.expr.has(Old, New):
                    raise ModelValidationError(
                        f"Time derivative {node} cannot contain an explicit old/new time level"
                    )
                return
            visit(node.expr, levelled)
            return
        if isinstance(node, AppliedUndef):
            if node.func in target_functions and not levelled:
                raise ModelValidationError(
                    f"Field {node.func.__name__} must be wrapped in old(...) or new(...) "
                    "unless it is differentiated in time"
                )
            if (
                node.func in coefficient_functions
                and problem.time in node.args
                and not levelled
            ):
                raise ModelValidationError(
                    f"Time-dependent coefficient field {node.func.__name__} must be wrapped "
                    "in old(...) or new(...)"
                )
            return
        if isinstance(node, sp.Symbol) and node == problem.time and not levelled:
            raise ModelValidationError(
                "The time symbol must be wrapped in old(...) or new(...) unless it is a derivative variable"
            )
        for argument in node.args:
            if isinstance(argument, sp.Basic):
                visit(argument, levelled)

    visit(expression)


def _validate_model(problem: PDEProblem, grid: UniformGrid) -> dict[sp.Function, int]:
    if not isinstance(problem, PDEProblem):
        raise ModelValidationError("verify expects a PDEProblem")
    if not isinstance(grid, UniformGrid):
        raise ModelValidationError("verify expects a UniformGrid")
    if not isinstance(problem.discretization, TwoLevelDiscretization):
        raise ModelValidationError(
            "PDEProblem.discretization must be a TwoLevelDiscretization"
        )
    if not isinstance(problem.time, sp.Symbol):
        raise ModelValidationError("PDEProblem.time must be a SymPy Symbol")
    if problem.time in problem.domain.symbols:
        raise ModelValidationError("The time symbol cannot also be a spatial axis")
    if grid.ndim != problem.domain.ndim:
        raise ModelValidationError(
            f"Grid dimensionality {grid.ndim} does not match domain dimensionality {problem.domain.ndim}"
        )
    if not problem.equations:
        raise ModelValidationError("PDEProblem requires at least one equation")

    targets = tuple(equation.target for equation in problem.equations)
    if len(set(targets)) != len(targets):
        raise ModelValidationError("Every PDEEquation target must be unique")
    names = tuple(target.func.__name__ for target in targets)
    if len(set(names)) != len(names):
        raise ModelValidationError("Target function names must be unique")
    expected_arguments = (problem.time, *problem.domain.symbols)
    for target in targets:
        if tuple(target.args) != expected_arguments:
            raise ModelValidationError(
                f"Field {target.func.__name__} arguments must be exactly {expected_arguments} in that order"
            )

    boundary_keys = set(problem.boundaries)
    if boundary_keys != set(targets):
        missing = [target.func.__name__ for target in targets if target not in boundary_keys]
        extra = [str(key) for key in boundary_keys if key not in set(targets)]
        raise ModelValidationError(f"Boundaries must cover exactly the target fields; missing={missing}, extra={extra}")
    for target in targets:
        if not isinstance(
            problem.boundaries[target],
            (UniformDirichlet, HomogeneousNeumann, Periodic),
        ):
            raise ModelValidationError(f"Unsupported boundary condition for {target.func.__name__}")

    parameter_symbols = tuple(parameter.symbol for parameter in problem.parameters)
    if len(set(parameter_symbols)) != len(parameter_symbols):
        raise ModelValidationError("Scalar parameters must be unique")
    coefficient_functions = _coefficient_function_names(problem)
    if len(coefficient_functions) != len(problem.coefficient_fields):
        raise ModelValidationError("Coefficient fields must be unique")
    if set(names) & set(coefficient_functions.values()):
        raise ModelValidationError("State fields and coefficient fields cannot share names")

    allowed_arguments = expected_arguments
    target_functions = {target.func for target in targets}
    for equation in problem.equations:
        if equation.target not in equation.residual.atoms(AppliedUndef):
            raise ModelValidationError(
                f"Equation for {equation.target.func.__name__} does not reference its declared target"
            )
        _validate_time_annotations(
            equation.residual,
            problem,
            target_functions,
            set(coefficient_functions),
        )
        for function in _stable_expressions(equation.residual.atoms(AppliedUndef)):
            if function.func in target_functions:
                if tuple(function.args) != expected_arguments:
                    raise ModelValidationError(
                        f"Every occurrence of field {function.func.__name__} must use arguments {expected_arguments}"
                    )
            elif function.func in coefficient_functions:
                arguments = tuple(function.args)
                positions = [
                    allowed_arguments.index(argument)
                    for argument in arguments
                    if argument in allowed_arguments
                ]
                if len(positions) != len(arguments) or positions != sorted(set(positions)):
                    raise ModelValidationError(
                        f"Coefficient field {function.func.__name__} arguments must be an ordered subset "
                        f"of {allowed_arguments}"
                    )
            else:
                raise ModelValidationError(
                    f"Undeclared function {function.func.__name__}; declare sampled coefficients with CoefficientField"
                )
        undeclared = equation.residual.free_symbols - set(expected_arguments) - set(parameter_symbols)
        if undeclared:
            raise ModelValidationError(
                "Undeclared scalar symbols: "
                + ", ".join(
                    symbol.name for symbol in sorted(undeclared, key=sp.default_sort_key)
                )
            )
        for node in sp.preorder_traversal(equation.residual):
            supported = (
                isinstance(
                    node,
                    (
                        sp.Add,
                        sp.Mul,
                        sp.Pow,
                        sp.Symbol,
                        sp.Number,
                        sp.NumberSymbol,
                        sp.Derivative,
                        AppliedUndef,
                        Old,
                        New,
                        sp.Tuple,
                    ),
                )
                or node.func in _SUPPORTED_FUNCTIONS
            )
            if not supported:
                raise ModelValidationError(
                    f"Equation contains unsupported symbolic operation {node.func}"
                )
    return {target: index for index, target in enumerate(targets)}


def _resolve_rule(
    rule: FiniteDifferenceRule,
    problem: PDEProblem,
    grid: UniformGrid,
) -> tuple[tuple[tuple[int, ...], sp.Expr], ...]:
    derivative_axes = tuple(variable for variable, _ in rule.derivative.variable_count)
    if problem.time in derivative_axes:
        raise DiscretizationError("Spatial finite-difference rules cannot include the time axis")
    if any(axis not in problem.domain.symbols for axis in derivative_axes):
        raise DiscretizationError(f"Rule {rule.derivative} differentiates along an undeclared axis")

    if rule.local_axes:
        local_axes = rule.local_axes
    elif rule.stencil.ndim == problem.domain.ndim:
        local_axes = problem.domain.symbols
    else:
        local_axes = tuple(dict.fromkeys(derivative_axes))
    if rule.stencil.ndim != len(local_axes):
        raise DiscretizationError(
            f"Stencil dimensionality for {rule.derivative} cannot be mapped to domain axes"
        )
    if any(axis not in problem.domain.symbols for axis in local_axes):
        raise DiscretizationError(f"Rule {rule.derivative} contains a non-spatial local axis")

    axis_indices = tuple(problem.domain.symbols.index(axis) for axis in local_axes)
    spacing_factor = sp.S.One
    if rule.scale_by_spacing:
        for variable, order in rule.derivative.variable_count:
            axis = problem.domain.symbols.index(variable)
            spacing = problem.domain.axes[axis].length / grid.shape[axis]
            spacing_factor *= spacing ** int(order)
    resolved: list[tuple[tuple[int, ...], sp.Expr]] = []
    for local_offset, coefficient in rule.stencil.entries:
        global_offset = [0] * problem.domain.ndim
        for local_axis, domain_axis in enumerate(axis_indices):
            global_offset[domain_axis] = local_offset[local_axis]
        resolved.append((tuple(global_offset), sp.simplify(coefficient / spacing_factor)))
    return tuple(sorted(resolved))


def _contains_exact(expression: sp.Basic, target: sp.Expr, *, proper: bool = False) -> bool:
    for index, node in enumerate(sp.preorder_traversal(expression)):
        if proper and index == 0:
            continue
        if node == target:
            return True
    return False


def _validate_expression_rule_symbols(
    expression: sp.Expr,
    problem: PDEProblem,
    label: str,
) -> None:
    if expression.has(Old, New):
        raise DiscretizationError(
            f"Expression rule {label} cannot contain old/new annotations"
        )
    time_derivatives = tuple(
        derivative
        for derivative in expression.atoms(sp.Derivative)
        if _time_order(derivative, problem.time)
    )
    if time_derivatives:
        raise DiscretizationError(
            f"Expression rule {label} cannot contain time derivative {time_derivatives[0]}"
        )

    expected_arguments = (problem.time, *problem.domain.symbols)
    allowed_arguments = expected_arguments
    target_functions = {target.func for target in problem.fields}
    coefficient_functions = _coefficient_function_names(problem)
    for function in _stable_expressions(expression.atoms(AppliedUndef)):
        if function.func in target_functions:
            if tuple(function.args) != expected_arguments:
                raise DiscretizationError(
                    f"Every occurrence of field {function.func.__name__} in an expression rule "
                    f"must use arguments {expected_arguments}"
                )
        elif function.func in coefficient_functions:
            arguments = tuple(function.args)
            positions = [
                allowed_arguments.index(argument)
                for argument in arguments
                if argument in allowed_arguments
            ]
            if len(positions) != len(arguments) or positions != sorted(set(positions)):
                raise DiscretizationError(
                    f"Coefficient field {function.func.__name__} arguments in an expression rule "
                    f"must be an ordered subset of {allowed_arguments}"
                )
        else:
            raise DiscretizationError(
                f"Undeclared function {function.func.__name__} in expression rule {label}"
            )

    parameter_symbols = {parameter.symbol for parameter in problem.parameters}
    undeclared = expression.free_symbols - set(expected_arguments) - parameter_symbols
    if undeclared:
        names = ", ".join(
            symbol.name for symbol in sorted(undeclared, key=sp.default_sort_key)
        )
        raise DiscretizationError(
            f"Undeclared scalar symbols in expression rule {label}: {names}"
        )

    for node in sp.preorder_traversal(expression):
        supported = (
            isinstance(
                node,
                (
                    sp.Add,
                    sp.Mul,
                    sp.Pow,
                    sp.Symbol,
                    sp.Number,
                    sp.NumberSymbol,
                    sp.Derivative,
                    AppliedUndef,
                    sp.Tuple,
                ),
            )
            or node.func in _SUPPORTED_FUNCTIONS
        )
        if not supported:
            raise DiscretizationError(
                f"Expression rule {label} contains unsupported symbolic operation {node.func}"
            )


def _required_spatial_derivatives(
    expression: sp.Expr,
    time: sp.Symbol,
    masked_targets: frozenset[sp.Expr] = frozenset(),
) -> set[sp.Derivative]:
    required: set[sp.Derivative] = set()

    def visit(node: sp.Basic) -> None:
        if isinstance(node, sp.Expr) and node in masked_targets:
            return
        if isinstance(node, sp.Derivative):
            time_order = _time_order(node, time)
            if not time_order:
                required.add(node)
            else:
                spatial_variables: list[sp.Symbol] = []
                for variable, count in node.variable_count:
                    if variable != time:
                        spatial_variables.extend([variable] * int(count))
                if spatial_variables:
                    required.add(
                        sp.Derivative(
                            node.expr,
                            *spatial_variables,
                            evaluate=False,
                        )
                    )
            visit(node.expr)
            return
        for argument in node.args:
            if isinstance(argument, sp.Basic):
                visit(argument)

    visit(expression)
    return required


def _resolve_difference_rules(
    differences: Sequence[FiniteDifferenceRule],
    problem: PDEProblem,
    grid: UniformGrid,
    *,
    local_target: sp.Expr | None = None,
) -> Mapping[sp.Derivative, _ResolvedDifference]:
    rules: dict[sp.Derivative, _ResolvedDifference] = {}
    for rule in differences:
        if rule.derivative in rules:
            qualifier = "local " if local_target is not None else ""
            raise DiscretizationError(
                f"Duplicate {qualifier}spatial rule for {rule.derivative}"
            )
        resolved = _resolve_rule(rule, problem, grid)
        radius = tuple(
            max(abs(offset[axis]) for offset, _ in resolved)
            for axis in range(grid.ndim)
        )
        for axis, required in enumerate(radius):
            if required > grid.shape[axis]:
                context = (
                    f" in expression rule {local_target}"
                    if local_target is not None
                    else ""
                )
                raise DiscretizationError(
                    f"Grid axis {problem.domain.axes[axis].symbol} has {grid.shape[axis]} cells "
                    f"but stencil radius is {required}{context}"
                )
        rules[rule.derivative] = resolved
    return MappingProxyType(rules)


def _validate_expression_rules(
    problem: PDEProblem,
    grid: UniformGrid,
    global_rules: Mapping[sp.Derivative, _ResolvedDifference],
) -> Mapping[sp.Expr, _ResolvedExpressionRule]:
    declared = problem.discretization.expression_rules
    by_target: dict[sp.Expr, ExpressionRule] = {}
    for rule in declared:
        if rule.target in by_target:
            raise DiscretizationError(
                f"Duplicate expression rule for exact target {rule.target}"
            )
        _validate_expression_rule_symbols(rule.target, problem, "target")
        _validate_expression_rule_symbols(rule.approximation, problem, "approximation")
        by_target[rule.target] = rule

    targets = tuple(by_target)
    for index, left in enumerate(targets):
        for right in targets[index + 1 :]:
            if _contains_exact(left, right, proper=True) or _contains_exact(
                right, left, proper=True
            ):
                raise DiscretizationError(
                    f"Expression rule targets overlap structurally: {left} and {right}"
                )
    for rule in declared:
        for other in targets:
            if other != rule.target and _contains_exact(rule.approximation, other):
                raise DiscretizationError(
                    f"Expression rule for {rule.target} depends on expression rule target {other}"
                )

    sources = tuple(equation.residual for equation in problem.equations)
    for target in targets:
        if not any(_contains_exact(source, target) for source in sources):
            raise DiscretizationError(
                f"Expression rule target {target} does not occur exactly in any equation"
            )

    resolved: dict[sp.Expr, _ResolvedExpressionRule] = {}
    for rule in declared:
        local_rules = _resolve_difference_rules(
            rule.differences,
            problem,
            grid,
            local_target=rule.target,
        )
        missing = _required_spatial_derivatives(
            rule.approximation,
            problem.time,
        ) - (set(local_rules) | set(global_rules))
        if missing:
            derivative = min(missing, key=sp.default_sort_key)
            raise DiscretizationError(
                f"No exact spatial rule was declared for {derivative} in the "
                f"approximation of {rule.target}"
            )
        resolved[rule.target] = _ResolvedExpressionRule(
            target=rule.target,
            approximation=rule.approximation,
            differences=local_rules,
        )
    return MappingProxyType(resolved)


def _validate_discretization(
    problem: PDEProblem,
    grid: UniformGrid,
) -> tuple[
    Mapping[sp.Derivative, _ResolvedDifference],
    Mapping[sp.Expr, _ResolvedExpressionRule],
]:
    rules = _resolve_difference_rules(
        problem.discretization.differences,
        problem,
        grid,
    )
    expression_rules = _validate_expression_rules(problem, grid, rules)
    masked_targets = frozenset(expression_rules)
    required: set[sp.Derivative] = set()
    for equation in problem.equations:
        required.update(
            _required_spatial_derivatives(
                equation.residual,
                problem.time,
                masked_targets,
            )
        )
    missing = required - set(rules)
    if missing:
        derivative = min(missing, key=sp.default_sort_key)
        raise DiscretizationError(f"No exact spatial rule was declared for {derivative}")
    return rules, expression_rules


def verify(
    problem: PDEProblem,
    grid: UniformGrid,
    options: CompileOptions | None = None,
) -> VerifiedModel:
    """Verify immutable compiler inputs and resolve exact stencil weights."""

    options = options or CompileOptions()
    if not isinstance(options, CompileOptions):
        raise ModelValidationError("options must be CompileOptions or None")
    field_indices = _validate_model(problem, grid)
    resolved_rules, resolved_expression_rules = _validate_discretization(problem, grid)
    return VerifiedModel(
        problem=problem,
        grid=grid,
        options=options,
        field_indices=MappingProxyType(dict(field_indices)),
        resolved_rules=resolved_rules,
        resolved_expression_rules=resolved_expression_rules,
    )


class _Registry:
    def __init__(self, forbidden_names: set[str]) -> None:
        self._forbidden_names = forbidden_names
        self._by_key: dict[tuple[object, ...], sp.Symbol] = {}
        self._atoms: dict[sp.Symbol, ExpressionIR] = {}
        self._counter = 0

    def symbol(
        self,
        key: tuple[object, ...],
        reference: ExpressionIR,
        **assumptions: bool,
    ) -> sp.Symbol:
        if key in self._by_key:
            return self._by_key[key]
        while True:
            name = f"__THALASSA_INTERNAL_REF_{self._counter:08d}"
            self._counter += 1
            if name not in self._forbidden_names:
                break
        symbol = sp.Symbol(name, **assumptions)
        self._by_key[key] = symbol
        self._atoms[symbol] = reference
        return symbol

    @property
    def atoms(self) -> Mapping[sp.Symbol, ExpressionIR]:
        return MappingProxyType(dict(self._atoms))


def _execution_add(arguments: Iterable[ExpressionIR]) -> ExpressionIR:
    flattened: list[ExpressionIR] = []
    for argument in arguments:
        if isinstance(argument, NaryOp) and argument.op == "add":
            flattened.extend(argument.arguments)
        elif isinstance(argument, Literal) and sp.sympify(argument.value) == 0:
            continue
        else:
            flattened.append(argument)
    if not flattened:
        return Literal("0")
    if len(flattened) == 1:
        return flattened[0]
    return NaryOp("add", tuple(flattened))


def _execution_mul(arguments: Iterable[ExpressionIR]) -> ExpressionIR:
    flattened: list[ExpressionIR] = []
    numeric = sp.S.One
    for argument in arguments:
        if isinstance(argument, NaryOp) and argument.op == "mul":
            nested = argument.arguments
        else:
            nested = (argument,)
        for item in nested:
            if isinstance(item, Literal):
                value = sp.sympify(item.value)
                if value.is_number:
                    numeric *= value
                    continue
            flattened.append(item)
    numeric = sp.simplify(numeric)
    if numeric == 0:
        return Literal("0")
    if numeric != 1 or not flattened:
        flattened.insert(0, Literal(sp.sstr(numeric)))
    if len(flattened) == 1:
        return flattened[0]
    return NaryOp("mul", tuple(flattened))


def _execution_negate(expression: ExpressionIR) -> ExpressionIR:
    return _execution_mul((Literal("-1"), expression))


@dataclass(frozen=True)
class _LoweredExpression:
    symbolic: sp.Expr
    execution: ExpressionIR


class _Discretizer:
    """Lower once to both the symbolic analysis form and execution IR."""

    def __init__(self, verified: VerifiedModel, registry: _Registry) -> None:
        self.verified = verified
        self.problem = verified.problem
        self.registry = registry
        self.field_functions = {
            target.func: index for target, index in verified.field_indices.items()
        }
        self.coefficient_functions = _coefficient_function_names(self.problem)
        self.parameters = {parameter.symbol: parameter.name for parameter in self.problem.parameters}
        self.ndim = self.problem.domain.ndim

    @staticmethod
    def _add_offsets(
        left: tuple[int, ...], right: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(a + b for a, b in zip(left, right, strict=True))

    def lower(
        self,
        expression: sp.Expr,
        level: TimeLevel | None = None,
        offset: tuple[int, ...] | None = None,
        local_rules: Mapping[sp.Derivative, _ResolvedDifference] | None = None,
        apply_expression_rules: bool = True,
    ) -> _LoweredExpression:
        offset = offset or (0,) * self.ndim
        if apply_expression_rules:
            expression_rule = self.verified.resolved_expression_rules.get(expression)
            if expression_rule is not None:
                return self.lower(
                    expression_rule.approximation,
                    level,
                    offset,
                    expression_rule.differences,
                    False,
                )
        if isinstance(expression, Old):
            return self.lower(
                expression.args[0],
                TimeLevel.OLD,
                offset,
                local_rules,
                apply_expression_rules,
            )
        if isinstance(expression, New):
            return self.lower(
                expression.args[0],
                TimeLevel.NEW,
                offset,
                local_rules,
                apply_expression_rules,
            )
        if isinstance(expression, sp.Derivative):
            if _time_order(expression, self.problem.time):
                if level is not None or expression.expr.has(Old, New):
                    raise DiscretizationError(
                        f"Time derivative {expression} cannot have an explicit old/new time level"
                    )
                return self._time_derivative(
                    expression,
                    offset,
                    local_rules,
                    apply_expression_rules,
                )
            if level is None:
                if expression.expr.has(self.problem.time):
                    raise DiscretizationError(
                        f"Spatial derivative {expression} has no old/new time level"
                    )
                level = TimeLevel.OLD
            return self._spatial_derivative(
                expression,
                level,
                offset,
                local_rules,
                apply_expression_rules,
            )
        if isinstance(expression, AppliedUndef):
            if expression.func in self.field_functions:
                if level is None:
                    raise DiscretizationError(
                        f"Field {expression.func.__name__} remains without an old/new time level"
                    )
                field = self.field_functions[expression.func]
                reference: ExpressionIR = GridRef(field, level, offset)
                return _LoweredExpression(
                    self.registry.symbol(
                        ("grid", field, level.value, offset),
                        reference,
                    ),
                    reference,
                )
            if expression.func in self.coefficient_functions:
                if level is None:
                    if self.problem.time in expression.args:
                        raise DiscretizationError(
                            f"Time-dependent coefficient field {expression.func.__name__} "
                            "remains without an old/new time level"
                        )
                    level = TimeLevel.OLD
                name = self.coefficient_functions[expression.func]
                assumptions: dict[str, bool] = {}
                if expression.is_positive is True:
                    assumptions["positive"] = True
                elif expression.is_negative is True:
                    assumptions["negative"] = True
                reference = CoefficientRef(name, level, offset)
                return _LoweredExpression(
                    self.registry.symbol(
                        ("coefficient", name, level.value, offset),
                        reference,
                        **assumptions,
                    ),
                    reference,
                )
            raise DiscretizationError(
                f"Undeclared symbolic function {expression.func.__name__}"
            )
        if isinstance(expression, sp.Symbol):
            if expression == self.problem.time:
                if level is None:
                    raise DiscretizationError(
                        "The time symbol remains without an old/new time level"
                    )
                reference = TimeRef(level)
                return _LoweredExpression(
                    self.registry.symbol(
                        ("time", level.value),
                        reference,
                    ),
                    reference,
                )
            if expression in self.problem.domain.symbols:
                axis = self.problem.domain.symbols.index(expression)
                reference = CoordinateRef(axis, offset[axis])
                return _LoweredExpression(
                    self.registry.symbol(
                        ("coordinate", axis, offset[axis]),
                        reference,
                    ),
                    reference,
                )
            if expression in self.parameters:
                return _LoweredExpression(expression, ScalarRef(expression.name))
            raise DiscretizationError(f"Undeclared symbol {expression}")
        if expression.is_Number or expression in (sp.pi, sp.E, sp.oo, -sp.oo):
            return _LoweredExpression(expression, Literal(sp.sstr(expression)))
        if isinstance(expression, (sp.Add, sp.Mul, sp.Pow)):
            arguments = tuple(
                self.lower(
                    argument,
                    level,
                    offset,
                    local_rules,
                    apply_expression_rules,
                )
                for argument in expression.args
            )
            symbolic = expression.func(*(argument.symbolic for argument in arguments))
            if isinstance(expression, sp.Add):
                execution = _execution_add(argument.execution for argument in arguments)
            elif isinstance(expression, sp.Mul):
                execution = _execution_mul(argument.execution for argument in arguments)
            else:
                execution = Power(arguments[0].execution, arguments[1].execution)
            return _LoweredExpression(symbolic, execution)
        if expression.func in _SUPPORTED_FUNCTIONS:
            arguments = tuple(
                self.lower(
                    argument,
                    level,
                    offset,
                    local_rules,
                    apply_expression_rules,
                )
                for argument in expression.args
            )
            return _LoweredExpression(
                expression.func(*(argument.symbolic for argument in arguments)),
                Call(
                    _SUPPORTED_FUNCTIONS[expression.func],
                    tuple(argument.execution for argument in arguments),
                ),
            )
        if not expression.args and expression.is_number:
            return _LoweredExpression(expression, Literal(sp.sstr(expression)))
        raise DiscretizationError(f"Unsupported symbolic operation {expression.func}")

    def _spatial_derivative(
        self,
        derivative: sp.Derivative,
        level: TimeLevel,
        offset: tuple[int, ...],
        local_rules: Mapping[sp.Derivative, _ResolvedDifference] | None = None,
        apply_expression_rules: bool = True,
    ) -> _LoweredExpression:
        entries = local_rules.get(derivative) if local_rules is not None else None
        if entries is None:
            entries = self.verified.resolved_rules.get(derivative)
        if entries is None:
            raise DiscretizationError(
                f"No exact spatial rule was declared for {derivative}"
            )
        shifted = tuple(
            (
                coefficient,
                self.lower(
                    derivative.expr,
                    level,
                    self._add_offsets(offset, stencil_offset),
                    local_rules,
                    apply_expression_rules,
                ),
            )
            for stencil_offset, coefficient in entries
        )
        symbolic = sp.Add(
            *(coefficient * value.symbolic for coefficient, value in shifted)
        )
        if all(coefficient.is_number for _, coefficient in entries):
            execution: ExpressionIR = StencilApply(
                tuple((stencil_offset, sp.sstr(coefficient)) for stencil_offset, coefficient in entries),
                self.lower(
                    derivative.expr,
                    level,
                    offset,
                    local_rules,
                    apply_expression_rules,
                ).execution,
            )
        else:
            execution = _execution_add(
                _execution_mul(
                    (
                        self.lower(
                            coefficient,
                            level,
                            offset,
                            local_rules,
                            apply_expression_rules,
                        ).execution,
                        value.execution,
                    )
                )
                for coefficient, value in shifted
            )
        return _LoweredExpression(symbolic, execution)

    def _time_derivative(
        self,
        expression: sp.Derivative,
        offset: tuple[int, ...],
        local_rules: Mapping[sp.Derivative, _ResolvedDifference] | None = None,
        apply_expression_rules: bool = True,
    ) -> _LoweredExpression:
        if _time_order(expression, self.problem.time) != 1:
            raise DiscretizationError(
                "Automatic backward Euler supports exactly one time differentiation"
            )
        spatial_variables: list[sp.Symbol] = []
        for variable, count in expression.variable_count:
            if variable != self.problem.time:
                spatial_variables.extend([variable] * int(count))
        if spatial_variables:
            spatial = sp.Derivative(
                expression.expr, *spatial_variables, evaluate=False
            )
            new_value = self._spatial_derivative(
                spatial,
                TimeLevel.NEW,
                offset,
                local_rules,
                apply_expression_rules,
            )
            old_value = self._spatial_derivative(
                spatial,
                TimeLevel.OLD,
                offset,
                local_rules,
                apply_expression_rules,
            )
        else:
            new_value = self.lower(
                expression.expr,
                TimeLevel.NEW,
                offset,
                local_rules,
                apply_expression_rules,
            )
            old_value = self.lower(
                expression.expr,
                TimeLevel.OLD,
                offset,
                local_rules,
                apply_expression_rules,
            )
        return _LoweredExpression(
            sp.simplify(
                (new_value.symbolic - old_value.symbolic)
                / self.problem.discretization.dt
            ),
            _execution_mul(
                (
                    Literal(sp.sstr(1 / self.problem.discretization.dt)),
                    _execution_add(
                        (new_value.execution, _execution_negate(old_value.execution))
                    ),
                )
            ),
        )


def _contains_stencil(expression: ExpressionIR) -> bool:
    return isinstance(expression, StencilApply) or any(
        _contains_stencil(child) for child in _expression_children(expression)
    )


def _spatially_uniform(expression: ExpressionIR) -> bool:
    if isinstance(expression, (GridRef, CoefficientRef, CoordinateRef, StencilApply)):
        return False
    return all(_spatially_uniform(child) for child in _expression_children(expression))


def _stencil_from_ir(expression: StencilApply) -> Stencil:
    return Stencil(
        tuple(
            (offset, sp.sympify(coefficient))
            for offset, coefficient in expression.entries
        )
    )


def _stencil_ir(stencil: Stencil, expression: ExpressionIR) -> ExpressionIR:
    entries = tuple(
        (offset, sp.sstr(coefficient))
        for offset, coefficient in stencil.entries
    )
    if all(coefficient == 0 for _, coefficient in stencil.entries):
        return Literal("0")
    zero_offset = (0,) * stencil.ndim
    if len(stencil.entries) == 1 and stencil.entries[0][0] == zero_offset:
        return _execution_mul(
            (Literal(sp.sstr(stencil.entries[0][1])), expression)
        )
    return StencilApply(entries, expression)


def _scaled_stencil(
    expression: ExpressionIR,
) -> tuple[Stencil, ExpressionIR] | None:
    if isinstance(expression, StencilApply):
        return _stencil_from_ir(expression), expression.expression
    if isinstance(expression, GridRef) and expression.level == TimeLevel.OLD:
        zero_offset = (0,) * len(expression.offset)
        return (
            Stencil(((expression.offset, sp.S.One),)),
            GridRef(expression.field, expression.level, zero_offset),
        )
    if not isinstance(expression, NaryOp) or expression.op != "mul":
        return None
    stencil_arguments = [
        argument
        for argument in expression.arguments
        if isinstance(argument, StencilApply)
        or (
            isinstance(argument, GridRef)
            and argument.level == TimeLevel.OLD
        )
    ]
    if len(stencil_arguments) != 1:
        return None
    scale = sp.S.One
    for argument in expression.arguments:
        if argument is stencil_arguments[0]:
            continue
        if not isinstance(argument, Literal):
            return None
        value = sp.sympify(argument.value)
        if not value.is_number:
            return None
        scale *= value
    stencil_argument = stencil_arguments[0]
    if isinstance(stencil_argument, StencilApply):
        return _stencil_from_ir(stencil_argument) * scale, stencil_argument.expression
    zero_offset = (0,) * len(stencil_argument.offset)
    return (
        Stencil(((stencil_argument.offset, scale),)),
        GridRef(stencil_argument.field, stencil_argument.level, zero_offset),
    )


def _optimize_stencils(expression: ExpressionIR) -> ExpressionIR:
    if isinstance(expression, NaryOp):
        arguments = tuple(_optimize_stencils(item) for item in expression.arguments)
        result = (
            _execution_add(arguments)
            if expression.op == "add"
            else _execution_mul(arguments)
        )
        if not isinstance(result, NaryOp) or result.op != "add":
            return result
        grouped: dict[ExpressionIR, Stencil] = {}
        remaining: list[ExpressionIR] = []
        for argument in result.arguments:
            item = _scaled_stencil(argument)
            if item is None:
                remaining.append(argument)
                continue
            stencil, operand = item
            grouped[operand] = (
                grouped[operand] + stencil if operand in grouped else stencil
            )
        remaining.extend(
            _stencil_ir(stencil, operand)
            for operand, stencil in grouped.items()
        )
        return _execution_add(remaining)
    if isinstance(expression, Power):
        return Power(
            _optimize_stencils(expression.base),
            _optimize_stencils(expression.exponent),
        )
    if isinstance(expression, Call):
        return Call(
            expression.function,
            tuple(_optimize_stencils(argument) for argument in expression.arguments),
        )
    if not isinstance(expression, StencilApply):
        return expression

    operand = _optimize_stencils(expression.expression)
    stencil = _stencil_from_ir(expression)
    if isinstance(operand, StencilApply):
        return _optimize_stencils(
            _stencil_ir(stencil * _stencil_from_ir(operand), operand.expression)
        )
    if isinstance(operand, NaryOp) and operand.op == "mul":
        uniform = tuple(
            argument for argument in operand.arguments if _spatially_uniform(argument)
        )
        spatial = tuple(
            argument for argument in operand.arguments if not _spatially_uniform(argument)
        )
        if uniform and spatial:
            return _execution_mul(
                (*uniform, _optimize_stencils(_stencil_ir(stencil, _execution_mul(spatial))))
            )
    if isinstance(operand, NaryOp) and operand.op == "add" and _contains_stencil(operand):
        return _optimize_stencils(
            _execution_add(_stencil_ir(stencil, argument) for argument in operand.arguments)
        )
    return _stencil_ir(stencil, operand)


def _replace_target_new_with_zero(
    expression: ExpressionIR,
    target: int,
) -> tuple[ExpressionIR, bool]:
    """Replace the direct unknown and report other new-state dependencies."""

    if isinstance(expression, GridRef):
        if expression.level != TimeLevel.NEW:
            return expression, False
        zero_offset = (0,) * len(expression.offset)
        if expression.field == target and expression.offset == zero_offset:
            return Literal("0"), False
        return expression, True
    if isinstance(expression, NaryOp):
        replaced = tuple(
            _replace_target_new_with_zero(argument, target)
            for argument in expression.arguments
        )
        arguments = tuple(item[0] for item in replaced)
        result = (
            _execution_add(arguments)
            if expression.op == "add"
            else _execution_mul(arguments)
        )
        return result, any(item[1] for item in replaced)
    if isinstance(expression, Power):
        base, base_pending = _replace_target_new_with_zero(
            expression.base, target
        )
        exponent, exponent_pending = _replace_target_new_with_zero(
            expression.exponent, target
        )
        return Power(base, exponent), base_pending or exponent_pending
    if isinstance(expression, Call):
        replaced = tuple(
            _replace_target_new_with_zero(argument, target)
            for argument in expression.arguments
        )
        return (
            Call(expression.function, tuple(item[0] for item in replaced)),
            any(item[1] for item in replaced),
        )
    if isinstance(expression, StencilApply):
        operand, pending = _replace_target_new_with_zero(
            expression.expression, target
        )
        return StencilApply(expression.entries, operand), pending
    return expression, False


def _direct_stencil_solution(
    residual: ExpressionIR,
    diagonal: ExpressionIR | None,
    target: int,
) -> ExpressionIR | None:
    """Solve a constant-diagonal direct equation as one old-state stencil."""

    if not isinstance(diagonal, Literal):
        return None
    coefficient = sp.sympify(diagonal.value)
    if not coefficient.is_number or coefficient == 0:
        return None
    residual_zero, has_other_new = _replace_target_new_with_zero(
        residual, target
    )
    if has_other_new:
        return None
    candidate = _optimize_stencils(
        _execution_mul((Literal(sp.sstr(-1 / coefficient)), residual_zero))
    )
    scaled = _scaled_stencil(candidate)
    if scaled is not None:
        candidate = _stencil_ir(*scaled)
    return candidate if isinstance(candidate, StencilApply) else None


def _required_execution_halo(
    expression: ExpressionIR,
    requested: tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(expression, (GridRef, CoefficientRef)):
        return tuple(
            radius + abs(offset)
            for radius, offset in zip(requested, expression.offset, strict=True)
        )
    if isinstance(expression, StencilApply):
        radius = _stencil_from_ir(expression).radius
        return _required_execution_halo(
            expression.expression,
            tuple(
                left + right
                for left, right in zip(requested, radius, strict=True)
            ),
        )
    children = tuple(
        _required_execution_halo(child, requested)
        for child in _expression_children(expression)
    )
    if not children:
        return requested
    return tuple(max(items) for items in zip(*children, strict=True))


def discretize(verified: VerifiedModel) -> DiscreteSystem:
    """Lower temporal annotations and exact spatial rules to symbolic grid atoms."""

    forbidden = {
        symbol.name
        for equation in verified.problem.equations
        for symbol in equation.residual.free_symbols
    }
    registry = _Registry(forbidden)
    discretizer = _Discretizer(verified, registry)
    equations: list[_SymbolicEquation] = []
    for equation in verified.problem.equations:
        source = equation.residual
        lowered = discretizer.lower(source)
        residual = lowered.symbolic
        execution_residual = _optimize_stencils(lowered.execution)
        required_halo = _required_execution_halo(
            execution_residual,
            (0,) * verified.problem.domain.ndim,
        )
        for axis, required in enumerate(required_halo):
            if required > verified.grid.shape[axis]:
                raise DiscretizationError(
                    f"Grid axis {verified.problem.domain.axes[axis].symbol} has "
                    f"{verified.grid.shape[axis]} cells but composed stencil radius "
                    f"is {required}"
                )
        if verified.options.simplify:
            residual = sp.factor_terms(sp.cancel(residual))
        if residual.has(sp.Derivative, Old, New, AppliedUndef):
            raise DiscretizationError(f"Incomplete lowering for target {equation.target.func.__name__}")
        equations.append(
            _SymbolicEquation(
                target=verified.field_indices[equation.target],
                residual=residual,
                execution_residual=execution_residual,
                continuous_residual=sp.srepr(equation.continuous_residual),
                annotated_residual=source,
            )
        )
    return DiscreteSystem(verified, tuple(equations), registry.atoms)


def _new_grid_symbols(
    equation: _SymbolicEquation,
    atoms: Mapping[sp.Symbol, ExpressionIR],
    fields: set[int] | None = None,
) -> tuple[sp.Symbol, ...]:
    result = []
    for symbol, info in atoms.items():
        if (
            symbol in equation.residual.free_symbols
            and isinstance(info, GridRef)
            and info.level == TimeLevel.NEW
            and (fields is None or info.field in fields)
        ):
            result.append(symbol)
    return tuple(result)


def canonicalize(discrete: DiscreteSystem) -> CanonicalSystem:
    """Represent each equation as a residual and derive exact new-state dependencies."""

    oriented: list[_SymbolicEquation] = []
    zero_offset = (0,) * discrete.verified.problem.domain.ndim
    for equation in discrete.equations:
        center_symbols = [
            symbol
            for symbol in _new_grid_symbols(equation, discrete.atoms, {equation.target})
            if cast(GridRef, discrete.atoms[symbol]).offset == zero_offset
        ]
        residual = equation.residual
        execution_residual = equation.execution_residual
        if center_symbols:
            center_coefficient = sp.simplify(sp.diff(residual, center_symbols[0]))
            if center_coefficient.is_negative is True:
                residual = -residual
                execution_residual = _execution_negate(execution_residual)
        oriented.append(
            _SymbolicEquation(
                target=equation.target,
                residual=residual,
                execution_residual=execution_residual,
                continuous_residual=equation.continuous_residual,
                annotated_residual=equation.annotated_residual,
            )
        )

    dependencies: list[tuple[int, ...]] = []
    for equation in oriented:
        target_new = _new_grid_symbols(equation, discrete.atoms, {equation.target})
        if not target_new:
            target_name = discrete.verified.problem.fields[equation.target].func.__name__
            raise CanonicalizationError(
                f"Residual for {target_name} contains no n+1 reference to its target"
            )
        fields = {
            info.field
            for symbol in _new_grid_symbols(equation, discrete.atoms)
            if isinstance((info := discrete.atoms[symbol]), GridRef)
            and info.field != equation.target
        }
        dependencies.append(tuple(sorted(fields)))
    return CanonicalSystem(discrete, tuple(oriented), tuple(dependencies))


def _strong_components(edges: Sequence[set[int]]) -> tuple[tuple[int, ...], ...]:
    index = 0
    stack: list[int] = []
    on_stack: set[int] = set()
    indices = [-1] * len(edges)
    lowlinks = [-1] * len(edges)
    components: list[tuple[int, ...]] = []

    def visit(node: int) -> None:
        nonlocal index
        indices[node] = lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in sorted(edges[node]):
            if indices[neighbor] == -1:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component: list[int] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            components.append(tuple(sorted(component)))

    for node in range(len(edges)):
        if indices[node] == -1:
            visit(node)
    return tuple(sorted(components, key=lambda component: component[0]))


def schedule(canonical: CanonicalSystem) -> ScheduledSystem:
    """Schedule SCC solve blocks into deterministic topological generations."""

    count = len(canonical.equations)
    edges: list[set[int]] = [set() for _ in range(count)]
    for target, dependencies in enumerate(canonical.dependencies):
        for dependency in dependencies:
            edges[dependency].add(target)
    components = _strong_components(edges)
    component_of = {node: index for index, component in enumerate(components) for node in component}
    condensed: list[set[int]] = [set() for _ in components]
    indegree = [0] * len(components)
    for source, neighbors in enumerate(edges):
        for destination in neighbors:
            left, right = component_of[source], component_of[destination]
            if left != right and right not in condensed[left]:
                condensed[left].add(right)
                indegree[right] += 1
    remaining = set(range(len(components)))
    layers: list[tuple[tuple[int, ...], ...]] = []
    while remaining:
        generation = tuple(sorted((index for index in remaining if indegree[index] == 0), key=lambda i: components[i]))
        if not generation:
            raise CanonicalizationError("Internal error while condensing dependency graph")
        layers.append(tuple(components[index] for index in generation))
        for index in generation:
            remaining.remove(index)
            for neighbor in condensed[index]:
                indegree[neighbor] -= 1
    return ScheduledSystem(canonical, tuple(layers))


def _is_affine(expression: sp.Expr, symbols: Sequence[sp.Symbol]) -> bool:
    symbol_set = set(symbols)
    for symbol in symbols:
        coefficient = sp.diff(expression, symbol)
        if coefficient.free_symbols & symbol_set:
            return False
    return True


def _temporal_treatment(
    expressions: Sequence[sp.Expr],
    time: sp.Symbol,
) -> TemporalTreatment:
    if not any(
        any(_time_order(derivative, time) for derivative in expression.atoms(sp.Derivative))
        for expression in expressions
    ):
        return TemporalTreatment.ALGEBRAIC
    has_old = any(expression.has(Old) for expression in expressions)
    has_new = any(expression.has(New) for expression in expressions)
    if has_new and has_old:
        return TemporalTreatment.IMEX
    if has_new:
        return TemporalTreatment.FULLY_IMPLICIT
    return TemporalTreatment.EXPLICIT


def _coefficient_table(
    equations: Sequence[_SymbolicEquation],
    fields: tuple[int, ...],
    atoms: Mapping[sp.Symbol, ExpressionIR],
) -> dict[tuple[int, int, tuple[int, ...]], sp.Expr]:
    table: dict[tuple[int, int, tuple[int, ...]], sp.Expr] = {}
    row_of = {field: row for row, field in enumerate(fields)}
    for equation in equations:
        row = row_of[equation.target]
        for symbol in _new_grid_symbols(equation, atoms, set(fields)):
            info = atoms[symbol]
            if not isinstance(info, GridRef):  # pragma: no cover - filtered above
                continue
            key = (row, row_of[info.field], info.offset)
            table[key] = sp.simplify(table.get(key, sp.S.Zero) + sp.diff(equation.residual, symbol))
    return table


def _shift_coefficient_expression(
    expression: sp.Expr,
    shift: tuple[int, ...],
    atoms: Mapping[sp.Symbol, ExpressionIR],
) -> sp.Expr | None:
    reverse = {info: symbol for symbol, info in atoms.items()}
    substitutions: dict[sp.Symbol, sp.Symbol] = {}
    for symbol in expression.free_symbols:
        info = atoms.get(symbol)
        if info is None:
            continue
        if isinstance(info, GridRef):
            shifted_info: ExpressionIR = GridRef(
                info.field,
                info.level,
                tuple(left + right for left, right in zip(info.offset, shift, strict=True)),
            )
        elif isinstance(info, CoefficientRef):
            shifted_info = CoefficientRef(
                info.name,
                info.level,
                tuple(left + right for left, right in zip(info.offset, shift, strict=True)),
            )
        elif isinstance(info, CoordinateRef):
            shifted_info = CoordinateRef(info.axis, info.offset + shift[info.axis])
        else:
            shifted_info = info
        replacement = reverse.get(shifted_info)
        if replacement is None:
            return None
        substitutions[symbol] = replacement
    return expression.xreplace(substitutions)


def _operator_properties(
    equations: Sequence[_SymbolicEquation],
    fields: tuple[int, ...],
    atoms: Mapping[sp.Symbol, ExpressionIR],
    problem: PDEProblem,
) -> OperatorProperties:
    table = _coefficient_table(equations, fields, atoms)
    if not table:
        return OperatorProperties(reason="block has no new-state linear operator")
    zero_offset = (0,) * problem.domain.ndim
    symmetric = True
    for (row, column, offset), coefficient in table.items():
        counterpart = table.get((column, row, tuple(-value for value in offset)), sp.S.Zero)
        shifted_counterpart = _shift_coefficient_expression(counterpart, offset, atoms)
        if shifted_counterpart is None or sp.simplify(coefficient - shifted_counterpart) != 0:
            symmetric = False
            break
    if not symmetric:
        return OperatorProperties(
            symmetry=PropertyState.REFUTED,
            reason="an operator offset does not match its block adjoint",
        )

    block_boundaries = tuple(
        problem.boundaries[problem.fields[field]] for field in fields
    )
    if any(isinstance(boundary, Periodic) for boundary in block_boundaries):
        coefficient_boundaries = {
            coefficient.name: coefficient.boundary
            for coefficient in problem.coefficient_fields
        }
        periodic_closure = all(
            isinstance(boundary, Periodic) for boundary in block_boundaries
        )
        for coefficient in table.values():
            for symbol in coefficient.free_symbols:
                reference = atoms.get(symbol)
                if isinstance(reference, CoordinateRef):
                    periodic_closure = False
                elif isinstance(reference, CoefficientRef):
                    periodic_closure = periodic_closure and isinstance(
                        coefficient_boundaries[reference.name], Periodic
                    )
                elif isinstance(reference, GridRef):
                    periodic_closure = periodic_closure and isinstance(
                        problem.boundaries[problem.fields[reference.field]],
                        Periodic,
                    )
        if not periodic_closure:
            return OperatorProperties(
                reason=(
                    "periodic operator closure is not proven for every "
                    "spatially varying coefficient"
                ),
            )

    diagonal_positive = True
    off_nonpositive = True
    strict_dominance = False
    row_sums: list[sp.Expr] = []
    for row in range(len(fields)):
        diagonal = sp.simplify(table.get((row, row, zero_offset), sp.S.Zero))
        if diagonal.is_positive is not True:
            diagonal_positive = False
        absolute_off = sp.S.Zero
        row_sum = sp.S.Zero
        for (item_row, column, offset), coefficient in table.items():
            if item_row != row:
                continue
            row_sum += coefficient
            if not (column == row and offset == zero_offset):
                absolute_off += sp.Abs(coefficient)
                if coefficient.is_nonpositive is not True:
                    off_nonpositive = False
        margin = sp.simplify(diagonal - absolute_off)
        strict_dominance = strict_dominance or margin.is_positive is True
        row_sums.append(sp.simplify(row_sum))

    constant_preserving = all(
        isinstance(boundary, (HomogeneousNeumann, Periodic))
        for boundary in block_boundaries
    )
    constant_nullspace = (
        all(row_sum == 0 for row_sum in row_sums) and constant_preserving
    )
    if diagonal_positive and off_nonpositive and constant_nullspace:
        return OperatorProperties(
            symmetry=PropertyState.PROVEN,
            positive_definite=PropertyState.REFUTED,
            positive_semidefinite=PropertyState.PROVEN,
            constant_nullspace=True,
            reason=(
                "symmetric graph-Laplacian form with constant-preserving "
                "boundaries"
            ),
        )
    has_dirichlet = any(isinstance(problem.boundaries[problem.fields[field]], UniformDirichlet) for field in fields)
    if diagonal_positive and off_nonpositive and (strict_dominance or has_dirichlet):
        return OperatorProperties(
            symmetry=PropertyState.PROVEN,
            positive_definite=PropertyState.PROVEN,
            positive_semidefinite=PropertyState.PROVEN,
            reason="symmetric positive M-matrix form including boundary linearization",
        )
    return OperatorProperties(
        symmetry=PropertyState.PROVEN,
        reason="symmetry is exact, but definiteness is inconclusive",
    )


def _expression_ir(
    expression: sp.Expr,
    atoms: Mapping[sp.Symbol, ExpressionIR],
    parameters: set[sp.Symbol],
    cache: dict[sp.Expr, ExpressionIR] | None = None,
) -> ExpressionIR:
    cache = {} if cache is None else cache
    cached = cache.get(expression)
    if cached is not None:
        return cached
    if isinstance(expression, sp.Symbol):
        if expression in atoms:
            result: ExpressionIR = atoms[expression]
            cache[expression] = result
            return result
        if expression in parameters:
            result = ScalarRef(expression.name)
            cache[expression] = result
            return result
        raise CompilationError(f"Symbol {expression} survived lowering without a typed reference")
    if expression.is_number or expression in (sp.pi, sp.E):
        result = Literal(sp.sstr(expression))
        cache[expression] = result
        return result
    if isinstance(expression, sp.Add):
        result = NaryOp(
            "add",
            tuple(
                _expression_ir(argument, atoms, parameters, cache)
                for argument in expression.args
            ),
        )
        cache[expression] = result
        return result
    if isinstance(expression, sp.Mul):
        result = NaryOp(
            "mul",
            tuple(
                _expression_ir(argument, atoms, parameters, cache)
                for argument in expression.args
            ),
        )
        cache[expression] = result
        return result
    if isinstance(expression, sp.Pow):
        result = Power(
            _expression_ir(expression.base, atoms, parameters, cache),
            _expression_ir(expression.exp, atoms, parameters, cache),
        )
        cache[expression] = result
        return result
    if expression.func in _SUPPORTED_FUNCTIONS:
        result = Call(
            _SUPPORTED_FUNCTIONS[expression.func],
            tuple(
                _expression_ir(argument, atoms, parameters, cache)
                for argument in expression.args
            ),
        )
        cache[expression] = result
        return result
    raise CompilationError(f"Unsupported lowered expression {expression.func}")


_IR_CONSTANT_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "abs": sp.Abs,
    "sign": sp.sign,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "minimum": sp.Min,
    "maximum": sp.Max,
}


def _constant_ir_expression(expression: ExpressionIR) -> sp.Expr | None:
    """Return an exact SymPy value for a constant-foldable IR subtree."""

    if isinstance(expression, Literal):
        return sp.sympify(expression.value)
    if isinstance(expression, NaryOp):
        values = tuple(_constant_ir_expression(argument) for argument in expression.arguments)
        if any(value is None for value in values):
            return None
        concrete = cast(tuple[sp.Expr, ...], values)
        if expression.op == "add":
            return sp.Add(*concrete)
        if expression.op == "mul":
            return sp.Mul(*concrete)
        return None
    if isinstance(expression, Power):
        base = _constant_ir_expression(expression.base)
        exponent = _constant_ir_expression(expression.exponent)
        return None if base is None or exponent is None else base**exponent
    if isinstance(expression, Call):
        function = _IR_CONSTANT_FUNCTIONS.get(expression.function)
        if function is None:
            return None
        values = tuple(_constant_ir_expression(argument) for argument in expression.arguments)
        if any(value is None for value in values):
            return None
        return cast(sp.FunctionClass, function)(*cast(tuple[sp.Expr, ...], values))
    return None


def _constant_magnitudes(
    expression: ExpressionIR,
    multiplier: sp.Expr = sp.S.One,
) -> tuple[float, ...]:
    """Collect values that become floating buffers or stencil weights."""

    values: list[float] = []
    constant = _constant_ir_expression(expression)
    if constant is not None:
        try:
            values.append(abs(float(sp.N(multiplier * constant, 17))))
        except (TypeError, ValueError, OverflowError):
            values.append(math.inf)
        return tuple(values)
    if isinstance(expression, NaryOp) and expression.op == "mul":
        constants: list[sp.Expr] = []
        dynamic: list[ExpressionIR] = []
        for argument in expression.arguments:
            value = _constant_ir_expression(argument)
            if value is None:
                dynamic.append(argument)
            else:
                constants.append(value)
        constant_product = sp.Mul(*constants) if constants else sp.S.One
        combined = sp.simplify(multiplier * constant_product)
        if constants:
            try:
                values.append(abs(float(sp.N(combined, 17))))
            except (TypeError, ValueError, OverflowError):
                values.append(math.inf)
        target = next(
            (
                argument
                for argument in dynamic
                if isinstance(argument, StencilApply)
            ),
            None,
        )
        if target is None:
            target = next(
                (
                    argument
                    for argument in dynamic
                    if isinstance(argument, NaryOp) and argument.op == "add"
                ),
                None,
            )
        for argument in dynamic:
            values.extend(
                _constant_magnitudes(
                    argument,
                    combined if argument is target else sp.S.One,
                )
            )
        return tuple(values)
    if isinstance(expression, StencilApply):
        for _, coefficient in expression.entries:
            try:
                value = multiplier * sp.sympify(coefficient)
                values.append(abs(float(sp.N(value, 17))))
            except (TypeError, ValueError, OverflowError):
                values.append(math.inf)
    children = _expression_children(expression)
    if isinstance(expression, NaryOp) and expression.op == "add":
        for child in children:
            values.extend(_constant_magnitudes(child, multiplier))
    else:
        if multiplier != 1:
            try:
                values.append(abs(float(sp.N(multiplier, 17))))
            except (TypeError, ValueError, OverflowError):
                values.append(math.inf)
        for child in children:
            values.extend(_constant_magnitudes(child))
    return tuple(values)


def _scaled_ir_expression(expression: ExpressionIR, scale: sp.Expr) -> ExpressionIR:
    """Multiply an expression by *scale*, pushing it into static coefficients."""

    if scale == 1:
        return expression
    if isinstance(expression, NaryOp) and expression.op == "add":
        return _execution_add(
            _scaled_ir_expression(argument, scale)
            for argument in expression.arguments
        )
    if isinstance(expression, StencilApply):
        return StencilApply(
            tuple(
                (offset, sp.sstr(sp.simplify(sp.sympify(coefficient) * scale)))
                for offset, coefficient in expression.entries
            ),
            expression.expression,
        )
    if isinstance(expression, NaryOp) and expression.op == "mul":
        arguments = list(expression.arguments)
        constant_indices: list[int] = []
        constant_product = sp.S.One
        for index, argument in enumerate(arguments):
            constant = _constant_ir_expression(argument)
            if constant is not None:
                constant_indices.append(index)
                constant_product *= constant
        for index, argument in enumerate(arguments):
            if isinstance(argument, StencilApply):
                combined_scale = sp.simplify(constant_product * scale)
                arguments[index] = _scaled_ir_expression(
                    argument, combined_scale
                )
                arguments = [
                    argument
                    for item_index, argument in enumerate(arguments)
                    if item_index not in constant_indices
                ]
                return _execution_mul(arguments)
        for index, argument in enumerate(arguments):
            if isinstance(argument, NaryOp) and argument.op == "add":
                combined_scale = sp.simplify(constant_product * scale)
                arguments[index] = _scaled_ir_expression(
                    argument, combined_scale
                )
                arguments = [
                    argument
                    for item_index, argument in enumerate(arguments)
                    if item_index not in constant_indices
                ]
                return _execution_mul(arguments)
        if constant_indices:
            first = constant_indices[0]
            arguments[first] = Literal(
                sp.sstr(sp.simplify(constant_product * scale))
            )
            arguments = [
                argument
                for item_index, argument in enumerate(arguments)
                if item_index == first or item_index not in constant_indices
            ]
            return _execution_mul(arguments)
    constant = _constant_ir_expression(expression)
    if constant is not None:
        return Literal(sp.sstr(sp.simplify(constant * scale)))
    return _execution_mul((Literal(sp.sstr(scale)), expression))


def _block_residual_scale(
    expressions: Sequence[ExpressionIR], *, normalize: bool = False
) -> sp.Expr:
    """Normalize affine residual coefficients with an exact power of two.

    Merely keeping each coefficient representable in FP16 is insufficient: a
    representable coefficient can still overflow when it is multiplied by a
    perfectly ordinary state value.  Krylov convergence is invariant under a
    common residual scaling, so iterative blocks keep the largest static
    coefficient at most one. Direct and pointwise blocks retain their exact IR
    unless a coefficient would exceed FP16 range. A power-of-two scale is exact
    in every supported binary dtype.
    """

    magnitudes = tuple(
        value
        for expression in expressions
        for value in _constant_magnitudes(expression)
        if value != 0.0
    )
    largest = max(magnitudes, default=1.0)
    limit = 1.0 if normalize else 65504.0
    if math.isfinite(largest) and largest <= limit:
        return sp.S.One
    if not math.isfinite(largest):
        raise CompilationError("An affine block contains a non-finite constant-folded coefficient")
    exponent = math.ceil(math.log2(largest))
    return sp.Rational(1, 2**exponent)


def characterize(scheduled: ScheduledSystem) -> ProgramIR:
    """Characterize scheduled blocks and lower their residuals to ProgramIR."""

    canonical = scheduled.canonical
    discrete = canonical.discrete
    verified = discrete.verified
    problem = verified.problem
    parameter_symbols = {parameter.symbol for parameter in problem.parameters}
    expression_cache: dict[sp.Expr, ExpressionIR] = {}
    equation_ir: list[DiscreteEquationIR] = []
    for equation in canonical.equations:
        center_symbols = [
            symbol
            for symbol, info in discrete.atoms.items()
            if (
                isinstance(info, GridRef)
                and info.field == equation.target
                and info.level == TimeLevel.NEW
                and info.offset == (0,) * problem.domain.ndim
                and symbol in equation.residual.free_symbols
            )
        ]
        diagonal_expression = None
        if center_symbols:
            diagonal = sp.simplify(sp.diff(equation.residual, center_symbols[0]))
            diagonal_expression = _expression_ir(
                diagonal, discrete.atoms, parameter_symbols, expression_cache
            )
        equation_ir.append(
            DiscreteEquationIR(
                target=equation.target,
                residual=equation.execution_residual,
                continuous_residual=equation.continuous_residual,
                diagonal=diagonal_expression,
            )
        )

    layers: list[ScheduleLayerIR] = []
    block_id = 0
    for layer in scheduled.layers:
        blocks: list[SolveBlockIR] = []
        for fields in layer:
            equations = tuple(canonical.equations[field] for field in fields)
            new_symbols = tuple(
                dict.fromkeys(
                    symbol
                    for equation in equations
                    for symbol in _new_grid_symbols(equation, discrete.atoms, set(fields))
                )
            )
            affine = all(_is_affine(equation.residual, new_symbols) for equation in equations)
            center_only = all(
                cast(GridRef, discrete.atoms[symbol]).offset == (0,) * problem.domain.ndim
                for symbol in new_symbols
            )
            if affine and center_only and len(fields) == 1:
                kind = BlockKind.DIRECT
            elif affine and center_only:
                kind = BlockKind.POINTWISE_LINEAR
            elif affine:
                kind = BlockKind.LINEAR_IMPLICIT
            else:
                kind = BlockKind.NONLINEAR_IMPLICIT
            properties = (
                _operator_properties(equations, fields, discrete.atoms, problem)
                if affine
                else OperatorProperties(reason="nonlinear residual")
            )
            treatment = _temporal_treatment(
                [equation.annotated_residual for equation in equations],
                problem.time,
            )
            pointwise_coefficients = None
            if kind == BlockKind.POINTWISE_LINEAR:
                table = _coefficient_table(equations, fields, discrete.atoms)
                zero_offset = (0,) * problem.domain.ndim
                pointwise_coefficients = tuple(
                    tuple(
                        _expression_ir(
                            table.get(
                                (row, column, zero_offset), sp.S.Zero
                            ),
                            discrete.atoms,
                            parameter_symbols,
                            expression_cache,
                        )
                        for column in range(len(fields))
                    )
                    for row in range(len(fields))
                )
            block_expressions: list[ExpressionIR] = []
            for equation_index in fields:
                equation = equation_ir[equation_index]
                block_expressions.append(equation.residual)
                if equation.diagonal is not None:
                    block_expressions.append(equation.diagonal)
            if pointwise_coefficients is not None:
                block_expressions.extend(
                    expression
                    for row in pointwise_coefficients
                    for expression in row
                )
            residual_scale = (
                _block_residual_scale(
                    block_expressions,
                    normalize=kind == BlockKind.LINEAR_IMPLICIT,
                )
                if affine
                else sp.S.One
            )
            if residual_scale != 1:
                for equation_index in fields:
                    equation = equation_ir[equation_index]
                    equation_ir[equation_index] = DiscreteEquationIR(
                        target=equation.target,
                        residual=_scaled_ir_expression(
                            equation.residual, residual_scale
                        ),
                        continuous_residual=equation.continuous_residual,
                        diagonal=(
                            _scaled_ir_expression(
                                equation.diagonal, residual_scale
                            )
                            if equation.diagonal is not None
                            else None
                        ),
                    )
                if pointwise_coefficients is not None:
                    pointwise_coefficients = tuple(
                        tuple(
                            _scaled_ir_expression(expression, residual_scale)
                            for expression in row
                        )
                        for row in pointwise_coefficients
                    )
            if kind == BlockKind.DIRECT:
                equation_index = fields[0]
                equation = equation_ir[equation_index]
                direct_solution = _direct_stencil_solution(
                    equation.residual,
                    equation.diagonal,
                    equation.target,
                )
                if direct_solution is not None:
                    equation_ir[equation_index] = DiscreteEquationIR(
                        target=equation.target,
                        residual=equation.residual,
                        continuous_residual=equation.continuous_residual,
                        diagonal=equation.diagonal,
                        direct_solution=direct_solution,
                    )
            blocks.append(
                SolveBlockIR(
                    id=block_id,
                    targets=fields,
                    equations=fields,
                    kind=kind,
                    temporal_treatment=treatment,
                    properties=properties,
                    pointwise_coefficients=pointwise_coefficients,
                    residual_scale=sp.sstr(residual_scale),
                )
            )
            block_id += 1
        layers.append(ScheduleLayerIR(tuple(blocks)))

    option_items: tuple[tuple[str, object], ...] = tuple(
        sorted(
            {
                "krylov_max_iterations": verified.options.krylov.max_iterations,
                "krylov_rtol_float16": verified.options.krylov.rtol_float16,
                "krylov_rtol_bfloat16": verified.options.krylov.rtol_bfloat16,
                "krylov_rtol_float32": verified.options.krylov.rtol_float32,
                "krylov_rtol_float64": verified.options.krylov.rtol_float64,
                "krylov_atol": verified.options.krylov.atol,
                "newton_max_iterations": verified.options.newton.max_iterations,
                "newton_max_backtracks": verified.options.newton.max_backtracks,
                "newton_backtrack_factor": verified.options.newton.backtrack_factor,
                "newton_armijo_constant": verified.options.newton.armijo_constant,
                "newton_atol_float32": verified.options.newton.atol_float32,
                "newton_atol_float64": verified.options.newton.atol_float64,
            }.items()
        )
    )
    def boundary_ir(boundary: BoundaryCondition) -> BoundaryIR:
        if isinstance(boundary, UniformDirichlet):
            return BoundaryIR("dirichlet", boundary.value)
        if isinstance(boundary, Periodic):
            return BoundaryIR("periodic")
        return BoundaryIR("neumann")

    return ProgramIR(
        version=PROGRAM_IR_VERSION,
        name=problem.name,
        fields=tuple(field.func.__name__ for field in problem.fields),
        axes=tuple(symbol.name for symbol in problem.domain.symbols),
        bounds=tuple((sp.sstr(axis.lower), sp.sstr(axis.upper)) for axis in problem.domain.axes),
        grid_shape=verified.grid.shape,
        dt=sp.sstr(problem.discretization.dt),
        boundaries=tuple(boundary_ir(problem.boundaries[field]) for field in problem.fields),
        parameters=tuple(parameter.name for parameter in problem.parameters),
        coefficient_fields=tuple(coefficient.name for coefficient in problem.coefficient_fields),
        coefficient_boundaries=tuple(
            boundary_ir(coefficient.boundary)
            for coefficient in problem.coefficient_fields
        ),
        equations=tuple(equation_ir),
        schedule=tuple(layers),
        options=option_items,
    )


def compile_program(
    problem: PDEProblem,
    grid: UniformGrid,
    options: CompileOptions | None = None,
) -> ProgramIR:
    """Run all pure compiler passes and return immutable ProgramIR."""

    return characterize(schedule(canonicalize(discretize(verify(problem, grid, options)))))


def compile_solver(
    problem: PDEProblem,
    grid: UniformGrid,
    options: CompileOptions | None = None,
):
    """Compile a PDE problem into the primary :class:`CompiledSolver` artifact."""

    from .runtime import CompiledSolver

    return CompiledSolver(compile_program(problem, grid, options))


__all__ = [
    "CanonicalSystem",
    "DiscreteSystem",
    "ScheduledSystem",
    "VerifiedModel",
    "canonicalize",
    "characterize",
    "compile_program",
    "compile_solver",
    "discretize",
    "schedule",
    "verify",
]
