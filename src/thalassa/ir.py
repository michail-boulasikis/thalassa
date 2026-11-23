"""Frozen, versioned intermediate representation shared by compiler and runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, TypeAlias

PROGRAM_IR_VERSION = "2.0"


class TimeLevel(str, Enum):
    OLD = "old"
    NEW = "new"


class PropertyState(str, Enum):
    PROVEN = "proven"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


class BlockKind(str, Enum):
    DIRECT = "direct"
    POINTWISE_LINEAR = "pointwise_linear"
    LINEAR_IMPLICIT = "linear_implicit"
    NONLINEAR_IMPLICIT = "nonlinear_implicit"


class TemporalTreatment(str, Enum):
    EXPLICIT = "explicit"
    IMEX = "imex"
    FULLY_IMPLICIT = "fully_implicit"
    ALGEBRAIC = "algebraic"


@dataclass(frozen=True)
class Literal:
    value: str


@dataclass(frozen=True)
class GridRef:
    field: int
    level: TimeLevel
    offset: tuple[int, ...]


@dataclass(frozen=True)
class ScalarRef:
    name: str


@dataclass(frozen=True)
class CoefficientRef:
    name: str
    level: TimeLevel
    offset: tuple[int, ...]


@dataclass(frozen=True)
class CoordinateRef:
    axis: int
    offset: int = 0


@dataclass(frozen=True)
class TimeRef:
    level: TimeLevel


@dataclass(frozen=True)
class NaryOp:
    op: str
    arguments: tuple[ExpressionIR, ...]


@dataclass(frozen=True)
class Power:
    base: ExpressionIR
    exponent: ExpressionIR


@dataclass(frozen=True)
class Call:
    function: str
    arguments: tuple[ExpressionIR, ...]


@dataclass(frozen=True)
class StencilApply:
    """Typed stencil node retained for custom/programmatic IR construction."""

    entries: tuple[tuple[tuple[int, ...], str], ...]
    expression: ExpressionIR


ExpressionIR: TypeAlias = (
    Literal
    | GridRef
    | ScalarRef
    | CoefficientRef
    | CoordinateRef
    | TimeRef
    | NaryOp
    | Power
    | Call
    | StencilApply
)


def _expression_children(expression: ExpressionIR) -> tuple[ExpressionIR, ...]:
    if isinstance(expression, (NaryOp, Call)):
        return expression.arguments
    if isinstance(expression, Power):
        return (expression.base, expression.exponent)
    if isinstance(expression, StencilApply):
        return (expression.expression,)
    return ()


@dataclass(frozen=True)
class OperatorProperties:
    symmetry: PropertyState = PropertyState.UNKNOWN
    positive_definite: PropertyState = PropertyState.UNKNOWN
    positive_semidefinite: PropertyState = PropertyState.UNKNOWN
    constant_nullspace: bool = False
    reason: str = "operator properties were not proven"


@dataclass(frozen=True)
class DiscreteEquationIR:
    target: int
    residual: ExpressionIR
    continuous_residual: str
    diagonal: ExpressionIR | None = None
    direct_solution: ExpressionIR | None = None


@dataclass(frozen=True)
class SolveBlockIR:
    id: int
    targets: tuple[int, ...]
    equations: tuple[int, ...]
    kind: BlockKind
    temporal_treatment: TemporalTreatment
    properties: OperatorProperties
    pointwise_coefficients: tuple[tuple[ExpressionIR, ...], ...] | None = None
    residual_scale: str = "1"

    def __post_init__(self) -> None:
        try:
            value = float(Fraction(self.residual_scale))
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
            raise ValueError("SolveBlockIR residual_scale must be a finite positive number") from error
        if not (value > 0.0 and math.isfinite(value)):
            raise ValueError("SolveBlockIR residual_scale must be a finite positive number")
        coefficients = self.pointwise_coefficients
        if coefficients is None:
            return
        if self.kind != BlockKind.POINTWISE_LINEAR:
            raise ValueError(
                "Pointwise coefficients are only valid for pointwise-linear blocks"
            )
        size = len(self.targets)
        if len(coefficients) != size or any(
            len(row) != size for row in coefficients
        ):
            raise ValueError(
                "Pointwise coefficient matrix dimensions must match block targets"
            )
        target_set = frozenset(self.targets)
        for row in coefficients:
            for expression in row:
                pending = [expression]
                while pending:
                    node = pending.pop()
                    if (
                        isinstance(node, GridRef)
                        and node.level == TimeLevel.NEW
                        and node.field in target_set
                    ):
                        raise ValueError(
                            "Pointwise coefficients cannot depend on the block's "
                            "new-state unknowns"
                        )
                    pending.extend(_expression_children(node))


@dataclass(frozen=True)
class ScheduleLayerIR:
    blocks: tuple[SolveBlockIR, ...]


@dataclass(frozen=True)
class BoundaryIR:
    kind: str
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"dirichlet", "neumann", "periodic"}:
            raise ValueError(f"Unknown boundary kind {self.kind!r}")
        if self.kind != "dirichlet" and self.value != 0.0:
            raise ValueError(f"Boundary kind {self.kind!r} cannot carry a value")


@dataclass(frozen=True)
class ProgramIR:
    version: str
    name: str
    fields: tuple[str, ...]
    axes: tuple[str, ...]
    bounds: tuple[tuple[str, str], ...]
    grid_shape: tuple[int, ...]
    dt: str
    boundaries: tuple[BoundaryIR, ...]
    parameters: tuple[str, ...]
    coefficient_fields: tuple[str, ...]
    coefficient_boundaries: tuple[BoundaryIR, ...]
    equations: tuple[DiscreteEquationIR, ...]
    schedule: tuple[ScheduleLayerIR, ...]
    options: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if self.version != PROGRAM_IR_VERSION:
            raise ValueError(f"Unsupported ProgramIR version {self.version!r}")
        if len(self.boundaries) != len(self.fields):
            raise ValueError("ProgramIR boundaries must align with fields")
        if len(self.coefficient_boundaries) != len(self.coefficient_fields):
            raise ValueError(
                "ProgramIR coefficient boundaries must align with coefficient fields"
            )

    def to_dict(self) -> dict[str, Any]:
        return program_to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProgramIR:
        return program_from_dict(value)


@dataclass(frozen=True)
class SolverMetadata:
    version: str
    name: str
    field_order: tuple[str, ...]
    axis_order: tuple[str, ...]
    grid_shape: tuple[int, ...]
    spacing: tuple[float, ...]
    parameter_order: tuple[str, ...]
    coefficient_field_order: tuple[str, ...]
    schedule: tuple[tuple[tuple[str, ...], ...], ...]


def expression_to_dict(expression: ExpressionIR) -> dict[str, Any]:
    if isinstance(expression, Literal):
        return {"type": "literal", "value": expression.value}
    if isinstance(expression, GridRef):
        return {
            "type": "grid_ref",
            "field": expression.field,
            "level": expression.level.value,
            "offset": list(expression.offset),
        }
    if isinstance(expression, ScalarRef):
        return {"type": "scalar_ref", "name": expression.name}
    if isinstance(expression, CoefficientRef):
        return {
            "type": "coefficient_ref",
            "name": expression.name,
            "level": expression.level.value,
            "offset": list(expression.offset),
        }
    if isinstance(expression, CoordinateRef):
        return {"type": "coordinate_ref", "axis": expression.axis, "offset": expression.offset}
    if isinstance(expression, TimeRef):
        return {"type": "time_ref", "level": expression.level.value}
    if isinstance(expression, NaryOp):
        return {
            "type": "nary",
            "op": expression.op,
            "arguments": [expression_to_dict(argument) for argument in expression.arguments],
        }
    if isinstance(expression, Power):
        return {
            "type": "power",
            "base": expression_to_dict(expression.base),
            "exponent": expression_to_dict(expression.exponent),
        }
    if isinstance(expression, Call):
        return {
            "type": "call",
            "function": expression.function,
            "arguments": [expression_to_dict(argument) for argument in expression.arguments],
        }
    if isinstance(expression, StencilApply):
        return {
            "type": "stencil",
            "entries": [[list(offset), coefficient] for offset, coefficient in expression.entries],
            "expression": expression_to_dict(expression.expression),
        }
    raise TypeError(f"Unsupported expression node {type(expression).__name__}")


def expression_from_dict(value: Mapping[str, Any]) -> ExpressionIR:
    node_type = value["type"]
    if node_type == "literal":
        return Literal(str(value["value"]))
    if node_type == "grid_ref":
        return GridRef(int(value["field"]), TimeLevel(value["level"]), tuple(value["offset"]))
    if node_type == "scalar_ref":
        return ScalarRef(str(value["name"]))
    if node_type == "coefficient_ref":
        return CoefficientRef(str(value["name"]), TimeLevel(value["level"]), tuple(value["offset"]))
    if node_type == "coordinate_ref":
        return CoordinateRef(int(value["axis"]), int(value["offset"]))
    if node_type == "time_ref":
        return TimeRef(TimeLevel(value["level"]))
    if node_type == "nary":
        return NaryOp(str(value["op"]), tuple(expression_from_dict(item) for item in value["arguments"]))
    if node_type == "power":
        return Power(expression_from_dict(value["base"]), expression_from_dict(value["exponent"]))
    if node_type == "call":
        return Call(str(value["function"]), tuple(expression_from_dict(item) for item in value["arguments"]))
    if node_type == "stencil":
        entries = tuple((tuple(item[0]), str(item[1])) for item in value["entries"])
        return StencilApply(entries, expression_from_dict(value["expression"]))
    raise ValueError(f"Unknown expression node type {node_type!r}")


def _properties_to_dict(properties: OperatorProperties) -> dict[str, Any]:
    return {
        "symmetry": properties.symmetry.value,
        "positive_definite": properties.positive_definite.value,
        "positive_semidefinite": properties.positive_semidefinite.value,
        "constant_nullspace": properties.constant_nullspace,
        "reason": properties.reason,
    }


def _properties_from_dict(value: Mapping[str, Any]) -> OperatorProperties:
    return OperatorProperties(
        symmetry=PropertyState(value["symmetry"]),
        positive_definite=PropertyState(value["positive_definite"]),
        positive_semidefinite=PropertyState(value["positive_semidefinite"]),
        constant_nullspace=bool(value["constant_nullspace"]),
        reason=str(value["reason"]),
    )


def program_to_dict(program: ProgramIR) -> dict[str, Any]:
    return {
        "version": program.version,
        "name": program.name,
        "fields": list(program.fields),
        "axes": list(program.axes),
        "bounds": [list(bound) for bound in program.bounds],
        "grid_shape": list(program.grid_shape),
        "dt": program.dt,
        "boundaries": [{"kind": bc.kind, "value": bc.value} for bc in program.boundaries],
        "parameters": list(program.parameters),
        "coefficient_fields": list(program.coefficient_fields),
        "coefficient_boundaries": [
            {"kind": bc.kind, "value": bc.value}
            for bc in program.coefficient_boundaries
        ],
        "equations": [
            {
                "target": equation.target,
                "residual": expression_to_dict(equation.residual),
                "continuous_residual": equation.continuous_residual,
                "diagonal": expression_to_dict(equation.diagonal) if equation.diagonal is not None else None,
                "direct_solution": (
                    expression_to_dict(equation.direct_solution)
                    if equation.direct_solution is not None
                    else None
                ),
            }
            for equation in program.equations
        ],
        "schedule": [
            {
                "blocks": [
                    {
                        **{
                            "id": block.id,
                            "targets": list(block.targets),
                            "equations": list(block.equations),
                            "kind": block.kind.value,
                            "temporal_treatment": block.temporal_treatment.value,
                            "properties": _properties_to_dict(block.properties),
                            "residual_scale": block.residual_scale,
                        },
                        **(
                            {
                                "pointwise_coefficients": [
                                    [
                                        expression_to_dict(expression)
                                        for expression in row
                                    ]
                                    for row in block.pointwise_coefficients
                                ]
                            }
                            if block.pointwise_coefficients is not None
                            else {}
                        ),
                    }
                    for block in layer.blocks
                ]
            }
            for layer in program.schedule
        ],
        "options": {key: value for key, value in program.options},
    }


def program_from_dict(value: Mapping[str, Any]) -> ProgramIR:
    coefficient_fields = tuple(value["coefficient_fields"])
    coefficient_boundary_values = value.get("coefficient_boundaries")
    if coefficient_boundary_values is None:
        coefficient_boundaries = tuple(
            BoundaryIR("neumann") for _ in coefficient_fields
        )
    else:
        coefficient_boundaries = tuple(
            BoundaryIR(str(item["kind"]), float(item.get("value", 0.0)))
            for item in coefficient_boundary_values
        )
    equations = tuple(
        DiscreteEquationIR(
            target=int(item["target"]),
            residual=expression_from_dict(item["residual"]),
            continuous_residual=str(item["continuous_residual"]),
            diagonal=expression_from_dict(item["diagonal"]) if item.get("diagonal") is not None else None,
            direct_solution=(
                expression_from_dict(item["direct_solution"])
                if item.get("direct_solution") is not None
                else None
            ),
        )
        for item in value["equations"]
    )
    schedule = tuple(
        ScheduleLayerIR(
            tuple(
                SolveBlockIR(
                    id=int(block["id"]),
                    targets=tuple(block["targets"]),
                    equations=tuple(block["equations"]),
                    kind=BlockKind(block["kind"]),
                    temporal_treatment=TemporalTreatment(block["temporal_treatment"]),
                    properties=_properties_from_dict(block["properties"]),
                    pointwise_coefficients=(
                        tuple(
                            tuple(expression_from_dict(expression) for expression in row)
                            for row in block["pointwise_coefficients"]
                        )
                        if block.get("pointwise_coefficients") is not None
                        else None
                    ),
                    residual_scale=str(block.get("residual_scale", "1")),
                )
                for block in layer["blocks"]
            )
        )
        for layer in value["schedule"]
    )
    options_value = value.get("options", {})
    options = tuple(sorted(options_value.items())) if isinstance(options_value, Mapping) else tuple(options_value)
    return ProgramIR(
        version=str(value["version"]),
        name=str(value["name"]),
        fields=tuple(value["fields"]),
        axes=tuple(value["axes"]),
        bounds=tuple((str(item[0]), str(item[1])) for item in value["bounds"]),
        grid_shape=tuple(value["grid_shape"]),
        dt=str(value["dt"]),
        boundaries=tuple(BoundaryIR(str(item["kind"]), float(item.get("value", 0.0))) for item in value["boundaries"]),
        parameters=tuple(value["parameters"]),
        coefficient_fields=coefficient_fields,
        coefficient_boundaries=coefficient_boundaries,
        equations=equations,
        schedule=schedule,
        options=options,
    )


__all__ = [
    "BlockKind",
    "BoundaryIR",
    "Call",
    "CoefficientRef",
    "CoordinateRef",
    "DiscreteEquationIR",
    "ExpressionIR",
    "GridRef",
    "Literal",
    "NaryOp",
    "OperatorProperties",
    "PROGRAM_IR_VERSION",
    "Power",
    "ProgramIR",
    "PropertyState",
    "ScalarRef",
    "ScheduleLayerIR",
    "SolveBlockIR",
    "SolverMetadata",
    "StencilApply",
    "TemporalTreatment",
    "TimeLevel",
    "TimeRef",
    "expression_from_dict",
    "expression_to_dict",
    "program_from_dict",
    "program_to_dict",
]
