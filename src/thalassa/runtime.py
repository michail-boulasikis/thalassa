"""PyTorch execution runtime for versioned Thalassa programs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias, cast

import sympy as sp
import torch
import torch.nn.functional as torch_functional

from .exceptions import (
    BoundaryCompatibilityError,
    LinearSolveError,
    NonlinearSolveError,
)
from .ir import (
    BlockKind,
    BoundaryIR,
    Call,
    CoefficientRef,
    CoordinateRef,
    ExpressionIR,
    GridRef,
    Literal,
    NaryOp,
    Power,
    ProgramIR,
    PropertyState,
    ScalarRef,
    SolverMetadata,
    StencilApply,
    TimeLevel,
    TimeRef,
    _expression_children,
    program_from_dict,
)

TensorOperator: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
EvaluationValue: TypeAlias = torch.Tensor | float


def _exact_float(value: str) -> float:
    return float(sp.N(sp.sympify(value), 17))


class SolveOutput(NamedTuple):
    """Compile-friendly solver output with tensor-valued convergence summaries."""

    state: torch.Tensor
    converged: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    relative_residual: torch.Tensor
    tolerance_ratio: torch.Tensor
    precision_limited: torch.Tensor

    @property
    def final_state(self) -> torch.Tensor:
        return self.state


@dataclass(frozen=True)
class SolveDiagnostics:
    step_index: int
    block_id: int
    block_kind: str
    targets: tuple[str, ...]
    converged: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    breakdown: torch.Tensor
    relative_residual: torch.Tensor
    tolerance_ratio: torch.Tensor
    precision_limited: torch.Tensor


@dataclass(frozen=True)
class SolveResult:
    state: torch.Tensor
    diagnostics: tuple[SolveDiagnostics, ...]
    converged: bool

    @property
    def final_state(self) -> torch.Tensor:
        return self.state


class KrylovResult(NamedTuple):
    solution: torch.Tensor
    converged: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    breakdown: torch.Tensor
    relative_residual: torch.Tensor
    tolerance_ratio: torch.Tensor
    precision_limited: torch.Tensor


@dataclass
class _EvaluationSession:
    mutable_new_fields: frozenset[int]
    invariant_cache: dict[tuple[ExpressionIR, tuple[int, ...]], EvaluationValue]
    dependency_cache: dict[ExpressionIR, bool]

    def __init__(self, mutable_new_fields: Sequence[int]) -> None:
        self.mutable_new_fields = frozenset(mutable_new_fields)
        self.invariant_cache = {}
        self.dependency_cache = {}


@dataclass(frozen=True)
class _PointwiseStrategy:
    kind: str
    lu: tuple[tuple[float, ...], ...] = ()
    pivots: tuple[int, ...] = ()


def _factor_constant_pointwise_matrix(
    matrix: Sequence[Sequence[float]],
) -> _PointwiseStrategy:
    if any(not math.isfinite(value) for row in matrix for value in row):
        return _PointwiseStrategy("constant_breakdown")
    tensor = torch.tensor(matrix, dtype=torch.float64, device="cpu")
    lu, pivots, info = torch.linalg.lu_factor_ex(tensor, check_errors=False)
    if int(info) != 0 or not bool(torch.all(torch.isfinite(lu))):
        return _PointwiseStrategy("constant_breakdown")
    return _PointwiseStrategy(
        "constant_lu",
        tuple(tuple(float(value) for value in row) for row in lu.tolist()),
        tuple(int(value) - 1 for value in pivots.tolist()),
    )


def _stencil_radius(
    entries: Sequence[tuple[tuple[int, ...], str]],
) -> tuple[int, ...]:
    return tuple(
        max(abs(offset[axis]) for offset, _ in entries)
        for axis in range(len(entries[0][0]))
    )


def _dense_stencil_kernel(
    entries: Sequence[tuple[tuple[int, ...], str]],
) -> torch.Tensor:
    radii = _stencil_radius(entries)
    shape = tuple(2 * radius + 1 for radius in radii)
    kernel = torch.zeros(shape, dtype=torch.float64)
    for offset, coefficient in entries:
        index = tuple(
            item + radius for item, radius in zip(offset, radii, strict=True)
        )
        kernel[index] += _exact_float(coefficient)
    return kernel


def _depends_on_new_fields(
    expression: ExpressionIR,
    fields: frozenset[int],
    cache: dict[ExpressionIR, bool],
) -> bool:
    cached = cache.get(expression)
    if cached is not None:
        return cached
    if isinstance(expression, GridRef):
        result = expression.level == TimeLevel.NEW and expression.field in fields
    else:
        result = any(
            _depends_on_new_fields(child, fields, cache)
            for child in _expression_children(expression)
        )
    cache[expression] = result
    return result


def _spatial_dimensions(tensor: torch.Tensor, spatial_ndim: int) -> tuple[int, ...]:
    return tuple(range(tensor.ndim - spatial_ndim, tensor.ndim))


def _boundary_axis_map(
    logical: torch.Tensor,
    cells: int,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map logical cell indices to storage indices and reflected locations."""

    if kind == "periodic":
        return torch.remainder(logical, cells), torch.zeros_like(
            logical, dtype=torch.bool
        )
    if kind not in {"dirichlet", "neumann"}:
        raise ValueError(f"Unknown boundary kind {kind!r}")
    if logical.numel() and (
        int(logical.min()) < -cells or int(logical.max()) >= 2 * cells
    ):
        raise ValueError("Reflected region exceeds the axis size")
    reflected = (logical < 0) | (logical >= cells)
    gathered = torch.where(
        logical < 0,
        -logical - 1,
        torch.where(logical >= cells, 2 * cells - logical - 1, logical),
    )
    return gathered, reflected


def extend_ghosts(
    tensor: torch.Tensor,
    radii: Sequence[int],
    *,
    kind: str,
    value: float = 0.0,
) -> torch.Tensor:
    """Extend cell-centered data axis by axis with reflected or periodic ghosts.

    ``radii`` follows the physical axis order, not PyTorch's reversed padding
    convention.  The function supports arbitrary leading batch dimensions.
    """

    radii = tuple(int(radius) for radius in radii)
    spatial_ndim = len(radii)
    if tensor.ndim < spatial_ndim:
        raise ValueError("Tensor has fewer dimensions than the requested ghost extension")
    if kind not in {"dirichlet", "neumann", "periodic"}:
        raise ValueError(f"Unknown boundary kind {kind!r}")
    result = tensor
    for axis, radius in enumerate(radii):
        if radius < 0:
            raise ValueError("Ghost radii must be nonnegative")
        if radius == 0:
            continue
        dimension = result.ndim - spatial_ndim + axis
        cells = result.shape[dimension]
        if radius > cells:
            raise ValueError(f"Ghost radius {radius} exceeds axis size {cells}")
        logical = torch.arange(
            -radius,
            cells + radius,
            dtype=torch.int32,
            device=result.device,
        )
        indices, reflected = _boundary_axis_map(logical, cells, kind)
        result = torch.index_select(result, dimension, indices)
        if kind == "dirichlet":
            boundary_value = torch.as_tensor(value, dtype=result.dtype, device=result.device)
            shape = [1] * result.ndim
            shape[dimension] = reflected.numel()
            reflected_region = reflected.reshape(shape)
            result = torch.where(
                reflected_region,
                2 * boundary_value - result,
                result,
            )
    return result


def ghost_shift(
    tensor: torch.Tensor,
    offset: Sequence[int],
    boundary: BoundaryIR | str,
    value: float = 0.0,
) -> torch.Tensor:
    """Evaluate a cell-centered tensor at an integer spatial offset."""

    offset = tuple(int(item) for item in offset)
    kind = boundary.kind if isinstance(boundary, BoundaryIR) else boundary
    boundary_value = boundary.value if isinstance(boundary, BoundaryIR) else value
    radii = tuple(abs(item) for item in offset)
    result = extend_ghosts(tensor, radii, kind=kind, value=boundary_value)
    spatial_ndim = len(offset)
    slices: list[slice] = [slice(None)] * result.ndim
    for axis, (radius, shift) in enumerate(zip(radii, offset, strict=True)):
        dimension = result.ndim - spatial_ndim + axis
        start = radius + shift
        size = tensor.shape[tensor.ndim - spatial_ndim + axis]
        slices[dimension] = slice(start, start + size)
    return result[tuple(slices)]


def _batch_expand(value: torch.Tensor, tensor: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    return value.reshape((*value.shape, *((1,) * (tensor.ndim - batch_ndim))))


def _dot(left: torch.Tensor, right: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    dimensions = tuple(range(batch_ndim, left.ndim))
    if left.dtype in (torch.float16, torch.bfloat16):
        left = left.to(torch.float32)
    if right.dtype in (torch.float16, torch.bfloat16):
        right = right.to(torch.float32)
    return torch.sum(left * right, dim=dimensions)


def _norm(value: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.to(torch.float32)
    dimensions = tuple(range(batch_ndim, value.ndim))
    if not dimensions:
        return torch.abs(value)
    return torch.linalg.vector_norm(value, dim=dimensions)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide diagnostics without turning an exact zero-over-zero into NaN."""

    positive = denominator > 0
    safe_denominator = torch.where(positive, denominator, 1.0)
    ratio = numerator / safe_denominator
    infinity = torch.full_like(ratio, torch.inf)
    return torch.where(positive, ratio, torch.where(numerator == 0, 0.0, infinity))


def _linear_diagnostics(
    rhs: torch.Tensor,
    applied: torch.Tensor,
    residual_norm: torch.Tensor,
    batch_ndim: int,
    base_threshold: torch.Tensor,
    effective_threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scale-free equation defect and convergence-criterion usage."""

    balance = _norm(rhs, batch_ndim) + _norm(applied, batch_ndim)
    relative_residual = _safe_ratio(residual_norm, balance)
    tolerance_ratio = _safe_ratio(residual_norm, effective_threshold)
    precision_limited = (
        torch.isfinite(residual_norm)
        & (residual_norm > base_threshold)
        & (residual_norm <= effective_threshold)
    )
    return relative_residual, tolerance_ratio, precision_limited


def _algebraic_diagnostics(
    rhs: torch.Tensor,
    applied: torch.Tensor,
    residual_norm: torch.Tensor,
    converged: torch.Tensor,
    batch_ndim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return equation defect for an exact direct or pointwise solve."""

    balance = _norm(rhs, batch_ndim) + _norm(applied, batch_ndim)
    relative_residual = _safe_ratio(residual_norm, balance)
    tolerance_ratio = torch.where(
        converged,
        torch.zeros_like(residual_norm),
        torch.full_like(residual_norm, torch.inf),
    )
    return relative_residual, tolerance_ratio, torch.zeros_like(converged)


def _default_batch_ndim(tensor: torch.Tensor) -> int:
    return max(tensor.ndim - 1, 0)


def _can_eager_break(mask: torch.Tensor) -> bool:
    compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
    return not compiling and bool(torch.all(mask).detach().cpu().item())


def _krylov_threshold(
    base: torch.Tensor,
    value: torch.Tensor,
    batch_ndim: int,
    requested_dtype: torch.dtype,
    roundoff_diagonal: torch.Tensor | None,
) -> torch.Tensor:
    """Add a backward-error floor for a reduced-precision solution.

    Rounding a low-precision iterate introduces a residual proportional to the
    operator diagonal and the iterate magnitude.  Without this floor, a valid
    FP16/BF16 solution of a cancellation-heavy operator (notably a discrete
    Poisson operator) can never satisfy an RHS-relative stopping test.
    """

    if roundoff_diagonal is None or requested_dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        return base
    diagonal = roundoff_diagonal.to(dtype=torch.float32)
    roundoff = torch.finfo(requested_dtype).eps * _norm(
        diagonal * value.to(dtype=torch.float32), batch_ndim
    )
    return base + roundoff


def conjugate_gradient(
    operator: TensorOperator,
    rhs: torch.Tensor,
    x0: torch.Tensor | None = None,
    *,
    preconditioner: TensorOperator | None = None,
    max_iterations: int = 200,
    rtol: float = 1e-5,
    atol: float = 0.0,
    batch_ndim: int | None = None,
    projection: TensorOperator | None = None,
    roundoff_diagonal: torch.Tensor | None = None,
) -> KrylovResult:
    """Batch-aware preconditioned CG with per-batch diagnostics."""

    requested_dtype = rhs.dtype
    reduced_precision = requested_dtype in (torch.float16, torch.bfloat16)
    matvec = operator
    if reduced_precision:
        rhs = rhs.to(torch.float32)
        x0 = None if x0 is None else x0.to(torch.float32)

        def fp32_recurrence_matvec(value: torch.Tensor) -> torch.Tensor:
            return matvec(value.to(requested_dtype)).to(requested_dtype).to(torch.float32)

        operator = fp32_recurrence_matvec
    if projection is not None:
        rhs = projection(rhs)
    batch_ndim = _default_batch_ndim(rhs) if batch_ndim is None else batch_ndim
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    if projection is not None:
        x = projection(x)
    residual = rhs - operator(x)
    if projection is not None:
        residual = projection(residual)
    norm_rhs = _norm(rhs, batch_ndim)
    threshold = atol + rtol * norm_rhs
    residual_norm = _norm(residual, batch_ndim)
    effective_threshold = _krylov_threshold(
        threshold, x, batch_ndim, requested_dtype, roundoff_diagonal
    )
    converged = residual_norm <= effective_threshold
    breakdown = ~torch.isfinite(residual_norm)
    iterations = torch.zeros_like(residual_norm, dtype=torch.int32)
    if preconditioner is None:

        def identity_preconditioner(value: torch.Tensor) -> torch.Tensor:
            return value

        apply_preconditioner: TensorOperator = identity_preconditioner
    else:
        apply_preconditioner = preconditioner
    z = apply_preconditioner(residual).to(rhs.dtype)
    if projection is not None:
        z = projection(z)
    direction = z.clone()
    rho = _dot(residual, z, batch_ndim)
    epsilon = torch.finfo(rhs.dtype).tiny * 16

    def body(
        iteration: int | torch.Tensor,
        current_x: torch.Tensor,
        current_residual: torch.Tensor,
        current_direction: torch.Tensor,
        current_rho: torch.Tensor,
        current_converged: torch.Tensor,
        current_breakdown: torch.Tensor,
        current_iterations: torch.Tensor,
    ):
        active = ~(current_converged | current_breakdown)
        applied = operator(current_direction)
        if projection is not None:
            applied = projection(applied)
        denominator = _dot(current_direction, applied, batch_ndim)
        invalid = (~torch.isfinite(denominator)) | (
            torch.abs(denominator) <= epsilon
        )
        next_breakdown = current_breakdown | (active & invalid)
        safe_denominator = torch.where(invalid, 1.0, denominator)
        alpha = torch.where(
            active & ~invalid,
            current_rho / safe_denominator,
            0.0,
        )
        expanded_alpha = _batch_expand(alpha, rhs, batch_ndim)
        next_x = torch.addcmul(current_x, expanded_alpha, current_direction)
        if projection is not None:
            next_x = projection(next_x)
        next_residual = torch.addcmul(
            current_residual, -expanded_alpha, applied
        )
        if projection is not None:
            next_residual = projection(next_residual)
        next_iterations = current_iterations + active.to(torch.int32)
        next_residual_norm = _norm(next_residual, batch_ndim)
        recursive_success = (
            torch.isfinite(next_residual_norm)
            & (next_residual_norm <= threshold)
        )
        true_residual = rhs - operator(next_x)
        if projection is not None:
            true_residual = projection(true_residual)
        true_residual_norm = _norm(true_residual, batch_ndim)
        confirmed = recursive_success & torch.isfinite(true_residual_norm) & (
            true_residual_norm
            <= _krylov_threshold(
                threshold,
                next_x,
                batch_ndim,
                requested_dtype,
                roundoff_diagonal,
            )
        )
        replace_recursive = active & recursive_success & ~confirmed
        if reduced_precision:
            rounded_x = next_x.to(requested_dtype).to(torch.float32)
            next_x = torch.where(
                _batch_expand(replace_recursive, rhs, batch_ndim),
                rounded_x,
                next_x,
            )
        next_residual = torch.where(
            _batch_expand(replace_recursive, rhs, batch_ndim),
            true_residual,
            next_residual,
        )
        next_residual_norm = _norm(next_residual, batch_ndim)
        next_converged = current_converged | confirmed
        next_breakdown = next_breakdown | ~torch.isfinite(next_residual_norm)
        next_z = apply_preconditioner(next_residual).to(rhs.dtype)
        if projection is not None:
            next_z = projection(next_z)
        rho_new = _dot(next_residual, next_z, batch_ndim)
        rho_invalid = (~torch.isfinite(current_rho)) | (
            torch.abs(current_rho) <= epsilon
        )
        beta = torch.where(
            ~(next_converged | next_breakdown | rho_invalid),
            rho_new / torch.where(rho_invalid, 1.0, current_rho),
            0.0,
        )
        next_direction = torch.addcmul(
            next_z,
            _batch_expand(beta, rhs, batch_ndim),
            current_direction,
        )
        return (
            iteration + 1,
            next_x,
            next_residual,
            next_direction,
            rho_new,
            next_converged,
            next_breakdown,
            next_iterations,
        )

    # Dynamo captures this higher-order loop without executing Python line tracers.
    if getattr(torch.compiler, "is_compiling", lambda: False)():  # pragma: no cover
        counter = torch.zeros((), dtype=torch.int32, device=rhs.device)

        def condition(
            iteration: torch.Tensor,
            _x: torch.Tensor,
            _residual: torch.Tensor,
            _direction: torch.Tensor,
            _rho: torch.Tensor,
            current_converged: torch.Tensor,
            current_breakdown: torch.Tensor,
            _iterations: torch.Tensor,
        ) -> torch.Tensor:
            del _x, _residual, _direction, _rho, _iterations
            return (iteration < max_iterations) & torch.any(
                ~(current_converged | current_breakdown)
            )

        (
            _,
            x,
            residual,
            direction,
            rho,
            converged,
            breakdown,
            iterations,
        ) = torch.while_loop(
            condition,
            body,
            (
                counter,
                x,
                residual,
                direction,
                rho,
                converged,
                breakdown,
                iterations,
            ),
        )
    else:
        for iteration in range(max_iterations):
            if _can_eager_break(converged | breakdown):
                break
            (
                _,
                x,
                residual,
                direction,
                rho,
                converged,
                breakdown,
                iterations,
            ) = body(
                iteration,
                x,
                residual,
                direction,
                rho,
                converged,
                breakdown,
                iterations,
            )
    final_applied = operator(x)
    if projection is not None:
        final_applied = projection(final_applied)
    residual_norm = _norm(rhs - final_applied, batch_ndim)
    effective_threshold = _krylov_threshold(
        threshold, x, batch_ndim, requested_dtype, roundoff_diagonal
    )
    converged = torch.isfinite(residual_norm) & (
        residual_norm <= effective_threshold
    )
    solution = x.to(requested_dtype) if reduced_precision else x
    relative_residual, tolerance_ratio, precision_limited = _linear_diagnostics(
        rhs,
        final_applied,
        residual_norm,
        batch_ndim,
        threshold,
        effective_threshold,
    )
    precision_limited = precision_limited & converged & ~breakdown
    return KrylovResult(
        solution,
        converged & ~breakdown,
        residual_norm,
        iterations,
        breakdown,
        relative_residual,
        tolerance_ratio,
        precision_limited,
    )


def projected_conjugate_gradient(
    operator: TensorOperator,
    rhs: torch.Tensor,
    x0: torch.Tensor | None = None,
    *,
    spatial_ndim: int = 1,
    preconditioner: TensorOperator | None = None,
    max_iterations: int = 200,
    rtol: float = 1e-5,
    atol: float = 0.0,
    batch_ndim: int | None = None,
    roundoff_diagonal: torch.Tensor | None = None,
) -> KrylovResult:
    """CG restricted to the per-channel, spatially mean-zero subspace."""

    spatial_dims = tuple(range(rhs.ndim - spatial_ndim, rhs.ndim))

    def project(value: torch.Tensor) -> torch.Tensor:
        return value - value.mean(dim=spatial_dims, keepdim=True)

    return conjugate_gradient(
        operator,
        rhs,
        x0,
        preconditioner=preconditioner,
        max_iterations=max_iterations,
        rtol=rtol,
        atol=atol,
        batch_ndim=batch_ndim,
        projection=project,
        roundoff_diagonal=roundoff_diagonal,
    )


def bicgstab(
    operator: TensorOperator,
    rhs: torch.Tensor,
    x0: torch.Tensor | None = None,
    *,
    preconditioner: TensorOperator | None = None,
    max_iterations: int = 200,
    rtol: float = 1e-5,
    atol: float = 0.0,
    batch_ndim: int | None = None,
    roundoff_diagonal: torch.Tensor | None = None,
) -> KrylovResult:
    """Batch-aware preconditioned BiCGStab for general real operators."""

    requested_dtype = rhs.dtype
    reduced_precision = requested_dtype in (torch.float16, torch.bfloat16)
    matvec = operator
    if reduced_precision:
        rhs = rhs.to(torch.float32)
        x0 = None if x0 is None else x0.to(torch.float32)

        def fp32_recurrence_matvec(value: torch.Tensor) -> torch.Tensor:
            return matvec(value.to(requested_dtype)).to(requested_dtype).to(torch.float32)

        operator = fp32_recurrence_matvec
    batch_ndim = _default_batch_ndim(rhs) if batch_ndim is None else batch_ndim
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    residual = rhs - operator(x)
    shadow = residual.clone()
    norm_rhs = _norm(rhs, batch_ndim)
    threshold = atol + rtol * norm_rhs
    residual_norm = _norm(residual, batch_ndim)
    effective_threshold = _krylov_threshold(
        threshold, x, batch_ndim, requested_dtype, roundoff_diagonal
    )
    converged = residual_norm <= effective_threshold
    breakdown = ~torch.isfinite(residual_norm)
    iterations = torch.zeros_like(residual_norm, dtype=torch.int32)
    rho_previous = torch.ones_like(residual_norm)
    alpha = torch.ones_like(rho_previous)
    omega = torch.ones_like(rho_previous)
    direction = torch.zeros_like(rhs)
    applied_direction = torch.zeros_like(rhs)
    if preconditioner is None:

        def identity_preconditioner(value: torch.Tensor) -> torch.Tensor:
            return value

        apply_preconditioner: TensorOperator = identity_preconditioner
    else:
        apply_preconditioner = preconditioner
    epsilon = torch.finfo(rhs.dtype).tiny * 16

    def body(
        iteration: int | torch.Tensor,
        current_x: torch.Tensor,
        current_residual: torch.Tensor,
        current_rho_previous: torch.Tensor,
        current_alpha: torch.Tensor,
        current_omega: torch.Tensor,
        current_direction: torch.Tensor,
        current_applied_direction: torch.Tensor,
        current_converged: torch.Tensor,
        current_breakdown: torch.Tensor,
        current_iterations: torch.Tensor,
    ):
        active = ~(current_converged | current_breakdown)
        rho = _dot(shadow, current_residual, batch_ndim)
        rho_bad = (~torch.isfinite(rho)) | (torch.abs(rho) <= epsilon)
        omega_bad = (~torch.isfinite(current_omega)) | (
            torch.abs(current_omega) <= epsilon
        )
        next_breakdown = current_breakdown | (active & (rho_bad | omega_bad))
        safe_rho_previous = torch.where(
            torch.abs(current_rho_previous) <= epsilon,
            1.0,
            current_rho_previous,
        )
        safe_omega = torch.where(omega_bad, 1.0, current_omega)
        beta = torch.where(
            active & ~(rho_bad | omega_bad),
            (rho / safe_rho_previous) * (current_alpha / safe_omega),
            0.0,
        )
        direction_delta = torch.addcmul(
            current_direction,
            -_batch_expand(current_omega, rhs, batch_ndim),
            current_applied_direction,
        )
        next_direction = torch.addcmul(
            current_residual,
            _batch_expand(beta, rhs, batch_ndim),
            direction_delta,
        )
        preconditioned_direction = apply_preconditioner(next_direction).to(rhs.dtype)
        next_applied_direction = operator(preconditioned_direction)
        denominator = _dot(shadow, next_applied_direction, batch_ndim)
        denominator_bad = (~torch.isfinite(denominator)) | (
            torch.abs(denominator) <= epsilon
        )
        next_breakdown = next_breakdown | (active & denominator_bad)
        safe_denominator = torch.where(denominator_bad, 1.0, denominator)
        next_alpha = torch.where(
            active & ~denominator_bad, rho / safe_denominator, 0.0
        )
        expanded_alpha = _batch_expand(next_alpha, rhs, batch_ndim)
        x_alpha = torch.addcmul(
            current_x, expanded_alpha, preconditioned_direction
        )
        intermediate = torch.addcmul(
            current_residual, -expanded_alpha, next_applied_direction
        )
        intermediate_norm = _norm(intermediate, batch_ndim)
        early = (
            active
            & ~next_breakdown
            & torch.isfinite(intermediate_norm)
            & (intermediate_norm <= threshold)
        )
        true_alpha_residual = rhs - operator(x_alpha)
        true_alpha_norm = _norm(true_alpha_residual, batch_ndim)
        confirmed_early = early & torch.isfinite(true_alpha_norm) & (
            true_alpha_norm
            <= _krylov_threshold(
                threshold,
                x_alpha,
                batch_ndim,
                requested_dtype,
                roundoff_diagonal,
            )
        )
        intermediate = torch.where(
            _batch_expand(early & ~confirmed_early, rhs, batch_ndim),
            true_alpha_residual,
            intermediate,
        )
        early = confirmed_early
        remaining = active & ~next_breakdown & ~early
        preconditioned_intermediate = apply_preconditioner(intermediate).to(rhs.dtype)
        applied_intermediate = operator(preconditioned_intermediate)
        omega_denominator = _dot(applied_intermediate, applied_intermediate, batch_ndim)
        omega_bad = (~torch.isfinite(omega_denominator)) | (
            torch.abs(omega_denominator) <= epsilon
        )
        omega_numerator = _dot(applied_intermediate, intermediate, batch_ndim)
        omega_new = omega_numerator / torch.where(
            omega_bad, 1.0, omega_denominator
        )
        omega_value_bad = (~torch.isfinite(omega_new)) | (
            torch.abs(omega_new) <= epsilon
        )
        next_breakdown = next_breakdown | (
            remaining & (omega_bad | omega_value_bad)
        )
        next_omega = torch.where(
            remaining & ~(omega_bad | omega_value_bad), omega_new, 1.0
        )
        expanded_omega = _batch_expand(next_omega, rhs, batch_ndim)
        x_full = torch.addcmul(
            x_alpha, expanded_omega, preconditioned_intermediate
        )
        next_residual_full = torch.addcmul(
            intermediate, -expanded_omega, applied_intermediate
        )
        use_alpha = _batch_expand(early, rhs, batch_ndim)
        use_full = _batch_expand(remaining & ~next_breakdown, rhs, batch_ndim)
        next_x = torch.where(
            use_alpha, x_alpha, torch.where(use_full, x_full, current_x)
        )
        next_residual = torch.where(
            use_alpha,
            intermediate,
            torch.where(use_full, next_residual_full, current_residual),
        )
        next_iterations = current_iterations + active.to(torch.int32)
        next_residual_norm = _norm(next_residual, batch_ndim)
        recursive_success = early | (
            torch.isfinite(next_residual_norm)
            & (next_residual_norm <= threshold)
        )
        true_residual = rhs - operator(next_x)
        true_residual_norm = _norm(true_residual, batch_ndim)
        confirmed = recursive_success & torch.isfinite(true_residual_norm) & (
            true_residual_norm
            <= _krylov_threshold(
                threshold,
                next_x,
                batch_ndim,
                requested_dtype,
                roundoff_diagonal,
            )
        )
        replace_recursive = active & recursive_success & ~confirmed
        if reduced_precision:
            rounded_x = next_x.to(requested_dtype).to(torch.float32)
            next_x = torch.where(
                _batch_expand(replace_recursive, rhs, batch_ndim),
                rounded_x,
                next_x,
            )
        next_residual = torch.where(
            _batch_expand(replace_recursive, rhs, batch_ndim),
            true_residual,
            next_residual,
        )
        next_residual_norm = _norm(next_residual, batch_ndim)
        next_converged = current_converged | confirmed
        next_breakdown = next_breakdown | ~torch.isfinite(next_residual_norm)
        return (
            iteration + 1,
            next_x,
            next_residual,
            rho,
            next_alpha,
            next_omega,
            next_direction,
            next_applied_direction,
            next_converged,
            next_breakdown,
            next_iterations,
        )

    # Dynamo captures this higher-order loop without executing Python line tracers.
    if getattr(torch.compiler, "is_compiling", lambda: False)():  # pragma: no cover
        counter = torch.zeros((), dtype=torch.int32, device=rhs.device)

        def condition(
            iteration: torch.Tensor,
            _x: torch.Tensor,
            _residual: torch.Tensor,
            _rho_previous: torch.Tensor,
            _alpha: torch.Tensor,
            _omega: torch.Tensor,
            _direction: torch.Tensor,
            _applied_direction: torch.Tensor,
            current_converged: torch.Tensor,
            current_breakdown: torch.Tensor,
            _iterations: torch.Tensor,
        ) -> torch.Tensor:
            del (
                _x,
                _residual,
                _rho_previous,
                _alpha,
                _omega,
                _direction,
                _applied_direction,
                _iterations,
            )
            return (iteration < max_iterations) & torch.any(
                ~(current_converged | current_breakdown)
            )

        (
            _,
            x,
            residual,
            rho_previous,
            alpha,
            omega,
            direction,
            applied_direction,
            converged,
            breakdown,
            iterations,
        ) = torch.while_loop(
            condition,
            body,
            (
                counter,
                x,
                residual,
                rho_previous,
                alpha,
                omega,
                direction,
                applied_direction,
                converged,
                breakdown,
                iterations,
            ),
        )
    else:
        for iteration in range(max_iterations):
            if _can_eager_break(converged | breakdown):
                break
            (
                _,
                x,
                residual,
                rho_previous,
                alpha,
                omega,
                direction,
                applied_direction,
                converged,
                breakdown,
                iterations,
            ) = body(
                iteration,
                x,
                residual,
                rho_previous,
                alpha,
                omega,
                direction,
                applied_direction,
                converged,
                breakdown,
                iterations,
            )
    final_applied = operator(x)
    residual_norm = _norm(rhs - final_applied, batch_ndim)
    effective_threshold = _krylov_threshold(
        threshold, x, batch_ndim, requested_dtype, roundoff_diagonal
    )
    converged = torch.isfinite(residual_norm) & (
        residual_norm <= effective_threshold
    )
    solution = x.to(requested_dtype) if reduced_precision else x
    relative_residual, tolerance_ratio, precision_limited = _linear_diagnostics(
        rhs,
        final_applied,
        residual_norm,
        batch_ndim,
        threshold,
        effective_threshold,
    )
    precision_limited = precision_limited & converged & ~breakdown
    return KrylovResult(
        solution,
        converged & ~breakdown,
        residual_norm,
        iterations,
        breakdown,
        relative_residual,
        tolerance_ratio,
        precision_limited,
    )


@dataclass(frozen=True)
class _EvaluationContext:
    old_state: torch.Tensor
    new_state: torch.Tensor
    inputs: Mapping[str, torch.Tensor]
    time: torch.Tensor
    override_targets: tuple[int, ...] = ()
    override_values: torch.Tensor | None = None
    leading_batch_shape: tuple[int, ...] = ()


def _field(tensor: torch.Tensor, index: int, spatial_ndim: int) -> torch.Tensor:
    return tensor.select(tensor.ndim - spatial_ndim - 1, index)


def _stack_fields(values: Sequence[torch.Tensor], spatial_ndim: int) -> torch.Tensor:
    channel_dimension = values[0].ndim - spatial_ndim
    if len(values) == 1:
        return values[0].unsqueeze(channel_dimension)
    return torch.stack(tuple(values), dim=channel_dimension)


def _select_fields(
    state: torch.Tensor,
    targets: Sequence[int],
    spatial_ndim: int,
) -> torch.Tensor:
    return _stack_fields(
        tuple(_field(state, target, spatial_ndim) for target in targets),
        spatial_ndim,
    )


def _commit_field_updates(
    state: torch.Tensor,
    updates: Sequence[tuple[Sequence[int], torch.Tensor]],
    spatial_ndim: int,
) -> torch.Tensor:
    replacements = {
        target: values.select(values.ndim - spatial_ndim - 1, local)
        for targets, values in updates
        for local, target in enumerate(targets)
    }
    channel_dimension = state.ndim - spatial_ndim - 1
    return _stack_fields(
        tuple(
            replacements.get(index, _field(state, index, spatial_ndim))
            for index in range(state.shape[channel_dimension])
        ),
        spatial_ndim,
    )


def _tensor_or_scalar(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _broadcast_batch_scalar(
    value: torch.Tensor,
    state: torch.Tensor,
    spatial_ndim: int,
) -> torch.Tensor:
    batch_ndim = state.ndim - spatial_ndim - 1
    batch_shape = tuple(state.shape[:batch_ndim])
    if value.ndim == batch_ndim and tuple(value.shape) == batch_shape:
        return value.reshape((*batch_shape, *((1,) * spatial_ndim)))
    return value


def _make_jacobi_preconditioner(
    diagonal: torch.Tensor,
    epsilon: float,
) -> TensorOperator:
    safe = torch.where(torch.abs(diagonal) > epsilon, diagonal, 1.0)

    def jacobi(value: torch.Tensor) -> torch.Tensor:
        return value / safe

    return jacobi


class CompiledSolver(torch.nn.Module):
    """A reusable matrix-free PyTorch solver compiled from :class:`ProgramIR`."""

    _time_step_value: torch.Tensor
    _newton_alphas: torch.Tensor

    def __init__(self, program: ProgramIR | Mapping[str, Any]) -> None:
        super().__init__()
        self._program = program_from_dict(program) if isinstance(program, Mapping) else program
        self._ndim = len(self._program.grid_shape)
        self._dt = _exact_float(self._program.dt)
        self._options = dict(self._program.options)
        self._residual_scales = {
            block.id: _exact_float(block.residual_scale)
            for layer in self._program.schedule
            for block in layer.blocks
        }
        spacings: list[float] = []
        for axis, ((lower_text, upper_text), cells) in enumerate(
            zip(self._program.bounds, self._program.grid_shape, strict=True)
        ):
            lower = _exact_float(lower_text)
            upper = _exact_float(upper_text)
            spacing = (upper - lower) / cells
            spacings.append(spacing)
            coordinate = lower + (torch.arange(cells, dtype=torch.float64) + 0.5) * spacing
            shape = [1] * self._ndim
            shape[axis] = cells
            self.register_buffer(f"coordinate_{axis}", coordinate.reshape(shape), persistent=True)
        self._spacing = tuple(spacings)
        self._literal_values: dict[Literal, float] = {}
        self._constant_values: dict[ExpressionIR, float] = {}
        stencil_entries: dict[
            tuple[tuple[tuple[int, ...], str], ...], None
        ] = {}
        expressions: list[ExpressionIR] = []
        for equation in self._program.equations:
            expressions.append(equation.residual)
            self._collect_literals(equation.residual)
            self._collect_stencil_entries(equation.residual, stencil_entries)
            if equation.diagonal is not None:
                expressions.append(equation.diagonal)
                self._collect_literals(equation.diagonal)
                self._collect_stencil_entries(equation.diagonal, stencil_entries)
            if equation.direct_solution is not None:
                expressions.append(equation.direct_solution)
                self._collect_literals(equation.direct_solution)
                self._collect_stencil_entries(
                    equation.direct_solution, stencil_entries
                )
        for layer in self._program.schedule:
            for block in layer.blocks:
                if block.pointwise_coefficients is None:
                    continue
                for row in block.pointwise_coefficients:
                    for expression in row:
                        expressions.append(expression)
                        self._collect_literals(expression)
                        self._collect_stencil_entries(
                            expression, stencil_entries
                        )
        for expression in expressions:
            self._fold_constants(expression)
        self._pointwise_strategies: dict[int, _PointwiseStrategy] = {}
        for layer in self._program.schedule:
            for block in layer.blocks:
                if block.kind != BlockKind.POINTWISE_LINEAR:
                    continue
                coefficients = block.pointwise_coefficients
                if coefficients is None:
                    kind = (
                        "legacy_small"
                        if len(block.targets) <= 4
                        else "legacy_fallback"
                    )
                    self._pointwise_strategies[block.id] = _PointwiseStrategy(
                        kind
                    )
                    continue
                constant_matrix = tuple(
                    tuple(
                        self._constant_values.get(expression)
                        for expression in row
                    )
                    for row in coefficients
                )
                if all(
                    value is not None
                    for row in constant_matrix
                    for value in row
                ):
                    self._pointwise_strategies[block.id] = (
                        _factor_constant_pointwise_matrix(
                            tuple(
                                tuple(
                                    cast(float, value) for value in row
                                )
                                for row in constant_matrix
                            )
                        )
                    )
                else:
                    kind = (
                        "variable_small"
                        if len(block.targets) <= 4
                        else "variable_fallback"
                    )
                    self._pointwise_strategies[block.id] = _PointwiseStrategy(
                        kind
                    )
        self._constant_buffer_names: dict[ExpressionIR, str] = {}
        for index, (expression, value) in enumerate(self._constant_values.items()):
            name = f"_constant_value_{index}"
            self.register_buffer(
                name,
                torch.tensor(value, dtype=torch.float64),
                persistent=False,
            )
            self._constant_buffer_names[expression] = name
        self._boundary_value_names: dict[BoundaryIR, str] = {}
        all_boundaries = (
            *self._program.boundaries,
            *self._program.coefficient_boundaries,
        )
        for index, boundary in enumerate(dict.fromkeys(all_boundaries)):
            name = f"_boundary_value_{index}"
            self.register_buffer(
                name,
                torch.tensor(boundary.value, dtype=torch.float64),
                persistent=False,
            )
            self._boundary_value_names[boundary] = name
        self.register_buffer(
            "_time_step_value",
            torch.tensor(self._dt, dtype=torch.float64),
            persistent=False,
        )

        region_requests: dict[
            tuple[tuple[int, ...], tuple[int, ...]], None
        ] = {}
        zero_halo = (0,) * self._ndim
        for expression in expressions:
            self._collect_region_requests(expression, zero_halo, region_requests)
        self._region_index_names: dict[
            tuple[tuple[int, ...], tuple[int, ...]], tuple[str, ...]
        ] = {}
        self._periodic_region_index_names: dict[
            tuple[tuple[int, ...], tuple[int, ...]], tuple[str, ...]
        ] = {}
        self._region_reflection_names: dict[
            tuple[tuple[int, ...], tuple[int, ...]], str
        ] = {}
        for index, request in enumerate(region_requests):
            self._register_region_maps(index, *request)

        self._coefficient_boundaries = dict(
            zip(
                self._program.coefficient_fields,
                self._program.coefficient_boundaries,
                strict=True,
            )
        )

        stencil_groups: dict[
            tuple[tuple[tuple[tuple[int, ...], str], ...], ...], None
        ] = {(entries,): None for entries in stencil_entries}
        for layer in self._program.schedule:
            for block in layer.blocks:
                residuals = tuple(
                    self._program.equations[equation].residual
                    for equation in block.equations
                )
                self._collect_stencil_group_signatures(
                    residuals, block.targets, stencil_groups
                )
                diagonals = tuple(
                    self._program.equations[equation].diagonal
                    for equation in block.equations
                )
                if all(expression is not None for expression in diagonals):
                    self._collect_stencil_group_signatures(
                        tuple(
                            expression
                            for expression in diagonals
                            if expression is not None
                        ),
                        (),
                        stencil_groups,
                    )
                direct_solutions = tuple(
                    self._program.equations[equation].direct_solution
                    for equation in block.equations
                    if self._program.equations[equation].direct_solution
                    is not None
                )
                if direct_solutions:
                    self._collect_stencil_group_signatures(
                        direct_solutions,
                        (),
                        stencil_groups,
                    )
                if block.pointwise_coefficients is not None:
                    self._collect_stencil_group_signatures(
                        tuple(
                            expression
                            for row in block.pointwise_coefficients
                            for expression in row
                        ),
                        (),
                        stencil_groups,
                    )
        self._stencil_group_kernel_names: dict[
            tuple[tuple[tuple[tuple[int, ...], str], ...], ...], str
        ] = {}
        for index, signature in enumerate(stencil_groups):
            name = f"_stencil_group_kernel_{index}"
            kernels = tuple(_dense_stencil_kernel(entries) for entries in signature)
            packed = (
                kernels[0].unsqueeze(0)
                if len(kernels) == 1
                else torch.stack(kernels, dim=0)
            ).unsqueeze(1)
            self.register_buffer(
                name,
                packed,
                persistent=False,
            )
            self._stencil_group_kernel_names[signature] = name
        alphas = tuple(
            float(self._options["newton_backtrack_factor"]) ** index
            for index in range(int(self._options["newton_max_backtracks"]))
        )
        self.register_buffer(
            "_newton_alphas",
            torch.tensor(alphas, dtype=torch.float64),
            persistent=False,
        )
        schedule_names = tuple(
            tuple(tuple(self._program.fields[target] for target in block.targets) for block in layer.blocks)
            for layer in self._program.schedule
        )
        self.metadata = SolverMetadata(
            version=self._program.version,
            name=self._program.name,
            field_order=self._program.fields,
            axis_order=self._program.axes,
            grid_shape=self._program.grid_shape,
            spacing=self._spacing,
            parameter_order=self._program.parameters,
            coefficient_field_order=self._program.coefficient_fields,
            schedule=schedule_names,
        )

    @property
    def program(self) -> ProgramIR:
        return self._program

    def _collect_literals(self, expression: ExpressionIR) -> None:
        if isinstance(expression, Literal):
            if expression not in self._literal_values:
                if expression.value == "pi":
                    value = torch.pi
                elif expression.value == "E":
                    value = 2.718281828459045
                else:
                    value = _exact_float(expression.value)
                self._literal_values[expression] = float(value)
            return
        for child in _expression_children(expression):
            self._collect_literals(child)

    def _fold_constants(self, expression: ExpressionIR) -> float | None:
        cached = self._constant_values.get(expression)
        if cached is not None:
            return cached
        if isinstance(expression, Literal):
            result: float | None = self._literal_values[expression]
        elif isinstance(expression, NaryOp):
            values = tuple(self._fold_constants(argument) for argument in expression.arguments)
            if any(value is None for value in values):
                result = None
            elif expression.op == "add":
                result = sum(value for value in values if value is not None)
            elif expression.op == "mul":
                result = math.prod(value for value in values if value is not None)
            else:
                result = None
        elif isinstance(expression, Power):
            base = self._fold_constants(expression.base)
            exponent = self._fold_constants(expression.exponent)
            result = None if base is None or exponent is None else base**exponent
        elif isinstance(expression, Call):
            values = tuple(self._fold_constants(argument) for argument in expression.arguments)
            if any(value is None for value in values):
                result = None
            else:
                arguments = tuple(value for value in values if value is not None)
                functions: dict[str, Callable[..., float]] = {
                    "minimum": min,
                    "maximum": max,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "asin": math.asin,
                    "acos": math.acos,
                    "atan": math.atan,
                    "sinh": math.sinh,
                    "cosh": math.cosh,
                    "tanh": math.tanh,
                    "asinh": math.asinh,
                    "acosh": math.acosh,
                    "atanh": math.atanh,
                    "exp": math.exp,
                    "log": math.log,
                    "abs": abs,
                    "sign": lambda value: float((value > 0) - (value < 0)),
                    "floor": math.floor,
                    "ceil": math.ceil,
                }
                function = functions.get(expression.function)
                result = None if function is None else float(function(*arguments))
        else:
            result = None
        if result is not None:
            self._constant_values[expression] = float(result)
        return result

    def _collect_stencil_entries(
        self,
        expression: ExpressionIR,
        result: dict[tuple[tuple[tuple[int, ...], str], ...], None],
    ) -> None:
        if isinstance(expression, StencilApply):
            result.setdefault(expression.entries, None)
        for child in _expression_children(expression):
            self._collect_stencil_entries(child, result)

    def _collect_region_requests(
        self,
        expression: ExpressionIR,
        halo: tuple[int, ...],
        result: dict[tuple[tuple[int, ...], tuple[int, ...]], None],
    ) -> None:
        if isinstance(expression, StencilApply):
            radius = _stencil_radius(expression.entries)
            input_halo = tuple(
                requested + item
                for requested, item in zip(halo, radius, strict=True)
            )
            self._collect_region_requests(expression.expression, input_halo, result)
        elif isinstance(expression, (GridRef, CoefficientRef)):
            result.setdefault((expression.offset, halo), None)
        else:
            for child in _expression_children(expression):
                self._collect_region_requests(child, halo, result)

    def _register_region_maps(
        self,
        index: int,
        offset: tuple[int, ...],
        halo: tuple[int, ...],
    ) -> None:
        reflected_names: list[str] = []
        periodic_names: list[str] = []
        reflected_axes: list[torch.Tensor] = []
        for axis, (cells, shift, requested) in enumerate(
            zip(self._program.grid_shape, offset, halo, strict=True)
        ):
            logical = torch.arange(
                -requested + shift,
                cells + requested + shift,
                dtype=torch.int32,
            )
            try:
                gathered, reflected = _boundary_axis_map(
                    logical, cells, "neumann"
                )
            except ValueError as error:
                raise ValueError(
                    f"Requested reflected region {(offset, halo)} exceeds axis {axis}"
                ) from error
            periodic, _ = _boundary_axis_map(logical, cells, "periodic")
            name = f"_region_index_{index}_{axis}"
            self.register_buffer(name, gathered, persistent=False)
            reflected_names.append(name)
            periodic_name = f"_periodic_region_index_{index}_{axis}"
            self.register_buffer(periodic_name, periodic, persistent=False)
            periodic_names.append(periodic_name)
            reflected_axes.append(reflected)
        spatial_shape = tuple(
            cells + 2 * requested
            for cells, requested in zip(self._program.grid_shape, halo, strict=True)
        )
        parity = torch.zeros(spatial_shape, dtype=torch.bool)
        for axis, reflected in enumerate(reflected_axes):
            shape = [1] * self._ndim
            shape[axis] = reflected.numel()
            parity = torch.logical_xor(parity, reflected.reshape(shape))
        parity_name = f"_region_reflection_{index}"
        self.register_buffer(parity_name, parity, persistent=False)
        key = (offset, halo)
        self._region_index_names[key] = tuple(reflected_names)
        self._periodic_region_index_names[key] = tuple(periodic_names)
        self._region_reflection_names[key] = parity_name

    def _collect_stencil_group_signatures(
        self,
        expressions: Sequence[ExpressionIR],
        mutable_fields: Sequence[int],
        result: dict[
            tuple[tuple[tuple[tuple[int, ...], str], ...], ...], None
        ],
    ) -> None:
        stages: dict[int, list[tuple[StencilApply, tuple[int, ...]]]] = {}
        zero_halo = (0,) * self._ndim
        for expression in expressions:
            self._collect_stencil_demands(expression, zero_halo, stages)
        fields = frozenset(mutable_fields)
        dependency_cache: dict[ExpressionIR, bool] = {}
        for depth in sorted(stages):
            groups: dict[
                tuple[tuple[int, ...], tuple[int, ...], bool],
                list[StencilApply],
            ] = {}
            for expression, halo in dict.fromkeys(stages[depth]):
                dynamic = bool(fields) and _depends_on_new_fields(
                    expression, fields, dependency_cache
                )
                groups.setdefault(
                    (_stencil_radius(expression.entries), halo, dynamic), []
                ).append(expression)
            for group in groups.values():
                signature = tuple(expression.entries for expression in group)
                result.setdefault(signature, None)

    def _evaluate(
        self,
        expression: ExpressionIR,
        context: _EvaluationContext,
        session: _EvaluationSession | None = None,
    ) -> EvaluationValue:
        return self._evaluate_many((expression,), context, session)[0]

    def _collect_stencil_demands(
        self,
        expression: ExpressionIR,
        halo: tuple[int, ...],
        stages: dict[int, list[tuple[StencilApply, tuple[int, ...]]]],
    ) -> int:
        if isinstance(expression, StencilApply):
            radius = _stencil_radius(expression.entries)
            input_halo = tuple(
                left + right
                for left, right in zip(halo, radius, strict=True)
            )
            child_depth = self._collect_stencil_demands(
                expression.expression,
                input_halo,
                stages,
            )
            depth = child_depth + 1
            stages.setdefault(depth, []).append((expression, halo))
            return depth
        children = _expression_children(expression)
        if not children:
            return -1
        depths = [
            self._collect_stencil_demands(child, halo, stages)
            for child in children
        ]
        return max([-1, *depths])

    def _prepare_stencils(
        self,
        expressions: Sequence[ExpressionIR],
        context: _EvaluationContext,
        session: _EvaluationSession | None,
        local_cache: dict[
            tuple[ExpressionIR, tuple[int, ...]], EvaluationValue
        ],
    ) -> None:
        stages: dict[
            int, list[tuple[StencilApply, tuple[int, ...]]]
        ] = {}
        zero_halo = (0,) * self._ndim
        for expression in expressions:
            self._collect_stencil_demands(expression, zero_halo, stages)
        for depth in sorted(stages):
            groups: dict[
                tuple[tuple[int, ...], tuple[int, ...], bool],
                list[StencilApply],
            ] = {}
            for expression, halo in dict.fromkeys(stages[depth]):
                cache = self._cache_for(expression, session, local_cache)
                if (expression, halo) in cache:
                    continue
                dynamic = session is not None and _depends_on_new_fields(
                    expression,
                    session.mutable_new_fields,
                    session.dependency_cache,
                )
                groups.setdefault(
                    (_stencil_radius(expression.entries), halo, dynamic),
                    [],
                ).append(expression)
            for (_radius, halo, _dynamic), group in groups.items():
                self._evaluate_stencil_group(
                    group,
                    context,
                    halo,
                    session,
                    local_cache,
                )

    def _evaluate_many(
        self,
        expressions: Sequence[ExpressionIR],
        context: _EvaluationContext,
        session: _EvaluationSession | None = None,
    ) -> list[EvaluationValue]:
        local_cache: dict[
            tuple[ExpressionIR, tuple[int, ...]], EvaluationValue
        ] = {}
        self._prepare_stencils(
            expressions,
            context,
            session,
            local_cache,
        )
        zero_halo = (0,) * self._ndim
        return [
            self._evaluate_region(
                expression,
                context,
                zero_halo,
                session,
                local_cache,
            )
            for expression in expressions
        ]

    def _cache_for(
        self,
        expression: ExpressionIR,
        session: _EvaluationSession | None,
        local_cache: dict[
            tuple[ExpressionIR, tuple[int, ...]], EvaluationValue
        ],
    ) -> dict[tuple[ExpressionIR, tuple[int, ...]], EvaluationValue]:
        if session is None:
            return local_cache
        if _depends_on_new_fields(
            expression,
            session.mutable_new_fields,
            session.dependency_cache,
        ):
            return local_cache
        return session.invariant_cache

    def _shifted_region(
        self,
        tensor: torch.Tensor,
        offset: tuple[int, ...],
        halo: tuple[int, ...],
        boundary: BoundaryIR | str,
        value: float = 0.0,
    ) -> torch.Tensor:
        key = (offset, halo)
        kind = boundary.kind if isinstance(boundary, BoundaryIR) else boundary
        names = (
            self._periodic_region_index_names.get(key)
            if kind == "periodic"
            else self._region_index_names.get(key)
        )
        if names is None:
            raise RuntimeError(f"Missing precomputed boundary region {key}")
        result = tensor
        for axis, name in enumerate(names):
            dimension = result.ndim - self._ndim + axis
            indices = getattr(self, name).to(device=result.device)
            result = torch.index_select(result, dimension, indices)
        boundary_value = boundary.value if isinstance(boundary, BoundaryIR) else value
        if kind == "dirichlet":
            if (
                isinstance(boundary, BoundaryIR)
                and getattr(torch.compiler, "is_compiling", lambda: False)()
            ):
                boundary_value = getattr(
                    self, self._boundary_value_names[boundary]
                ).to(dtype=result.dtype, device=result.device)
            reflected = getattr(
                self, self._region_reflection_names[key]
            ).to(device=result.device)
            result = torch.addcmul(
                result,
                reflected,
                2.0 * (boundary_value - result),
            )
        elif kind not in {"neumann", "periodic"}:
            raise ValueError(f"Unknown boundary kind {kind!r}")
        return result

    def _materialize_region(
        self,
        value: torch.Tensor | float,
        context: _EvaluationContext,
        halo: tuple[int, ...],
    ) -> torch.Tensor:
        channel_dimension = context.old_state.ndim - self._ndim - 1
        batch_shape = (
            *context.leading_batch_shape,
            *context.old_state.shape[:channel_dimension],
        )
        spatial_shape = tuple(
            cells + 2 * radius
            for cells, radius in zip(self._program.grid_shape, halo, strict=True)
        )
        shape = (*batch_shape, *spatial_shape)
        if isinstance(value, torch.Tensor):
            return torch.broadcast_to(value, shape)
        return torch.full(
            shape,
            value,
            dtype=context.old_state.dtype,
            device=context.old_state.device,
        )

    def _evaluate_stencil_group(
        self,
        expressions: Sequence[StencilApply],
        context: _EvaluationContext,
        halo: tuple[int, ...],
        session: _EvaluationSession | None,
        local_cache: dict[
            tuple[ExpressionIR, tuple[int, ...]], EvaluationValue
        ],
    ) -> None:
        unique = tuple(dict.fromkeys(expressions))
        if not unique:
            return
        radii = _stencil_radius(unique[0].entries)
        if any(_stencil_radius(expression.entries) != radii for expression in unique):
            raise RuntimeError("Internal stencil group has inconsistent radii")
        input_halo = tuple(
            requested + radius
            for requested, radius in zip(halo, radii, strict=True)
        )
        operands = [
            self._materialize_region(
                self._evaluate_region(
                    expression.expression,
                    context,
                    input_halo,
                    session,
                    local_cache,
                ),
                context,
                input_halo,
            )
            for expression in unique
        ]
        channel_dimension = context.old_state.ndim - self._ndim - 1
        batch_shape = (
            *context.leading_batch_shape,
            *context.old_state.shape[:channel_dimension],
        )
        batch_count = math.prod(batch_shape) if batch_shape else 1
        output_shape = tuple(
            cells + 2 * requested
            for cells, requested in zip(
                self._program.grid_shape, halo, strict=True
            )
        )
        if batch_count == 0:
            convolved = torch.zeros(
                (*batch_shape, len(unique), *output_shape),
                dtype=context.old_state.dtype,
                device=context.old_state.device,
            )
        else:
            padded_shape = tuple(
                cells + 2 * requested
                for cells, requested in zip(
                    self._program.grid_shape, input_halo, strict=True
                )
            )
            channels = (
                operands[0].unsqueeze(len(batch_shape))
                if len(operands) == 1
                else torch.stack(operands, dim=len(batch_shape))
            )
            convolution_input = channels.reshape(
                batch_count, len(unique), *padded_shape
            )
            signature = tuple(expression.entries for expression in unique)
            kernel_name = self._stencil_group_kernel_names.get(signature)
            if kernel_name is None:
                raise RuntimeError("Missing precomputed grouped stencil weights")
            kernels = getattr(self, kernel_name).to(
                dtype=context.old_state.dtype,
                device=context.old_state.device,
            )
            convolution_functions = (
                torch_functional.conv1d,
                torch_functional.conv2d,
                torch_functional.conv3d,
            )
            flat = convolution_functions[self._ndim - 1](
                convolution_input,
                kernels,
                groups=len(unique),
            )
            convolved = flat.reshape(
                (*batch_shape, len(unique), *output_shape)
            )
        for index, expression in enumerate(unique):
            cache = self._cache_for(expression, session, local_cache)
            cache[(expression, halo)] = convolved.select(
                len(batch_shape), index
            )

    def _evaluate_arguments(
        self,
        arguments: Sequence[ExpressionIR],
        context: _EvaluationContext,
        halo: tuple[int, ...],
        session: _EvaluationSession | None,
        local_cache: dict[
            tuple[ExpressionIR, tuple[int, ...]], EvaluationValue
        ],
    ) -> list[EvaluationValue]:
        groups: dict[
            tuple[tuple[int, ...], bool], list[StencilApply]
        ] = {}
        for argument in arguments:
            if not isinstance(argument, StencilApply):
                continue
            cache = self._cache_for(argument, session, local_cache)
            if (argument, halo) in cache:
                continue
            dynamic = session is not None and _depends_on_new_fields(
                argument,
                session.mutable_new_fields,
                session.dependency_cache,
            )
            groups.setdefault(
                (_stencil_radius(argument.entries), dynamic), []
            ).append(argument)
        for expressions in groups.values():
            self._evaluate_stencil_group(
                expressions,
                context,
                halo,
                session,
                local_cache,
            )
        return [
            self._evaluate_region(
                argument,
                context,
                halo,
                session,
                local_cache,
            )
            for argument in arguments
        ]

    def _evaluate_region(
        self,
        expression: ExpressionIR,
        context: _EvaluationContext,
        halo: tuple[int, ...],
        session: _EvaluationSession | None,
        local_cache: dict[
            tuple[ExpressionIR, tuple[int, ...]], EvaluationValue
        ],
    ) -> EvaluationValue:
        cache = self._cache_for(expression, session, local_cache)
        key = (expression, halo)
        cached = cache.get(key)
        if cached is not None:
            return cached
        reference = context.old_state
        result: EvaluationValue
        constant = self._constant_values.get(expression)
        if constant is not None:
            if getattr(torch.compiler, "is_compiling", lambda: False)():
                result = getattr(
                    self, self._constant_buffer_names[expression]
                ).to(dtype=reference.dtype, device=reference.device)
            else:
                result = constant
        elif isinstance(expression, Literal):
            # Stencil optimization can rebuild a spatial operand after the
            # top-level constant-folding pass.  Its Literal children are equal
            # to collected literals but are not necessarily present in the
            # folded-expression cache.  Evaluate the already-collected scalar
            # directly so composite operands such as D(u**m, (x, 2)) work in a
            # shifted/halo region too.
            result = self._literal_values[expression]
        elif isinstance(expression, GridRef):
            if expression.level == TimeLevel.OLD:
                value = _field(context.old_state, expression.field, self._ndim)
            elif (
                context.override_values is not None
                and expression.field in context.override_targets
            ):
                local = context.override_targets.index(expression.field)
                value = _field(context.override_values, local, self._ndim)
            else:
                value = _field(context.new_state, expression.field, self._ndim)
            result = self._shifted_region(
                value,
                expression.offset,
                halo,
                self._program.boundaries[expression.field],
            )
        elif isinstance(expression, ScalarRef):
            result = _broadcast_batch_scalar(
                context.inputs[expression.name], reference, self._ndim
            )
        elif isinstance(expression, CoefficientRef):
            value = context.inputs[expression.name]
            if value.ndim >= self._ndim and tuple(value.shape[-self._ndim :]) == self._program.grid_shape:
                result = self._shifted_region(
                    value,
                    expression.offset,
                    halo,
                    self._coefficient_boundaries[expression.name],
                )
            else:
                result = value
        elif isinstance(expression, CoordinateRef):
            cells = self._program.grid_shape[expression.axis]
            requested = halo[expression.axis]
            lower = _exact_float(self._program.bounds[expression.axis][0])
            coordinate_dtype = (
                torch.float32
                if reference.dtype in (torch.float16, torch.bfloat16)
                else reference.dtype
            )
            indices = torch.arange(
                -requested + expression.offset,
                cells + requested + expression.offset,
                dtype=coordinate_dtype,
                device=reference.device,
            )
            coordinate = lower + (indices + 0.5) * self._spacing[expression.axis]
            shape = [1] * self._ndim
            shape[expression.axis] = cells + 2 * requested
            result = coordinate.reshape(shape).to(reference.dtype)
        elif isinstance(expression, TimeRef):
            time = _broadcast_batch_scalar(context.time, reference, self._ndim)
            if expression.level == TimeLevel.NEW:
                step = self._time_step_value.to(
                    dtype=reference.dtype, device=reference.device
                )
                result = time + step
            else:
                result = time
        elif isinstance(expression, NaryOp):
            values = self._evaluate_arguments(
                expression.arguments,
                context,
                halo,
                session,
                local_cache,
            )
            result = values[0]
            if expression.op == "add":
                for item in values[1:]:
                    result = cast(Any, result) + cast(Any, item)
            elif expression.op == "mul":
                for item in values[1:]:
                    result = cast(Any, result) * cast(Any, item)
            else:
                raise RuntimeError(f"Unknown n-ary operator {expression.op!r}")
        elif isinstance(expression, Power):
            base, exponent = self._evaluate_arguments(
                (expression.base, expression.exponent),
                context,
                halo,
                session,
                local_cache,
            )
            result = torch.pow(cast(Any, base), cast(Any, exponent))
        elif isinstance(expression, Call):
            arguments = self._evaluate_arguments(
                expression.arguments,
                context,
                halo,
                session,
                local_cache,
            )
            if expression.function == "minimum":
                result = cast(torch.Tensor, arguments[0])
                for argument in arguments[1:]:
                    result = torch.minimum(result, cast(torch.Tensor, argument))
            elif expression.function == "maximum":
                result = cast(torch.Tensor, arguments[0])
                for argument in arguments[1:]:
                    result = torch.maximum(result, cast(torch.Tensor, argument))
            else:
                functions = {
                    "sin": torch.sin,
                    "cos": torch.cos,
                    "tan": torch.tan,
                    "asin": torch.asin,
                    "acos": torch.acos,
                    "atan": torch.atan,
                    "sinh": torch.sinh,
                    "cosh": torch.cosh,
                    "tanh": torch.tanh,
                    "asinh": torch.asinh,
                    "acosh": torch.acosh,
                    "atanh": torch.atanh,
                    "exp": torch.exp,
                    "log": torch.log,
                    "abs": torch.abs,
                    "sign": torch.sign,
                    "floor": torch.floor,
                    "ceil": torch.ceil,
                }
                result = functions[expression.function](
                    *(cast(torch.Tensor, argument) for argument in arguments)
                )
        elif isinstance(expression, StencilApply):
            self._evaluate_stencil_group(
                (expression,),
                context,
                halo,
                session,
                local_cache,
            )
            return self._cache_for(expression, session, local_cache)[key]
        else:
            raise RuntimeError(
                f"Unknown expression node {type(expression).__name__}"
            )
        cache[key] = result
        return result

    def _parse_inputs(
        self,
        state: torch.Tensor,
        positional: Sequence[Any],
        named: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        names = (*self._program.parameters, *self._program.coefficient_fields)
        if len(positional) > len(names):
            raise TypeError(f"Expected at most {len(names)} runtime inputs, received {len(positional)}")
        result: dict[str, torch.Tensor] = {}
        for name, value in zip(names, positional, strict=False):
            result[name] = _tensor_or_scalar(value, state)
        for name, value in named.items():
            if name not in names:
                raise TypeError(f"Unknown runtime input {name!r}")
            if name in result:
                raise TypeError(f"Runtime input {name!r} was supplied twice")
            result[name] = _tensor_or_scalar(value, state)
        missing = [name for name in names if name not in result]
        if missing:
            raise TypeError("Missing runtime inputs: " + ", ".join(missing))
        return result

    def _validate_state(self, state: torch.Tensor) -> tuple[int, ...]:
        if not isinstance(state, torch.Tensor) or not torch.is_floating_point(state):
            raise TypeError("State must be a floating-point PyTorch tensor")
        if state.ndim < self._ndim + 1:
            raise ValueError("State shape must be (*batch, fields, *grid_shape)")
        if tuple(state.shape[-self._ndim :]) != self._program.grid_shape:
            raise ValueError(
                f"State spatial shape {tuple(state.shape[-self._ndim:])} does not match grid {self._program.grid_shape}"
            )
        channel_dimension = state.ndim - self._ndim - 1
        if state.shape[channel_dimension] != len(self._program.fields):
            raise ValueError(f"State requires {len(self._program.fields)} field channels")
        return tuple(state.shape[:channel_dimension])

    def _residuals(
        self,
        equation_indices: Sequence[int],
        old_state: torch.Tensor,
        new_state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        session: _EvaluationSession | None = None,
        override_targets: Sequence[int] = (),
        override_values: torch.Tensor | None = None,
        leading_batch_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        context = _EvaluationContext(
            old_state,
            new_state,
            inputs,
            time,
            tuple(override_targets),
            override_values,
            leading_batch_shape,
        )
        expressions = tuple(
            self._program.equations[equation_index].residual
            for equation_index in equation_indices
        )
        values = [
            self._materialize_region(value, context, (0,) * self._ndim)
            for value in self._evaluate_many(expressions, context, session)
        ]
        return _stack_fields(values, self._ndim)

    def _diagonal(
        self,
        equation_indices: Sequence[int],
        old_state: torch.Tensor,
        new_state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        override_targets: Sequence[int] = (),
        override_values: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        context = _EvaluationContext(
            old_state,
            new_state,
            inputs,
            time,
            tuple(override_targets),
            override_values,
        )
        expressions: list[ExpressionIR] = []
        for equation_index in equation_indices:
            expression = self._program.equations[equation_index].diagonal
            if expression is None:
                return None
            expressions.append(expression)
        values = [
            self._materialize_region(value, context, (0,) * self._ndim)
            for value in self._evaluate_many(expressions, context)
        ]
        return _stack_fields(values, self._ndim)

    def _batch_reduce_any(self, value: torch.Tensor, batch_ndim: int) -> torch.Tensor:
        dimensions = tuple(range(batch_ndim, value.ndim))
        return torch.any(value, dim=dimensions) if dimensions else value

    def _solve_direct(
        self,
        block,
        old_state: torch.Tensor,
        layer_state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        batch_ndim: int,
    ) -> tuple[torch.Tensor, KrylovResult]:
        template = _select_fields(layer_state, block.targets, self._ndim)
        equation = self._program.equations[block.equations[0]]
        if equation.direct_solution is not None:
            context = _EvaluationContext(
                old_state,
                layer_state,
                inputs,
                time,
            )
            value = self._evaluate(equation.direct_solution, context)
            solution = _stack_fields(
                (self._materialize_region(value, context, (0,) * self._ndim),),
                self._ndim,
            )
            batch_shape = template.shape[: template.ndim - self._ndim - 1]
            residual_dtype = (
                torch.float32
                if template.dtype in (torch.float16, torch.bfloat16)
                else template.dtype
            )
            residual_norm = torch.zeros(
                batch_shape,
                dtype=residual_dtype,
                device=template.device,
            )
            converged = torch.ones(
                batch_shape, dtype=torch.bool, device=template.device
            )
            iterations = torch.ones(
                batch_shape, dtype=torch.int32, device=template.device
            )
            breakdown = torch.zeros_like(converged)
            relative_residual = torch.zeros_like(residual_norm)
            tolerance_ratio = torch.zeros_like(residual_norm)
            precision_limited = torch.zeros_like(converged)
            return solution, KrylovResult(
                solution,
                converged,
                residual_norm,
                iterations,
                breakdown,
                relative_residual,
                tolerance_ratio,
                precision_limited,
            )
        zero = torch.zeros_like(template)
        session = _EvaluationSession(block.targets)

        def residual(value: torch.Tensor) -> torch.Tensor:
            return self._residuals(
                block.equations,
                old_state,
                layer_state,
                inputs,
                time,
                session,
                block.targets,
                value,
            )

        residual_zero = residual(zero)
        coefficient = self._diagonal(
            block.equations,
            old_state,
            layer_state,
            inputs,
            time,
        )
        if coefficient is None:
            coefficient = residual(torch.ones_like(template)) - residual_zero
        epsilon = torch.finfo(template.dtype).tiny * 16
        invalid_points = (~torch.isfinite(coefficient)) | (torch.abs(coefficient) <= epsilon)
        if template.dtype in (torch.float16, torch.bfloat16):
            residual_zero_work = residual_zero.to(torch.float32)
            coefficient_work = coefficient.to(torch.float32)
            invalid_points = (~torch.isfinite(coefficient_work)) | (
                torch.abs(coefficient_work) <= torch.finfo(torch.float32).tiny * 16
            )
            solution = (
                -residual_zero_work
                / torch.where(invalid_points, 1.0, coefficient_work)
            ).to(template.dtype)
        else:
            solution = -residual_zero / torch.where(invalid_points, 1.0, coefficient)
        final_residual = residual(solution)
        residual_norm = _norm(final_residual, batch_ndim)
        breakdown = self._batch_reduce_any(invalid_points, batch_ndim)
        converged = torch.isfinite(residual_norm) & ~breakdown
        iterations = torch.ones_like(residual_norm, dtype=torch.int32)
        relative_residual, tolerance_ratio, precision_limited = (
            _algebraic_diagnostics(
                -residual_zero,
                final_residual - residual_zero,
                residual_norm,
                converged,
                batch_ndim,
            )
        )
        return solution, KrylovResult(
            solution,
            converged,
            residual_norm,
            iterations,
            breakdown,
            relative_residual,
            tolerance_ratio,
            precision_limited,
        )

    def _pointwise_coefficient_rows(
        self,
        block,
        old_state: torch.Tensor,
        layer_state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        session: _EvaluationSession,
    ) -> list[list[torch.Tensor]]:
        coefficients = block.pointwise_coefficients
        if coefficients is None:
            raise RuntimeError("Pointwise coefficient expressions are unavailable")
        context = _EvaluationContext(old_state, layer_state, inputs, time)
        expressions = tuple(
            expression for row in coefficients for expression in row
        )
        materialized = [
            self._materialize_region(value, context, (0,) * self._ndim)
            for value in self._evaluate_many(expressions, context, session)
        ]
        size = len(coefficients)
        return [
            materialized[row * size : (row + 1) * size]
            for row in range(size)
        ]

    def _apply_constant_pointwise_lu(
        self,
        strategy: _PointwiseStrategy,
        right_hand_side: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        bad_points = torch.zeros_like(right_hand_side[0], dtype=torch.bool)
        work: list[torch.Tensor] = []
        for value in right_hand_side:
            finite = torch.isfinite(value)
            bad_points = bad_points | ~finite
            work.append(torch.where(finite, value, torch.zeros_like(value)))

        epsilon = torch.finfo(right_hand_side[0].dtype).tiny * 16
        if any(
            abs(strategy.lu[index][index]) <= epsilon
            for index in range(len(strategy.lu))
        ):
            return (
                [torch.zeros_like(value) for value in right_hand_side],
                torch.ones_like(bad_points),
            )

        for row, pivot in enumerate(strategy.pivots):
            if row != pivot:
                work[row], work[pivot] = work[pivot], work[row]

        forward: list[torch.Tensor] = []
        for row in range(len(work)):
            value = work[row]
            for column in range(row):
                value = value - strategy.lu[row][column] * forward[column]
            finite = torch.isfinite(value)
            bad_points = bad_points | ~finite
            forward.append(
                torch.where(finite, value, torch.zeros_like(value))
            )

        solution: list[torch.Tensor] = [
            torch.zeros_like(value) for value in right_hand_side
        ]
        for row in range(len(work) - 1, -1, -1):
            value = forward[row]
            for column in range(row + 1, len(work)):
                value = value - strategy.lu[row][column] * solution[column]
            value = value / strategy.lu[row][row]
            finite = torch.isfinite(value)
            bad_points = bad_points | ~finite
            solution[row] = torch.where(
                finite, value, torch.zeros_like(value)
            )
        solution = [
            torch.where(bad_points, torch.zeros_like(value), value)
            for value in solution
        ]
        return solution, bad_points

    def _solve_small_pointwise_rows(
        self,
        matrix: Sequence[Sequence[torch.Tensor]],
        right_hand_side: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        rows: list[list[torch.Tensor]] = []
        bad_points = torch.zeros_like(right_hand_side[0], dtype=torch.bool)
        for row in matrix:
            safe_row: list[torch.Tensor] = []
            for value in row:
                finite = torch.isfinite(value)
                bad_points = bad_points | ~finite
                safe_row.append(
                    torch.where(finite, value, torch.zeros_like(value))
                )
            rows.append(safe_row)
        rhs: list[torch.Tensor] = []
        for value in right_hand_side:
            finite = torch.isfinite(value)
            bad_points = bad_points | ~finite
            rhs.append(torch.where(finite, value, torch.zeros_like(value)))

        size = len(rows)
        epsilon = torch.finfo(right_hand_side[0].dtype).tiny * 16
        for column in range(size):
            pivot_abs = torch.abs(rows[column][column])
            pivot_index = torch.full_like(
                pivot_abs, column, dtype=torch.int32
            )
            for candidate in range(column + 1, size):
                candidate_abs = torch.abs(rows[candidate][column])
                better = candidate_abs > pivot_abs
                pivot_abs = torch.where(better, candidate_abs, pivot_abs)
                pivot_index = torch.where(
                    better,
                    torch.full_like(pivot_index, candidate),
                    pivot_index,
                )

            for candidate in range(column + 1, size):
                swap = pivot_index == candidate
                for item in range(size):
                    top = rows[column][item]
                    bottom = rows[candidate][item]
                    rows[column][item] = torch.where(swap, bottom, top)
                    rows[candidate][item] = torch.where(swap, top, bottom)
                top_rhs = rhs[column]
                bottom_rhs = rhs[candidate]
                rhs[column] = torch.where(swap, bottom_rhs, top_rhs)
                rhs[candidate] = torch.where(swap, top_rhs, bottom_rhs)

            pivot = rows[column][column]
            invalid_pivot = (~torch.isfinite(pivot)) | (
                torch.abs(pivot) <= epsilon
            )
            bad_points = bad_points | invalid_pivot
            safe_pivot = torch.where(
                invalid_pivot, torch.ones_like(pivot), pivot
            )
            for row in range(column + 1, size):
                factor = rows[row][column] / safe_pivot
                finite_factor = torch.isfinite(factor)
                bad_points = bad_points | ~finite_factor
                factor = torch.where(
                    invalid_pivot | ~finite_factor,
                    torch.zeros_like(factor),
                    factor,
                )
                for item in range(column + 1, size):
                    value = rows[row][item] - factor * rows[column][item]
                    finite = torch.isfinite(value)
                    bad_points = bad_points | ~finite
                    rows[row][item] = torch.where(
                        finite, value, torch.zeros_like(value)
                    )
                value = rhs[row] - factor * rhs[column]
                finite = torch.isfinite(value)
                bad_points = bad_points | ~finite
                rhs[row] = torch.where(
                    finite, value, torch.zeros_like(value)
                )

        solution: list[torch.Tensor] = [
            torch.zeros_like(value) for value in right_hand_side
        ]
        for row in range(size - 1, -1, -1):
            value = rhs[row]
            for column in range(row + 1, size):
                value = value - rows[row][column] * solution[column]
            pivot = rows[row][row]
            invalid_pivot = (~torch.isfinite(pivot)) | (
                torch.abs(pivot) <= epsilon
            )
            bad_points = bad_points | invalid_pivot
            safe_pivot = torch.where(
                invalid_pivot, torch.ones_like(pivot), pivot
            )
            value = value / safe_pivot
            finite = torch.isfinite(value)
            bad_points = bad_points | ~finite
            solution[row] = torch.where(
                finite, value, torch.zeros_like(value)
            )
        solution = [
            torch.where(bad_points, torch.zeros_like(value), value)
            for value in solution
        ]
        return solution, bad_points

    def _solve_pointwise_rows_fallback(
        self,
        matrix_rows: Sequence[Sequence[torch.Tensor]],
        right_hand_side: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        matrix = torch.stack(
            tuple(torch.stack(tuple(row), dim=-1) for row in matrix_rows),
            dim=-2,
        )
        rhs = torch.stack(tuple(right_hand_side), dim=-1)
        matrix_finite = torch.all(torch.isfinite(matrix), dim=(-2, -1))
        rhs_finite = torch.all(torch.isfinite(rhs), dim=-1)
        safe_matrix = torch.where(
            matrix_finite[..., None, None], matrix, torch.zeros_like(matrix)
        )
        safe_rhs = torch.where(
            rhs_finite[..., None], rhs, torch.zeros_like(rhs)
        )
        solution, info = torch.linalg.solve_ex(
            safe_matrix, safe_rhs.unsqueeze(-1)
        )
        solution = solution.squeeze(-1)
        solution_finite = torch.all(torch.isfinite(solution), dim=-1)
        bad_points = (
            ~matrix_finite | ~rhs_finite | ~solution_finite | (info != 0)
        )
        solution = torch.where(
            bad_points[..., None], torch.zeros_like(solution), solution
        )
        return [solution[..., index] for index in range(solution.shape[-1])], bad_points

    def _solve_pointwise(
        self,
        block,
        old_state: torch.Tensor,
        layer_state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        batch_ndim: int,
    ) -> tuple[torch.Tensor, KrylovResult]:
        template = _select_fields(layer_state, block.targets, self._ndim)
        zero = torch.zeros_like(template)
        session = _EvaluationSession(block.targets)

        def residual(
            value: torch.Tensor,
            leading_batch_shape: tuple[int, ...] = (),
        ) -> torch.Tensor:
            return self._residuals(
                block.equations,
                old_state,
                layer_state,
                inputs,
                time,
                session,
                block.targets,
                value,
                leading_batch_shape,
            )

        residual_zero = residual(zero)
        size = len(block.targets)
        right_hand_side = [
            -_field(residual_zero, index, self._ndim)
            for index in range(size)
        ]
        reduced_precision = template.dtype in (torch.float16, torch.bfloat16)
        if reduced_precision:
            right_hand_side = [value.to(torch.float32) for value in right_hand_side]
        strategy = self._pointwise_strategies[block.id]
        if strategy.kind == "constant_breakdown":
            solution_rows = [
                torch.zeros_like(value) for value in right_hand_side
            ]
            bad_points = torch.ones_like(
                right_hand_side[0], dtype=torch.bool
            )
        elif strategy.kind == "constant_lu":
            solution_rows, bad_points = self._apply_constant_pointwise_lu(
                strategy, right_hand_side
            )
        else:
            if strategy.kind.startswith("legacy"):
                channel_dimension = template.ndim - self._ndim - 1
                batch_shape = tuple(template.shape[:channel_dimension])
                basis_shape = (
                    size,
                    *((1,) * len(batch_shape)),
                    size,
                    *((1,) * self._ndim),
                )
                basis = torch.eye(
                    size, dtype=template.dtype, device=template.device
                ).reshape(basis_shape).expand(size, *template.shape)
                basis_residual = (
                    residual(basis, (size,)) - residual_zero.unsqueeze(0)
                )
                matrix_last = basis_residual.movedim(
                    (0, channel_dimension + 1), (-1, -2)
                )
                matrix_rows = [
                    [
                        matrix_last[..., row, column]
                        for column in range(size)
                    ]
                    for row in range(size)
                ]
            else:
                matrix_rows = self._pointwise_coefficient_rows(
                    block,
                    old_state,
                    layer_state,
                    inputs,
                    time,
                    session,
                )
            if reduced_precision:
                matrix_rows = [
                    [value.to(torch.float32) for value in row]
                    for row in matrix_rows
                ]
            if strategy.kind in {"variable_small", "legacy_small"}:
                solution_rows, bad_points = self._solve_small_pointwise_rows(
                    matrix_rows, right_hand_side
                )
            else:
                solution_rows, bad_points = self._solve_pointwise_rows_fallback(
                    matrix_rows, right_hand_side
                )
        solution = _stack_fields(solution_rows, self._ndim)
        if reduced_precision:
            solution = solution.to(template.dtype)
        final_residual = residual(solution)
        residual_norm = _norm(final_residual, batch_ndim)
        breakdown = self._batch_reduce_any(bad_points, batch_ndim)
        converged = torch.isfinite(residual_norm) & ~breakdown
        iterations = torch.ones_like(residual_norm, dtype=torch.int32)
        relative_residual, tolerance_ratio, precision_limited = (
            _algebraic_diagnostics(
                -residual_zero,
                final_residual - residual_zero,
                residual_norm,
                converged,
                batch_ndim,
            )
        )
        return solution, KrylovResult(
            solution,
            converged,
            residual_norm,
            iterations,
            breakdown,
            relative_residual,
            tolerance_ratio,
            precision_limited,
        )

    def _linear_options(self, dtype: torch.dtype) -> tuple[int, float, float]:
        rtol_keys = {
            torch.float16: "krylov_rtol_float16",
            torch.bfloat16: "krylov_rtol_bfloat16",
            torch.float32: "krylov_rtol_float32",
            torch.float64: "krylov_rtol_float64",
        }
        if dtype not in rtol_keys:
            raise TypeError("Iterative implicit solves require a floating-point dtype")
        rtol_key = rtol_keys[dtype]
        default_rtols = {
            torch.float16: 1e-3,
            torch.bfloat16: 1e-2,
            torch.float32: 1e-5,
            torch.float64: 1e-8,
        }
        return (
            int(self._options["krylov_max_iterations"]),
            float(self._options.get(rtol_key, default_rtols[dtype])),
            float(self._options["krylov_atol"]),
        )

    def _linear_system(
        self,
        block,
        old_state: torch.Tensor,
        layer_state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        batch_ndim: int,
    ) -> tuple[torch.Tensor, KrylovResult]:
        template = _select_fields(layer_state, block.targets, self._ndim)
        zero = torch.zeros_like(template)
        session = _EvaluationSession(block.targets)

        def residual(value: torch.Tensor) -> torch.Tensor:
            return self._residuals(
                block.equations,
                old_state,
                layer_state,
                inputs,
                time,
                session,
                block.targets,
                value,
            )

        residual_zero = residual(zero)
        rhs = -residual_zero

        def operator(value: torch.Tensor) -> torch.Tensor:
            return residual(value) - residual_zero

        diagonal = self._diagonal(block.equations, old_state, layer_state, inputs, time)
        preconditioner = None
        if diagonal is not None:
            epsilon = torch.finfo(template.dtype).tiny * 16
            preconditioner = _make_jacobi_preconditioner(diagonal, epsilon)
        max_iterations, rtol, atol = self._linear_options(template.dtype)
        atol *= self._residual_scales[block.id]
        if block.properties.constant_nullspace:
            spatial_dims = _spatial_dimensions(rhs, self._ndim)
            compatibility_rhs = (
                rhs.to(torch.float32)
                if rhs.dtype in (torch.float16, torch.bfloat16)
                else rhs
            )
            mean = compatibility_rhs.mean(dim=spatial_dims, keepdim=True)
            scale = torch.amax(
                torch.abs(compatibility_rhs), dim=spatial_dims, keepdim=True
            )
            tolerance = 128 * torch.finfo(compatibility_rhs.dtype).eps * torch.clamp(scale, min=1)
            incompatible = torch.abs(mean) > tolerance
            compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
            if not compiling and bool(torch.any(incompatible).item()):
                maximum = (
                    float(torch.max(torch.abs(mean)).detach().cpu().item())
                    / self._residual_scales[block.id]
                )
                raise BoundaryCompatibilityError(
                    f"Singular constant-nullspace block {block.id} has "
                    f"incompatible RHS mean {maximum:.6e}"
                )
            rhs = (compatibility_rhs - mean).to(rhs.dtype)
            result = projected_conjugate_gradient(
                operator,
                rhs,
                template,
                spatial_ndim=self._ndim,
                preconditioner=preconditioner,
                max_iterations=max_iterations,
                rtol=rtol,
                atol=atol,
                batch_ndim=batch_ndim,
                roundoff_diagonal=diagonal,
            )
            if compiling:
                incompatible_batch = self._batch_reduce_any(incompatible, batch_ndim)
                compatibility_norm = _norm(mean.expand_as(rhs), batch_ndim)
                result = KrylovResult(
                    result.solution,
                    result.converged & ~incompatible_batch,
                    torch.maximum(result.residual_norm, compatibility_norm),
                    result.iterations,
                    result.breakdown | incompatible_batch,
                    torch.where(
                        incompatible_batch,
                        torch.full_like(result.relative_residual, torch.inf),
                        result.relative_residual,
                    ),
                    torch.where(
                        incompatible_batch,
                        torch.full_like(result.tolerance_ratio, torch.inf),
                        result.tolerance_ratio,
                    ),
                    result.precision_limited & ~incompatible_batch,
                )
        elif (
            block.properties.symmetry == PropertyState.PROVEN
            and block.properties.positive_definite == PropertyState.PROVEN
        ):
            result = conjugate_gradient(
                operator,
                rhs,
                template,
                preconditioner=preconditioner,
                max_iterations=max_iterations,
                rtol=rtol,
                atol=atol,
                batch_ndim=batch_ndim,
                roundoff_diagonal=diagonal,
            )
        else:
            result = bicgstab(
                operator,
                rhs,
                template,
                preconditioner=preconditioner,
                max_iterations=max_iterations,
                rtol=rtol,
                atol=atol,
                batch_ndim=batch_ndim,
                roundoff_diagonal=diagonal,
            )
        return result.solution, result

    def _nonlinear_system(
        self,
        block,
        old_state: torch.Tensor,
        layer_state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        batch_ndim: int,
    ) -> tuple[torch.Tensor, KrylovResult]:
        x = _select_fields(layer_state, block.targets, self._ndim)
        if x.dtype in (torch.float16, torch.bfloat16):
            raise TypeError("Nonlinear implicit solves do not support reduced precision")
        max_linear, rtol, linear_atol = self._linear_options(x.dtype)
        newton_atol = float(
            self._options["newton_atol_float32" if x.dtype == torch.float32 else "newton_atol_float64"]
        )
        max_newton = int(self._options["newton_max_iterations"])
        armijo = float(self._options["newton_armijo_constant"])
        session = _EvaluationSession(block.targets)

        def residual(
            value: torch.Tensor,
            leading_batch_shape: tuple[int, ...] = (),
        ) -> torch.Tensor:
            return self._residuals(
                block.equations,
                old_state,
                layer_state,
                inputs,
                time,
                session,
                block.targets,
                value,
                leading_batch_shape,
            )

        value = residual(x)
        residual_norm = _norm(value, batch_ndim)
        initial_residual_norm = residual_norm
        converged = torch.isfinite(residual_norm) & (residual_norm <= newton_atol)
        breakdown = ~torch.isfinite(residual_norm)
        iterations = torch.zeros_like(residual_norm, dtype=torch.int32)
        for _ in range(max_newton):
            if _can_eager_break(converged | breakdown):
                break
            active = ~(converged | breakdown)
            current_x = x

            def jacobian_operator(
                direction: torch.Tensor,
                linearization_point: torch.Tensor = current_x,
            ) -> torch.Tensor:
                return torch.func.jvp(
                    residual, (linearization_point,), (direction,)
                )[1]

            diagonal = self._diagonal(
                block.equations,
                old_state,
                layer_state,
                inputs,
                time,
                block.targets,
                current_x,
            )
            preconditioner = None
            if diagonal is not None:
                epsilon = torch.finfo(x.dtype).tiny * 16
                preconditioner = _make_jacobi_preconditioner(diagonal, epsilon)
            inner = bicgstab(
                jacobian_operator,
                -value,
                torch.zeros_like(x),
                preconditioner=preconditioner,
                max_iterations=max_linear,
                rtol=rtol,
                atol=linear_atol,
                batch_ndim=batch_ndim,
            )
            breakdown = breakdown | (active & (~inner.converged | inner.breakdown))
            direction = inner.solution
            alphas = self._newton_alphas.to(dtype=x.dtype, device=x.device)
            alpha_values = alphas.reshape(
                (alphas.numel(),) + (1,) * x.ndim
            )
            candidates = torch.addcmul(
                x.unsqueeze(0), alpha_values, direction.unsqueeze(0)
            )
            candidate_values = residual(candidates, (alphas.numel(),))
            candidate_norms = _norm(candidate_values, batch_ndim + 1)
            alpha_batch = alphas.reshape(
                (alphas.numel(),) + (1,) * batch_ndim
            )
            sufficient = torch.isfinite(candidate_norms) & (
                candidate_norms
                <= (1.0 - armijo * alpha_batch) * residual_norm.unsqueeze(0)
            )
            accepted = torch.any(sufficient, dim=0)
            first = torch.argmax(sufficient.to(torch.int32), dim=0)
            selected = torch.where(accepted, first, alphas.numel() - 1)

            def select_trial(
                trials: torch.Tensor,
                selected_trial: torch.Tensor = selected,
            ) -> torch.Tensor:
                trailing_ndim = trials.ndim - batch_ndim - 1
                index = selected_trial.reshape(
                    (1, *selected_trial.shape, *((1,) * trailing_ndim))
                ).expand((1, *trials.shape[1:]))
                return torch.take_along_dim(trials, index, dim=0).squeeze(0)

            selected_x = select_trial(candidates)
            selected_value = select_trial(candidate_values)
            active_expanded = _batch_expand(active, x, batch_ndim)
            x = torch.where(active_expanded, selected_x, x)
            value = torch.where(active_expanded, selected_value, value)
            failed_line_search = active & ~accepted
            breakdown = breakdown | failed_line_search
            iterations = iterations + active.to(torch.int32)
            residual_norm = _norm(value, batch_ndim)
            converged = converged | (torch.isfinite(residual_norm) & (residual_norm <= newton_atol))
            breakdown = breakdown | ~torch.isfinite(residual_norm)
        relative_residual = _safe_ratio(residual_norm, initial_residual_norm)
        tolerance_ratio = _safe_ratio(
            residual_norm, torch.full_like(residual_norm, newton_atol)
        )
        return x, KrylovResult(
            x,
            converged & ~breakdown,
            residual_norm,
            iterations,
            breakdown,
            relative_residual,
            tolerance_ratio,
            torch.zeros_like(converged),
        )

    def _execute_step(
        self,
        state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        step_index: int,
    ) -> tuple[SolveOutput, tuple[SolveDiagnostics, ...]]:
        batch_shape = self._validate_state(state)
        batch_ndim = len(batch_shape)
        next_state = state
        overall_converged = torch.ones(batch_shape, dtype=torch.bool, device=state.device)
        residual_dtype = (
            torch.float32
            if state.dtype in (torch.float16, torch.bfloat16)
            else state.dtype
        )
        overall_residual = torch.zeros(
            batch_shape, dtype=residual_dtype, device=state.device
        )
        overall_iterations = torch.zeros(batch_shape, dtype=torch.int32, device=state.device)
        overall_relative_residual = torch.zeros_like(overall_residual)
        overall_tolerance_ratio = torch.zeros_like(overall_residual)
        overall_precision_limited = torch.zeros(
            batch_shape, dtype=torch.bool, device=state.device
        )
        diagnostics: list[SolveDiagnostics] = []
        for layer in self._program.schedule:
            layer_state = next_state
            updates: list[tuple[Sequence[int], torch.Tensor]] = []
            for block in layer.blocks:
                if block.kind == BlockKind.DIRECT:
                    solution, result = self._solve_direct(
                        block, state, layer_state, inputs, time, batch_ndim
                    )
                elif block.kind == BlockKind.POINTWISE_LINEAR:
                    solution, result = self._solve_pointwise(
                        block, state, layer_state, inputs, time, batch_ndim
                    )
                elif block.kind == BlockKind.LINEAR_IMPLICIT:
                    solution, result = self._linear_system(
                        block, state, layer_state, inputs, time, batch_ndim
                    )
                else:
                    solution, result = self._nonlinear_system(
                        block, state, layer_state, inputs, time, batch_ndim
                    )
                residual_scale = self._residual_scales[block.id]
                if residual_scale != 1.0:
                    result = KrylovResult(
                        result.solution,
                        result.converged,
                        result.residual_norm / residual_scale,
                        result.iterations,
                        result.breakdown,
                        result.relative_residual,
                        result.tolerance_ratio,
                        result.precision_limited,
                    )
                updates.append((block.targets, solution))
                overall_converged = overall_converged & result.converged
                overall_residual = torch.maximum(overall_residual, result.residual_norm)
                overall_iterations = torch.maximum(overall_iterations, result.iterations)
                overall_relative_residual = torch.maximum(
                    overall_relative_residual, result.relative_residual
                )
                overall_tolerance_ratio = torch.maximum(
                    overall_tolerance_ratio, result.tolerance_ratio
                )
                overall_precision_limited = (
                    overall_precision_limited | result.precision_limited
                )
                diagnostics.append(
                    SolveDiagnostics(
                        step_index=step_index,
                        block_id=block.id,
                        block_kind=block.kind.value,
                        targets=tuple(self._program.fields[target] for target in block.targets),
                        converged=result.converged,
                        residual_norm=result.residual_norm,
                        iterations=result.iterations,
                        breakdown=result.breakdown,
                        relative_residual=result.relative_residual,
                        tolerance_ratio=result.tolerance_ratio,
                        precision_limited=result.precision_limited,
                    )
                )
            next_state = _commit_field_updates(layer_state, updates, self._ndim)
        return SolveOutput(
            next_state,
            overall_converged,
            overall_residual,
            overall_iterations,
            overall_relative_residual,
            overall_tolerance_ratio,
            overall_precision_limited,
        ), tuple(diagnostics)

    def step(
        self,
        state: torch.Tensor,
        *runtime_inputs: Any,
        t: float | torch.Tensor = 0.0,
        step_index: int = 0,
        **named_inputs: Any,
    ) -> SolveOutput:
        """Advance one time step using positional or symbolic-name runtime inputs."""

        inputs = self._parse_inputs(state, runtime_inputs, named_inputs)
        time = _tensor_or_scalar(t, state)
        return self._execute_step(state, inputs, time, int(step_index))[0]

    def forward(
        self,
        initial_state: torch.Tensor,
        *runtime_inputs: Any,
        t0: float | torch.Tensor = 0.0,
        steps: int = 1,
        **named_inputs: Any,
    ) -> SolveOutput:
        """Advance a fixed number of steps; ``steps`` is specialized by ``torch.compile``."""

        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        inputs = self._parse_inputs(initial_state, runtime_inputs, named_inputs)
        time = _tensor_or_scalar(t0, initial_state)
        state = initial_state
        batch_shape = self._validate_state(state)
        converged = torch.ones(batch_shape, dtype=torch.bool, device=state.device)
        residual_dtype = (
            torch.float32
            if state.dtype in (torch.float16, torch.bfloat16)
            else state.dtype
        )
        residual = torch.zeros(
            batch_shape, dtype=residual_dtype, device=state.device
        )
        iterations = torch.zeros(batch_shape, dtype=torch.int32, device=state.device)
        relative_residual = torch.zeros_like(residual)
        tolerance_ratio = torch.zeros_like(residual)
        precision_limited = torch.zeros(
            batch_shape, dtype=torch.bool, device=state.device
        )
        for step_index in range(steps):
            output, _ = self._execute_step(state, inputs, time, step_index)
            state = output.state
            converged = converged & output.converged
            residual = torch.maximum(residual, output.residual_norm)
            iterations = torch.maximum(iterations, output.iterations)
            relative_residual = torch.maximum(
                relative_residual, output.relative_residual
            )
            tolerance_ratio = torch.maximum(
                tolerance_ratio, output.tolerance_ratio
            )
            precision_limited = precision_limited | output.precision_limited
            time = time + self._dt
        return SolveOutput(
            state,
            converged,
            residual,
            iterations,
            relative_residual,
            tolerance_ratio,
            precision_limited,
        )

    def solve(
        self,
        initial_state: torch.Tensor,
        *,
        parameters: Mapping[Any, Any] | None = None,
        coefficient_fields: Mapping[Any, Any] | None = None,
        t0: float | torch.Tensor = 0.0,
        steps: int = 1,
        raise_on_nonconvergence: bool = True,
    ) -> SolveResult:
        """Eager convenience API accepting mappings and retaining all block diagnostics."""

        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")

        def normalize(values: Mapping[Any, Any] | None) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in (values or {}).items():
                if isinstance(key, str):
                    name = key
                elif hasattr(key, "name"):
                    name = key.name
                elif hasattr(key, "func"):
                    name = key.func.__name__
                else:
                    name = str(key)
                result[name] = value
            return result

        named = normalize(parameters)
        named.update(normalize(coefficient_fields))
        inputs = self._parse_inputs(initial_state, (), named)
        time = _tensor_or_scalar(t0, initial_state)
        state = initial_state
        all_diagnostics: list[SolveDiagnostics] = []
        all_converged = True
        for step_index in range(steps):
            output, diagnostics = self._execute_step(state, inputs, time, step_index)
            state = output.state
            all_diagnostics.extend(diagnostics)
            step_converged = bool(torch.all(output.converged).detach().cpu().item())
            all_converged = all_converged and step_converged
            if raise_on_nonconvergence and not step_converged:
                nonlinear = any(
                    item.block_kind == BlockKind.NONLINEAR_IMPLICIT.value
                    and not bool(torch.all(item.converged).item())
                    for item in diagnostics
                )
                error_type = NonlinearSolveError if nonlinear else LinearSolveError
                raise error_type(
                    f"Step {step_index} did not converge; inspect "
                    "solve(..., raise_on_nonconvergence=False).diagnostics"
                )
            time = time + self._dt
        return SolveResult(state, tuple(all_diagnostics), all_converged)


__all__ = [
    "CompiledSolver",
    "KrylovResult",
    "SolveDiagnostics",
    "SolveOutput",
    "SolveResult",
    "bicgstab",
    "conjugate_gradient",
    "extend_ghosts",
    "ghost_shift",
    "projected_conjugate_gradient",
]
