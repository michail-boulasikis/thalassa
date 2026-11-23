"""Experimental single-device PyTorch/XLA adapter.

PyTorch/XLA is deliberately imported only by :func:`compile_tpu_solver`.  The
ordinary compiler, IR, and runtime therefore remain usable without the TPU
extra installed.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from typing import Any

import torch

from .compiler import compile_program
from .exceptions import CompilationError
from .ir import BlockKind, ProgramIR
from .model import CompileOptions, PDEProblem, UniformGrid
from .runtime import CompiledSolver, SolveOutput, _exact_float

_TPU_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))
_INSTALL_GUIDANCE = (
    "Install the matching TPU packages with "
    "`python -m pip install 'thalassa[tpu]'`; Thalassa's TPU environment "
    "uses matching torch and torch-xla 2.8.0 releases."
)


def _release_version(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def _load_torch_xla():
    try:
        torch_xla = importlib.import_module("torch_xla")
    except (ImportError, OSError, RuntimeError) as error:
        raise CompilationError(
            "The experimental TPU backend requires PyTorch/XLA. "
            + _INSTALL_GUIDANCE
        ) from error

    torch_release = _release_version(torch.__version__)
    xla_release = _release_version(str(getattr(torch_xla, "__version__", "")))
    if torch_release is None or xla_release is None or torch_release != xla_release:
        raise CompilationError(
            "PyTorch and PyTorch/XLA release versions must match exactly "
            f"(found torch {torch.__version__!r} and torch-xla "
            f"{getattr(torch_xla, '__version__', 'unknown')!r}). "
            + _INSTALL_GUIDANCE
        )
    if not callable(getattr(torch_xla, "device", None)):
        raise CompilationError(
            "This PyTorch/XLA build does not expose torch_xla.device(). "
            + _INSTALL_GUIDANCE
        )
    return torch_xla


def _validate_program(program: ProgramIR, solver: CompiledSolver) -> None:
    for layer in program.schedule:
        for block in layer.blocks:
            if block.kind == BlockKind.NONLINEAR_IMPLICIT:
                raise CompilationError(
                    f"TPU v1 does not support nonlinear block {block.id}; "
                    "Newton/JVP lowering remains unsupported."
                )
            if block.kind == BlockKind.POINTWISE_LINEAR:
                strategy = solver._pointwise_strategies[block.id]
                if strategy.kind in {"variable_fallback", "legacy_fallback"}:
                    raise CompilationError(
                        f"TPU v1 supports variable pointwise blocks only through 4x4; "
                        f"block {block.id} would require torch.linalg.solve_ex."
                    )


def _rebuild_reduced_coordinates(solver: CompiledSolver) -> None:
    """Construct coordinate sources in FP32 before a reduced-precision cast."""

    for axis, ((lower_text, upper_text), cells) in enumerate(
        zip(solver.program.bounds, solver.program.grid_shape, strict=True)
    ):
        lower = _exact_float(lower_text)
        upper = _exact_float(upper_text)
        spacing = (upper - lower) / cells
        coordinate = lower + (
            torch.arange(cells, dtype=torch.float32) + 0.5
        ) * spacing
        shape = [1] * len(solver.program.grid_shape)
        shape[axis] = cells
        setattr(solver, f"coordinate_{axis}", coordinate.reshape(shape))


def _preflight_buffers(solver: CompiledSolver, dtype: torch.dtype) -> None:
    for name, buffer in solver.named_buffers():
        if not torch.is_floating_point(buffer):
            continue
        cast_buffer = buffer.detach().to(device="cpu", dtype=dtype)
        if not bool(torch.all(torch.isfinite(cast_buffer)).item()):
            raise CompilationError(
                f"TPU {dtype} preflight found a non-finite cast in solver buffer "
                f"{name!r}; reduce coefficient magnitudes or use a wider dtype."
            )


class TPUCompiledSolver(torch.nn.Module):
    """Single-XLA-device wrapper around a full-graph compiled v2 solver."""

    def __init__(
        self,
        solver: CompiledSolver,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device
        self._solver = solver
        self.metadata = solver.metadata
        self._compiled_solver = torch.compile(
            solver,
            backend="openxla",
            fullgraph=True,
            dynamic=False,
        )

    @property
    def program(self) -> ProgramIR:
        return self._solver.program

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def _validate_tensor(self, name: str, value: torch.Tensor) -> None:
        if value.device != self._device:
            raise ValueError(
                f"{name} must already be on {self._device}; received {value.device}"
            )
        if value.dtype != self._dtype:
            raise TypeError(
                f"{name} must already have dtype {self._dtype}; received {value.dtype}"
            )

    def _validate_invocation(
        self,
        state: torch.Tensor,
        positional: tuple[Any, ...],
        named: Mapping[str, Any],
        time: float | torch.Tensor,
    ) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError("State must be a PyTorch tensor")
        self._validate_tensor("state", state)
        for index, value in enumerate(positional):
            if isinstance(value, torch.Tensor):
                self._validate_tensor(f"runtime input {index}", value)
        for name, value in named.items():
            if isinstance(value, torch.Tensor):
                self._validate_tensor(f"runtime input {name!r}", value)
        if isinstance(time, torch.Tensor):
            self._validate_tensor("time", time)

    def step(
        self,
        state: torch.Tensor,
        *runtime_inputs: Any,
        t: float | torch.Tensor = 0.0,
        step_index: int = 0,
        **named_inputs: Any,
    ) -> SolveOutput:
        """Advance one lazily compiled TPU step."""

        if not isinstance(step_index, int):
            raise TypeError("step_index must be an integer")
        self._validate_invocation(
            state, runtime_inputs, named_inputs, t
        )
        return self._compiled_solver(
            state,
            *runtime_inputs,
            t0=t,
            steps=1,
            **named_inputs,
        )

    def forward(
        self,
        initial_state: torch.Tensor,
        *runtime_inputs: Any,
        t0: float | torch.Tensor = 0.0,
        steps: int = 1,
        **named_inputs: Any,
    ) -> SolveOutput:
        """Advance a statically specialized number of TPU steps."""

        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        self._validate_invocation(
            initial_state, runtime_inputs, named_inputs, t0
        )
        return self._compiled_solver(
            initial_state,
            *runtime_inputs,
            t0=t0,
            steps=steps,
            **named_inputs,
        )

    def solve(self, *args: Any, **kwargs: Any):
        del args, kwargs
        raise CompilationError(
            "TPUCompiledSolver.solve() is unavailable because it performs eager "
            "host synchronization. Use forward() or step() and synchronize at "
            "the application boundary with torch_xla.sync(wait=True)."
        )


def compile_tpu_solver(
    problem: PDEProblem,
    grid: UniformGrid,
    options: CompileOptions | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
) -> TPUCompiledSolver:
    """Compile a v2 solver for one Google TPU device through OpenXLA."""

    if dtype not in _TPU_DTYPES:
        supported = "torch.float16, torch.bfloat16, or torch.float32"
        raise TypeError(f"TPU dtype must be {supported}; received {dtype}")
    torch_xla = _load_torch_xla()
    try:
        selected_device = torch.device(
            torch_xla.device(0) if device is None else device
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise CompilationError(
            "Could not acquire the requested single XLA device. "
            "Verify that PJRT_DEVICE=TPU and TPU runtime access are configured."
        ) from error
    if selected_device.type != "xla":
        raise CompilationError(
            f"TPU device must be an XLA device; received {selected_device}"
        )

    program = compile_program(problem, grid, options)
    return _compile_tpu_program(
        program,
        dtype=dtype,
        device=selected_device,
    )


def _compile_tpu_program(
    program: ProgramIR,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TPUCompiledSolver:
    """Lower an already compiled program (used by the feasibility harness)."""

    solver = CompiledSolver(program)
    _validate_program(program, solver)
    if dtype in (torch.float16, torch.bfloat16):
        _rebuild_reduced_coordinates(solver)
    _preflight_buffers(solver, dtype)
    try:
        solver.to(device=device, dtype=dtype)
    except RuntimeError as error:
        raise CompilationError(
            f"Failed to materialize solver buffers on {device}; verify TPU runtime access."
        ) from error
    return TPUCompiledSolver(
        solver,
        dtype=dtype,
        device=device,
    )


__all__ = ["TPUCompiledSolver", "compile_tpu_solver"]
