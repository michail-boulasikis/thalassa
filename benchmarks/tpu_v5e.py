"""Single-device v5e feasibility and correctness matrix.

The benchmark intentionally synchronizes every timed invocation.  Run it on a
TPU VM; importing this module on a normal development machine does not require
PyTorch/XLA.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(SOURCE))

from benchmarks.runtime_workloads import (  # noqa: E402
    explicit_heat,
    implicit_advection,
    implicit_heat,
    pointwise_variable_4x4,
)
from examples.navier_stokes import make_solver, taylor_green_state  # noqa: E402
from thalassa.tpu import _compile_tpu_program  # noqa: E402

REFERENCE_MATRIX = {
    "explicit_heat": {"cells": 65_536, "batch": 32, "steps": 8},
    "implicit_heat_cg": {"cells": 4_096, "batch": 16, "steps": 1},
    "implicit_advection_bicgstab": {
        "cells": 4_096,
        "batch": 16,
        "steps": 1,
    },
    "variable_pointwise_4x4": {
        "cells": 16_384,
        "batch": 16,
        "steps": 1,
    },
    "taylor_green_projected_cg": {
        "cells": 128 * 128,
        "grid": [128, 128],
        "batch": 1,
        "steps": 1,
    },
}

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
ERROR_LIMITS = {
    "float32": 2e-4,
    "float16": 1e-2,
    "bfloat16": 5e-2,
}


def _workload(name: str):
    specification = REFERENCE_MATRIX[name]
    if name == "explicit_heat":
        return (
            *explicit_heat(
                specification["cells"], specification["batch"], torch.float64
            ),
            specification["steps"],
        )
    if name == "implicit_heat_cg":
        return (
            *implicit_heat(
                specification["cells"], specification["batch"], torch.float64
            ),
            specification["steps"],
        )
    if name == "implicit_advection_bicgstab":
        return (
            *implicit_advection(
                specification["cells"], specification["batch"], torch.float64
            ),
            specification["steps"],
        )
    if name == "variable_pointwise_4x4":
        return (
            *pointwise_variable_4x4(
                specification["cells"], specification["batch"], torch.float64
            ),
            specification["steps"],
        )
    solver = make_solver(128)
    state = taylor_green_state(128, 0.0, 1e-2).unsqueeze(0)
    return solver, state, (torch.tensor(1e-2, dtype=torch.float64),), 1


def _metric(metrics, name: str) -> dict[str, Any] | None:
    value = metrics.metric_data(name)
    if value is None:
        return None
    return {"samples": int(value[0]), "accumulator": float(value[1])}


def _compile_samples(metrics) -> int:
    value = _metric(metrics, "CompileTime")
    return 0 if value is None else int(value["samples"])


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)]


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.linalg.vector_norm(expected)
    return float((numerator / torch.clamp(denominator, min=torch.finfo(expected.dtype).tiny)).item())


def run_case(name: str, dtype_name: str, warmup: int, samples: int) -> dict[str, Any]:
    import torch_xla
    import torch_xla.debug.metrics as metrics

    dtype = DTYPES[dtype_name]
    eager_solver, state_cpu, inputs_cpu, steps = _workload(name)
    reference = eager_solver.to(dtype=torch.float64)(state_cpu, *inputs_cpu, steps=steps)

    device = torch.device(torch_xla.device(0))
    solver = _compile_tpu_program(eager_solver.program, dtype=dtype, device=device)
    state = state_cpu.to(device=device, dtype=dtype)
    inputs = tuple(value.to(device=device, dtype=dtype) for value in inputs_cpu)

    metrics.clear_all()
    start = perf_counter()
    output = solver(state, *inputs, steps=steps)
    torch_xla.sync(wait=True)
    first_call = perf_counter() - start

    for _ in range(warmup):
        output = solver(state, *inputs, steps=steps)
        torch_xla.sync(wait=True)
    compile_count_after_warmup = _compile_samples(metrics)

    durations: list[float] = []
    for _ in range(samples):
        start = perf_counter()
        output = solver(state, *inputs, steps=steps)
        torch_xla.sync(wait=True)
        durations.append(perf_counter() - start)
    compile_count_after_measurement = _compile_samples(metrics)

    actual = output.state.cpu().to(torch.float64)
    converged = bool(torch.all(output.converged.cpu()).item())
    finite = bool(torch.all(torch.isfinite(actual)).item())
    relative_error = _relative_l2(actual, reference.state)
    fallback_getter = getattr(metrics, "executed_fallback_ops", None)
    fallback_operations = list(fallback_getter()) if callable(fallback_getter) else []
    median = statistics.median(durations)
    cells = int(REFERENCE_MATRIX[name]["cells"])
    batch = int(REFERENCE_MATRIX[name]["batch"])
    return {
        "workload": name,
        "dtype": dtype_name,
        "shape": REFERENCE_MATRIX[name],
        "first_call_seconds": first_call,
        "median_seconds": median,
        "p95_seconds": _p95(durations),
        "throughput_cell_steps_per_second": cells * batch * steps / median,
        "compile_count_after_warmup": compile_count_after_warmup,
        "compile_count_after_measurement": compile_count_after_measurement,
        "post_warmup_recompilations": compile_count_after_measurement - compile_count_after_warmup,
        "xla_compile_time": _metric(metrics, "CompileTime"),
        "xla_execute_time": _metric(metrics, "ExecuteTime"),
        "fallback_operations": fallback_operations,
        "converged": converged,
        "finite": finite,
        "true_residual_max": float(torch.max(output.residual_norm.cpu()).item()),
        "relative_l2_error": relative_error,
        "relative_l2_limit": ERROR_LIMITS[dtype_name],
        "accepted": (
            finite
            and converged
            and not fallback_operations
            and compile_count_after_measurement == compile_count_after_warmup
            and relative_error <= ERROR_LIMITS[dtype_name]
        ),
        "xla_metrics_report": metrics.metrics_report(),
    }


def run_baseline(name: str, device_name: str, samples: int) -> dict[str, Any]:
    dtype = torch.float64 if device_name == "cpu" else torch.float32
    solver, state, inputs, steps = _workload(name)
    solver = solver.to(device=device_name, dtype=dtype)
    state = state.to(device=device_name, dtype=dtype)
    inputs = tuple(value.to(device=device_name, dtype=dtype) for value in inputs)
    for _ in range(2):
        output = solver(state, *inputs, steps=steps)
    if device_name == "cuda":
        torch.cuda.synchronize()
    durations: list[float] = []
    for _ in range(samples):
        start = perf_counter()
        output = solver(state, *inputs, steps=steps)
        if device_name == "cuda":
            torch.cuda.synchronize()
        durations.append(perf_counter() - start)
    median = statistics.median(durations)
    specification = REFERENCE_MATRIX[name]
    return {
        "workload": name,
        "device": device_name,
        "mode": "eager",
        "dtype": str(dtype),
        "median_seconds": median,
        "p95_seconds": _p95(durations),
        "throughput_cell_steps_per_second": (
            int(specification["cells"])
            * int(specification["batch"])
            * steps
            / median
        ),
        "converged": bool(torch.all(output.converged.cpu()).item()),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Thalassa experimental TPU v5e report",
        "",
        f"Status: **{payload['status']}**",
        "",
        "Hardware and software",
        "",
        f"- Hardware: {payload['environment'].get('hardware', 'not recorded')}",
        f"- Python: {payload['environment']['python']}",
        f"- PyTorch: {payload['environment']['torch']}",
        f"- PyTorch/XLA: {payload['environment']['torch_xla']}",
        "",
        "Correctness and performance",
        "",
        "| workload | dtype | first call (s) | median (s) | p95 (s) | "
        "converged | true residual | relative L2 | accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["results"]:
        lines.append(
            f"| {item['workload']} | {item['dtype']} | {item['first_call_seconds']:.6g} | "
            f"{item['median_seconds']:.6g} | {item['p95_seconds']:.6g} | "
            f"{item['converged']} | {item['true_residual_max']:.6g} | "
            f"{item['relative_l2_error']:.6g} | {item['accepted']} |"
        )
    lines.extend(["", "CPU/CUDA comparisons", ""])
    comparisons = payload["cpu_cuda_comparisons"]
    if comparisons:
        lines.extend(
            [
                "| workload | device | dtype | median (s) | p95 (s) | converged |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            lines.append(
                f"| {item['workload']} | {item['device']} | {item['dtype']} | "
                f"{item['median_seconds']:.6g} | {item['p95_seconds']:.6g} | "
                f"{item['converged']} |"
            )
    else:
        lines.append("Not collected. No speedup is required.")
    lines.extend(["", "Conclusion", "", payload["conclusion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=(*REFERENCE_MATRIX, "all"), default="all")
    parser.add_argument("--dtype", choices=(*DTYPES, "all"), default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--output", type=Path, default=SOURCE / "benchmarks/results/tpu_v5e_raw.json")
    parser.add_argument("--report", type=Path, default=SOURCE / "benchmarks/results/tpu_v5e_report.md")
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.samples <= 0:
        parser.error("--warmup must be nonnegative and --samples must be positive")

    import torch_xla

    workloads = REFERENCE_MATRIX if arguments.workload == "all" else (arguments.workload,)
    dtypes = DTYPES if arguments.dtype == "all" else (arguments.dtype,)
    results = [
        run_case(workload, dtype, arguments.warmup, arguments.samples)
        for workload in workloads
        for dtype in dtypes
    ]
    comparisons = []
    if arguments.baselines:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        comparisons = [
            run_baseline(workload, device, min(arguments.samples, 5))
            for workload in workloads
            for device in devices
        ]
    accepted = all(item["accepted"] for item in results)
    payload = {
        "schema_version": 1,
        "status": "accepted" if accepted else "failed",
        "environment": {
            "hardware": "Google Cloud TPU v5e-8 (single xla:0 device)",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_xla": torch_xla.__version__,
        },
        "command": " ".join(sys.argv),
        "reference_matrix": REFERENCE_MATRIX,
        "results": results,
        "cpu_cuda_comparisons": comparisons,
        "conclusion": (
            "All measured acceptance checks passed; performance is reported without a speedup requirement."
            if accepted
            else "At least one correctness, fallback, or recompilation acceptance check failed."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(_markdown(payload))
    print(json.dumps({"output": str(arguments.output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
