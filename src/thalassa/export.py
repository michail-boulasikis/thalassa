"""Deterministic thin Python export for compiled solvers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .ir import ProgramIR

if TYPE_CHECKING:
    from .runtime import CompiledSolver


def _source_for(program: ProgramIR) -> str:
    serialized = json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    source = (
        "import json as _json\n"
        "import thalassa.runtime as _runtime\n\n"
        f"PROGRAM = _json.loads({serialized!r})\n\n"
        "def load_solver():\n"
        "    return _runtime.CompiledSolver(PROGRAM)\n\n"
        "solver = load_solver()\n"
    )
    compile(source, "<thalassa-export>", "exec")
    return source


def export_python(
    solver_or_program: CompiledSolver | ProgramIR,
    path: str | Path | None = None,
) -> str:
    """Emit a deterministic module importing the shared Thalassa runtime.

    The returned string is always validated with :func:`compile`.  When *path*
    is supplied it is also written atomically after the parent directory has
    been validated.
    """

    from .runtime import CompiledSolver

    if isinstance(solver_or_program, CompiledSolver):
        program = solver_or_program.program
    elif isinstance(solver_or_program, ProgramIR):
        program = solver_or_program
    else:
        raise TypeError("export_python expects a CompiledSolver or ProgramIR")
    source = _source_for(program)
    if ProgramIR.from_dict(program.to_dict()) != program:
        raise ValueError("ProgramIR failed deterministic serialization round-trip validation")
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(destination)
        imported = import_export(destination)
        if imported.solver.program != program or imported.solver.metadata.version != program.version:
            raise ValueError("Exported solver failed isolated import validation")
    return source


def import_export(path: str | Path):
    """Import an exported module in isolation (primarily useful in tests)."""

    destination = Path(path)
    specification = importlib.util.spec_from_file_location("_thalassa_export", destination)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import exported solver from {destination}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


__all__ = ["export_python", "import_export"]
