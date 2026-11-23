"""Thalassa 2.0 typed PDE compiler and PyTorch runtime."""

from .compiler import compile_program, compile_solver, verify
from .dsl import new, old
from .exceptions import (
    BoundaryCompatibilityError,
    CanonicalizationError,
    CompilationError,
    DiscretizationError,
    LinearSolveError,
    ModelValidationError,
    NonlinearSolveError,
    ThalassaError,
)
from .export import export_python
from .ir import (
    BlockKind,
    DiscreteEquationIR,
    GridRef,
    OperatorProperties,
    ProgramIR,
    PropertyState,
    ScheduleLayerIR,
    SolveBlockIR,
    StencilApply,
    TemporalTreatment,
)
from .model import (
    Axis,
    CartesianDomain,
    CoefficientField,
    CompileOptions,
    HomogeneousNeumann,
    KrylovOptions,
    NewtonOptions,
    PDEEquation,
    PDEProblem,
    Periodic,
    ScalarParameter,
    TwoLevelDiscretization,
    UniformDirichlet,
    UniformGrid,
)
from .runtime import CompiledSolver, SolveDiagnostics, SolveOutput, SolveResult
from .schemes import ExpressionRule, arakawa_jacobian
from .stencils import FiniteDifferenceRule, Stencil, backward, centered, forward
from .tpu import TPUCompiledSolver, compile_tpu_solver

__version__ = "2.0.0"

__all__ = [
    "Axis",
    "BlockKind",
    "BoundaryCompatibilityError",
    "CanonicalizationError",
    "CartesianDomain",
    "CoefficientField",
    "CompilationError",
    "CompileOptions",
    "CompiledSolver",
    "DiscreteEquationIR",
    "DiscretizationError",
    "ExpressionRule",
    "FiniteDifferenceRule",
    "GridRef",
    "HomogeneousNeumann",
    "KrylovOptions",
    "LinearSolveError",
    "ModelValidationError",
    "NewtonOptions",
    "NonlinearSolveError",
    "OperatorProperties",
    "PDEEquation",
    "PDEProblem",
    "Periodic",
    "ProgramIR",
    "PropertyState",
    "ScalarParameter",
    "ScheduleLayerIR",
    "SolveBlockIR",
    "SolveDiagnostics",
    "SolveOutput",
    "SolveResult",
    "Stencil",
    "StencilApply",
    "TemporalTreatment",
    "TPUCompiledSolver",
    "ThalassaError",
    "TwoLevelDiscretization",
    "UniformDirichlet",
    "UniformGrid",
    "arakawa_jacobian",
    "backward",
    "centered",
    "compile_program",
    "compile_solver",
    "compile_tpu_solver",
    "export_python",
    "forward",
    "new",
    "old",
    "verify",
]
