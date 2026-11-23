"""Public exception hierarchy for Thalassa."""


class ThalassaError(Exception):
    """Base class for every error raised by Thalassa."""


class ModelValidationError(ThalassaError):
    """The continuous PDE model is incomplete or inconsistent."""


class DiscretizationError(ThalassaError):
    """A spatial derivative cannot be discretized safely."""


class CanonicalizationError(ThalassaError):
    """A discrete residual cannot be put into the typed canonical form."""


class CompilationError(ThalassaError):
    """A compiler pass failed while constructing a solver."""


class BoundaryCompatibilityError(ThalassaError):
    """A singular boundary-value problem has an incompatible right-hand side."""


class LinearSolveError(ThalassaError):
    """An iterative linear solve did not converge."""


class NonlinearSolveError(ThalassaError):
    """A nonlinear solve did not converge."""
