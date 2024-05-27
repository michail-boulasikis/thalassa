__version__ = '0.1.0'
__author__ = 'Boulasikis Michail'

from .front_end import PDESystem, PDEFrontEnd
from .discretization import FDApproximation, FDApproximationTaylor, fdm_simple_partial_derivative
from .back_end import pde_compile
