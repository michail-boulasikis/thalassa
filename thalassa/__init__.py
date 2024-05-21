__version__ = '0.1.0'
__author__ = 'Boulasikis Michail'

from .sym_pde import PDESystem
from .discretization import FDApproximationTaylor, fdm_simple_partial_derivative
from .code_gen import pde_compile
