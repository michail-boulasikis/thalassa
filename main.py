from sympy import Eq, Function, symbols
from sympy import Derivative as D
import thalassa

# Traditional sympy symbols and functions
t, x = symbols('t x')
u = Function('u')(t, x)

pde = thalassa.PDESystem(
    # Transport equation (*$u_t + u_x = 0$*), in a list to allow for multiple equations
    [Eq(D(u, t) + D(u, x), 0)],
    # Unknown function(s) in the equation
    [u],
    # Free variables in the equation (thalassa expects time to be the first variable)
    [t, x]
)

disc = [
    thalassa.fdm_simple_partial_derivative(D(u, t), 0.001, method='forward'),
    thalassa.fdm_simple_partial_derivative(D(u, x), 0.05, method='central')
]

with open('transport.py', 'w') as output_file:
    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', plot=True)
    output_file.write(code)
