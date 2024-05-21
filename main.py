from sympy import Eq, Function, symbols
from sympy import Derivative as D
import thalassa

t, x = symbols('t x')
u = Function('u')(t, x)
f = Function('f')

pde = thalassa.PDESystem(
    # Transport equation (*$u_t + u_x = 0$*), in a list to allow for multiple equations
    [Eq(D(u, t) + f(x) * D(u, x), 0)],
    # Unknown function(s) in the equation
    [u],
    # Free variables in the equation (thalassa expects time to be the first variable)
    [t, x]
)

disc = [
    thalassa.FDApproximationTaylor(u, [1, 0], [0, 0], [1, 0], [0.001, 0.05]),
    thalassa.FDApproximationTaylor(u, [0, 1], [0, -1], [0, 1], [0.001, 0.05])
]

print(disc[0].derivative_expr)
print(disc[0].fdm_expr)

#with open('transport.py', 'w') as output_file:
#    code = thalassa.pde_compile(pde, disc, target='pytorch', ics='external', plot=True)
#    output_file.write(code)
