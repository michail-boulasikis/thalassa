from lib.sym_pde import StencilCompiler
import sympy

discretization = {
    # du/dx ~= 0.005 * (u(t, x + 1) - u(t, x - 1))
    "x": {
        "coeffs": [0.005, -0.005],
        "indices": [[0, 1], [0, -1]]
    },
    # du/dt ~= 0.01 * (u(t + 1, x) - u(t, x))
    "t": {
        "coeffs": [0.01, -0.01],
        "indices": [[1, 0], [0, 0]]
    }
}

# The PDE
x, t = sympy.symbols("x t")
u, m = map(sympy.Function, ["u", "m"])
pde = sympy.Eq(sympy.Derivative(u(x, t), t) + u(x, t) * sympy.Derivative(u(x, t), x), 0)

sc = StencilCompiler(pde, discretization)
stencil, instructions = sc.compile()
print(f"{stencil=}\n{instructions=}")
