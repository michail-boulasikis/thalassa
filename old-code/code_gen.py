from multipledispatch import dispatch
import sympy
import numpy as np
import sys
import itertools


class CodeHolder(object):
    def __init__(self):
        self.code = ""
        self.indent = 0

    def increase_indent(self):
        self.indent += 1

    def decrease_indent(self):
        self.indent -= 1

    def __str__(self):
        return self.code

    def __add__(self, other):
        self.code += f"{'    ' * self.indent}{other}"
        return self


@dispatch(sympy.Add, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    operands = [node_code_gen_expr(i, code, temp_postfix + f"_{idx}") for idx, i in enumerate(expr.args)]
    code += f"{temp_postfix} = {' + '.join(operands)}\n"
    return temp_postfix


@dispatch(sympy.Mul, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    operands = [node_code_gen_expr(i, code, temp_postfix + f"_{idx}") for idx, i in enumerate(expr.args)]
    code += f"{temp_postfix} = {' * '.join(operands)}\n"
    return temp_postfix


@dispatch(sympy.core.numbers.NegativeOne, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    code += f"{temp_postfix} = -1.0\n"
    return temp_postfix


@dispatch(sympy.core.numbers.One, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    code += f"{temp_postfix} = 1.0\n"
    return temp_postfix


@dispatch(sympy.core.numbers.Zero, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    code += f"{temp_postfix} = 0.0\n"
    return temp_postfix


@dispatch(sympy.Indexed, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    if all(int(i) == 0 for i in expr.indices):
        code += f"{temp_postfix} = {expr.base.name}\n"
        return temp_postfix
    else:
        indices = [int(i) for i in expr.indices]
        paddings = [(
            -i if i < 0 else 0,
            i if i > 0 else 0
        ) for i in indices[1:]]
        slices = [f"{j if j != 0 else ''}:{-i if i != 0 else ''}" for i, j in paddings[::-1]]
        slices = ', '.join(slices)
        paddings = list(itertools.chain(*paddings))
        code += f"{temp_postfix}_0 = F.pad({expr.base.name}, {str(paddings)}, value=0.)\n"
        code += f"{temp_postfix}_1 = {temp_postfix}_0[{slices}]\n"
    return f"{temp_postfix}_1"


@dispatch(sympy.Function, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    code += f"{temp_postfix} = __{expr.name}({', '.join([node_code_gen_expr(e, code, f'{temp_postfix}_{j}') for j, e in enumerate(expr.args)])})\n"
    return temp_postfix


@dispatch(sympy.Symbol, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    code += f"{temp_postfix} = {expr.name}\n"
    return temp_postfix


@dispatch(int, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    code += f"{temp_postfix} = {expr}\n"
    return temp_postfix


@dispatch(float, CodeHolder, str)
def node_code_gen_expr(expr, code, temp_postfix):
    code += f"{temp_postfix} = {expr}\n"
    return temp_postfix


def code_gen_func(fname, fargs, fexpr, code):
    code += "\n@torch.jit.script\n"
    code += f"def __{fname}({', '.join([f'{i}: torch.Tensor' for i in fargs])}) -> torch.Tensor:\n"
    code.increase_indent()
    if fexpr is not None:
        code += f"return {node_code_gen_expr(fexpr, code, 'tmp')}\n\n"
    else:
        code += "return None\n\n"
    code.decrease_indent()
    return code


def generate_pytorch(kernel, instr, prob, **kwargs):
    code = CodeHolder()
    code += "import torch\n"
    code += "import torch.nn.functional as F\n"
    if kwargs.get("plot"):
        code += "import matplotlib.pyplot as plt\n"
    code += "\n"
    for i, f in enumerate(sum([v for v in instr.values()]).atoms(sympy.Function)):
        if f.name not in kwargs:
            print(f"\033[93mWarning:\033[0m '{f}' not found in known functions. "
                  f"The function will be implemented as a dummy function.")
        code = code_gen_func(f.name, f.args, kwargs.get(f), code)
    sorted_v = list(sorted(instr.items(), key=lambda x: x[0]))
    for i, v in sorted_v:
        fargs = prob["unknowns"] + prob["free_symbols"]
        code = code_gen_func(f"feature_{i}", fargs, v, code)
    code += "\n"

    code += "@torch.jit.script\n"
    code += (f"def _integrate(ic: torch.Tensor, N: torch.Tensor, "
             f"{', '.join([f'{i}: torch.Tensor' for i in prob['free_symbols']])}) -> torch.Tensor:\n")
    code.increase_indent()
    code += f"__feature_maps_shape = [{len(instr)}] + [int(i) for i in N[1:]]\n"
    code += f"__feature_maps = torch.zeros(__feature_maps_shape)\n"
    with np.printoptions(threshold=sys.maxsize):
        code += f"__conv_kernel = torch.tensor({kernel.tolist()})\n"
    code += "__kernel_perm = [0] + list(range(len(N), 0, -1))\n"
    # Permute the kernel in this script also:
    kernel = np.transpose(kernel, [0] + list(range(len(kernel.shape) - 1, 0, -1)))
    code += "__conv_kernel = torch.permute(__conv_kernel, __kernel_perm)\n"
    code += f"for _i in range(int(ic.size(0))):\n"
    code.increase_indent()
    code += f"__feature_maps[_i] = ic[_i]\n"
    code.decrease_indent()
    code += f"for _i in range(int(N[0])):\n"
    code.increase_indent()
    for j in range(len(instr)):
        code += (f"__feature_maps[{j}, ...] = __feature_{j}"
                 f"({', '.join([f'__feature_maps[{k}]' for k in range(len(prob['unknowns']))])}, "
                 f"{', '.join(prob['free_symbols'])})\n")
    code += f"_tmp = F.conv{len(prob['free_symbols']) - 1}d(__feature_maps[None, ...], __conv_kernel)\n"
    for j in range(len(prob['unknowns'])):
        code += f"_tmp_{j} = _tmp[0, {j}, ...]\n"
        padding_shape = [(kernel.shape[i] // 2, kernel.shape[i] // 2) for i in range(2, len(kernel.shape))]
        padding_shape = list(itertools.chain(*padding_shape))
        code += f"_tmp_{j} = F.pad(_tmp_{j}, {padding_shape}, value=0.)\n"
        code += f"__feature_maps[{j}, ...] = _tmp_{j}\n"
    code.decrease_indent()
    code += f"return __feature_maps[:{len(prob['unknowns'])}, ...]\n"
    code.decrease_indent()

    code += "\n"
    code += "def main():\n"
    code.increase_indent()
    code += f"_N = torch.tensor([101, {', '.join([str(41)] * (len(prob['free_symbols']) - 1))}], dtype=torch.int32)\n"
    code += f"_dt = 1 / (_N - 1)\n"
    for i, v in enumerate(prob["free_symbols"]):
        code += f"_{v} = torch.linspace(0, 1, int(_N[{i}]))\n"
    code += "\n"
    code += (f"_ic = torch.zeros(({len(prob['unknowns'])}, "
             f"{', '.join([f'_N[{i}]' for i in range(1, len(prob['free_symbols']))])}))\n")
    for i, v in enumerate(prob["unknowns"]):
        code += f"_ic[{i}, :] = __ic_{v}({', '.join([f'_{str(x)}' for x in prob['free_symbols'][1:]])})\n"
    code += f"result = _integrate(_ic, _N, {', '.join([f'_{i}' for i in prob['free_symbols']])})\n"
    if kwargs.get("plot"):
        dims = len(prob["free_symbols"]) - 1
        plot_type = len(prob["unknowns"])
        if dims == 3:
            print("\033[93mWarning:\033[0m 3D plots are not supported.")
            code.decrease_indent()
            code += "\n"
            code += "\n"
            code += "main()\n"
            return str(code)
        if plot_type == 1:
            if dims == 1:
                code += f"plt.plot(_{prob['free_symbols'][1]}.numpy(), result[0, ...].numpy())\n"
            elif dims == 2:
                code += f"plt.imshow(result[0, ...].numpy())\n"
                code += f"plt.colorbar()\n"
        elif plot_type == 2:
            if dims == 1:
                code += f"plt.plot(_{prob['free_symbols'][1]}.numpy(), result[0, ...].numpy())\n"
                code += f"plt.plot(_{prob['free_symbols'][1]}.numpy(), result[1, ...].numpy())\n"
            elif dims == 2:
                code += (f"plt.quiver(_{prob['free_symbols'][1]}.numpy(), _{prob['free_symbols'][2]}.numpy(), "
                         f"result[0, ...].numpy(), result[1, ...].numpy())\n")
        code += "plt.show()\n"
    code.decrease_indent()
    code += "\n"
    code += "\n"
    code += "main()\n"
    return str(code)
