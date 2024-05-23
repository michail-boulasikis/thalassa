from abc import ABC, abstractmethod
from multipledispatch import dispatch
import sympy
import itertools

import thalassa


def _resolve_func_name(func):
    match type(func):
        case sympy.exp:
            return False, "torch.exp"
        case sympy.cos:
            return False, "torch.cos"
        case sympy.sin:
            return False, "torch.sin"
        case sympy.tan:
            return False, "torch.tan"
        case sympy.Abs:
            return False, "torch.abs"
        case sympy.Pow:
            return False, "torch.pow"
        case sympy.log:
            return False, "torch.log"
        case sympy.sqrt:
            return False, "torch.sqrt"
        case sympy.sinh:
            return False, "torch.sinh"
        case sympy.cosh:
            return False, "torch.cosh"
        case sympy.tanh:
            return False, "torch.tanh"
        case sympy.asin:
            return False, "torch.asin"
        case sympy.acos:
            return False, "torch.acos"
        case sympy.atan:
            return False, "torch.atan"
        case sympy.asinh:
            return False, "torch.asinh"
        case sympy.acosh:
            return False, "torch.acosh"
        case sympy.atanh:
            return False, "torch.atanh"
        case _:
            return True, f"__{func.name}"


class CodeEmitter(object):
    def __init__(self):
        self.code = ""
        self.indent = 0
        self.symbol_stack = []
        self.symbol_id = 0

    def make_temp(self):
        name = f"_temp_{self.symbol_id}"
        self.symbol_stack.append(name)
        self.symbol_id += 1
        return name

    @dispatch(str)
    def emit(self, statement):
        self.code += f"{'    ' * self.indent}{statement}\n"

    @dispatch((sympy.Add, sympy.Mul))
    def emit(self, tree):
        res_name = self.make_temp()
        for subexpr in tree.args:
            self.emit(subexpr)
        operands = []
        for i in range(len(tree.args)):
            operands.append(self.symbol_stack.pop())
        operator = '+' if isinstance(tree, sympy.Add) else '*'
        self.emit(f"{res_name} = {f' {operator} '.join(operands)}")

    @dispatch(sympy.Function)
    def emit(self, func):
        res_name = self.make_temp()
        for arg in func.args:
            self.emit(arg)
        args = []
        for i in range(len(func.args)):
            args.append(f"{self.symbol_stack.pop()}")
        self.emit(f"{res_name} = {_resolve_func_name(func)[1]}({', '.join(args)})")

    @dispatch(sympy.Indexed)
    def emit(self, unk):
        res_name = self.make_temp()
        if all(int(i) == 0 for i in unk.indices):
            self.emit(f"{res_name} = _{unk.base.name}")
        else:
            indices = [int(i) for i in unk.indices]
            paddings = [(
                -i if i < 0 else 0,
                i if i > 0 else 0
            ) for i in indices[1:]]
            slices = [f"{j if j != 0 else ''}:{-i if i != 0 else ''}" for i, j in paddings[::-1]]
            slices = ', '.join(slices)
            paddings = list(itertools.chain(*paddings))
            self.emit(f"{res_name} = F.pad(_{unk.base.name}, {str(paddings)}, value=0.)[{slices}]")

    @dispatch((int, float, sympy.core.numbers.NegativeOne, sympy.core.numbers.One, sympy.core.numbers.Zero,
               sympy.core.numbers.Number, sympy.core.numbers.Pi, sympy.core.numbers.Exp1))
    def emit(self, num):
        res_name = self.make_temp()
        self.emit(f"{res_name} = {str(float(num))}")

    @dispatch(sympy.Symbol)
    def emit(self, symbol):
        res_name = self.make_temp()
        self.emit(f"{res_name} = _{symbol.name}")

    def emit_func(self, func, expr, torch_jit=True):
        formatted_args = [f"_{str(arg)}: torch.Tensor" for arg in func.args]
        if torch_jit:
            self.emit("@torch.jit.script")
        self.emit(f"def __{func.name}({', '.join(formatted_args)}) -> torch.Tensor:")
        self.increase_indent()
        if expr is None:
            self.emit("return None")
        else:
            self.emit(expr)
            self.emit(f"return {self.symbol_stack.pop()}")
        self.decrease_indent()
        self.emit("")

    def increase_indent(self):
        self.indent += 1

    def decrease_indent(self):
        self.indent = (self.indent - 1) if self.indent > 0 else 0

    def reset_indent(self):
        self.indent = 0

    def __str__(self):
        return self.code

    def __repr__(self):
        return self.__str__()


class BackEndPhase(ABC):
    def __init__(self, name, description, backend, enabled):
        self.name = name
        self.description = description
        self.backend = backend
        self.enabled = enabled

    def __str__(self):
        return f"BackEnd phase {self.name} ({self.description})"

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class EmitPreambleCodePhase(BackEndPhase):
    def __init__(self, backend, enabled=True, plot=False):
        super().__init__("emit_preamble",
                         "Emits code which includes pytorch libraries and matplotlib if plot is specified.", backend,
                         enabled)
        self.plot = plot

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.preamble_code
        code.emit("import torch")
        code.emit("import torch.nn.functional as F")
        if self.plot:
            code.emit("import matplotlib.pyplot as plt")


class EmitFreeFunctionsPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, func_exprs=None, torch_jit=True):
        super().__init__("emit_free_functions", "Emits code for the free functions in the features.", backend,
                         enabled)
        if func_exprs is None:
            func_exprs = {}
        self.func_exprs = func_exprs
        self.torch_jit = torch_jit

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.free_function_code
        funcs = set()
        for v in self.backend.ir['instr'].values():
            aux_feature_list = v['aux']
            if type(aux_feature_list) is not list:
                continue
            for features in aux_feature_list:
                funcs = funcs.union(features['expr'].atoms(sympy.Function))
        func_dict = {f: self.func_exprs.get(str(f).partition('(')[0], None) for f in funcs}
        for k, v in func_dict.items():
            if _resolve_func_name(k)[0]:
                code.emit_func(k, v, torch_jit=self.torch_jit)


class EmitAuxiliaryFeaturesPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, torch_jit=True):
        super().__init__("emit_auxiliary_features", "Emits code for the auxiliary features.", backend, enabled)
        self.torch_jit = torch_jit

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.aux_feature_code
        for i, (u_var, expr) in enumerate(self.backend.ir['order']):
            feature_func = sympy.Function(f'feature_{i}')(
                *list(filter(lambda x: not isinstance(x, sympy.Indexed), expr.free_symbols)))
            code.emit_func(feature_func, expr, torch_jit=self.torch_jit)


class PytorchPDEBackEnd(object):
    def __init__(self, ir, **kwargs):
        self.ir = ir
        self.preamble_code = CodeEmitter()
        self.free_function_code = CodeEmitter()
        self.aux_feature_code = CodeEmitter()
        self.isl_code = CodeEmitter()
        self.main_func_code = CodeEmitter()
        self.phases = [
            EmitPreambleCodePhase(self, True, kwargs.get('plot', False)),
            EmitFreeFunctionsPhase(self, True, kwargs.get('funcs', {}), kwargs.get("torch_jit", True)),
            EmitAuxiliaryFeaturesPhase(self, True)
        ]

    def execute(self):
        [phase() for phase in self.phases]
        return (f"{self.preamble_code}\n\n{self.free_function_code}\n"
                f"{self.aux_feature_code}\n{self.isl_code}\n{self.main_func_code}")


def pde_compile(pde, disc, **kwargs):
    front_end = thalassa.PDEFrontEnd(pde, disc, **kwargs)
    target_lang = kwargs.get('target', 'pytorch')
    match target_lang:
        case 'pytorch':
            return PytorchPDEBackEnd(front_end.execute(), **kwargs).execute()
        case 'tensorflow':
            raise NotImplementedError("Tensorflow target is not supported yet.")
        case 'mlir':
            raise NotImplementedError("MLIR target is not supported yet.")
        case _:
            raise ValueError(f"Unknown target {target_lang} provided to 'pde_compile'")
