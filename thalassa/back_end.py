import functools
from abc import ABC, abstractmethod
from multipledispatch import dispatch
import numpy as np
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

    @dispatch(sympy.Pow)
    def emit(self, tree):
        res_name = self.make_temp()
        self.emit(tree.base)
        self.emit(tree.exp)
        exp = self.symbol_stack.pop()
        base = self.symbol_stack.pop()
        self.emit(f"{res_name} = torch.pow({base}, {exp})")

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
    def __init__(self, backend, enabled=True, plot=False, ics='external'):
        super().__init__("emit_preamble",
                         "Emits code which includes pytorch libraries and matplotlib if plot is specified.", backend,
                         enabled)
        self.plot = plot
        self.ics = ics

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.preamble_code
        if self.ics == 'external':
            code.emit("import sys")
            code.emit("import numpy as np")
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
        for i, (u_var, c, expr) in enumerate(self.backend.ir['order']):
            feature_func = sympy.Function(f'feature_{i}')(
                *list(filter(lambda x: not isinstance(x, sympy.Indexed), expr.free_symbols)))
            self.backend.feature_funcs.append(feature_func)
            code.emit_func(feature_func, expr, torch_jit=self.torch_jit)


def _find_iminmax(disc):
    imin = np.zeros(len(disc[0].order), dtype=int) + 100
    imax = np.zeros(len(disc[0].order), dtype=int) - 100
    for d in disc:
        imin = np.minimum(imin, d.imin)
        imax = np.maximum(imax, d.imax)
    return imin, imax


class CalculateStencilShapePhase(BackEndPhase):
    def __init__(self, backend, enabled=True, use_conv=True):
        super().__init__("calculate_stencil", "Calculates the stencil for the PDE.", backend, enabled)
        self.use_conv = use_conv

    def __call__(self):
        if not (self.enabled and self.use_conv):
            return
        disc = self.backend.ir['disc']
        # NOTE: The assumption here is that all variables have the same number of dimensions
        imin, imax = _find_iminmax(disc)
        stencil_shape = [len(self.backend.ir['pdes'].unknowns)] + (imax - imin + 1).tolist()
        # NOTE: To fix, this assumes that every equation has the order in time
        stencil_shape[1] = (stencil_shape[1] - 1) * len(self.backend.ir['pdes'].unknowns)
        # Auxiliary features will lie in the same dimension as time, so that they fit in the channel dimension
        stencil_shape[1] += sum(
            len(v['aux']) for v in self.backend.ir['instr'].values() if type(v['aux']) is list)
        stencil_shape.append(stencil_shape[1])
        del stencil_shape[1]
        self.backend.stencil = np.zeros(shape=stencil_shape)


class CalculateStencilCoeffPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, use_conv=True):
        super().__init__("calculate_stencil_coeff", "Calculates the stencil coefficients for the PDE.", backend,
                         enabled)
        self.use_conv = use_conv

    def __call__(self):
        if not (self.enabled and self.use_conv):
            return
        disc = self.backend.ir['disc']
        stencil = self.backend.stencil
        aux_feature_count = 0
        unknown_names = list(map(lambda x: x.name, self.backend.ir['pdes'].unknowns))
        n_vars = len(self.backend.ir['pdes'].unknowns)
        for u_var, comp in self.backend.ir['instr'].items():
            linear_comp = comp['conv']
            aux_features = comp['aux']
            u_var_idx = unknown_names.index(u_var)
            imin, imax = _find_iminmax(disc)
            if linear_comp is not None and linear_comp != 0:
                for term in sympy.Add.make_args(linear_comp):
                    base_var_idx = unknown_names.index(term.atoms(sympy.Indexed).pop().base.name)
                    coeff = functools.reduce(lambda x, y: x * y,
                                             [c for c in term.as_ordered_factors() if
                                              c.is_constant() and not isinstance(c, sympy.Indexed)] or [1.0])
                    index = [int(i) for i in list(term.atoms(sympy.Indexed)).pop().indices]
                    stencil_idx = [u_var_idx] + [i - j for i, j in zip(index, imin)]
                    stencil_idx.append(stencil_idx[1] + base_var_idx)
                    del stencil_idx[1]
                    stencil[tuple(stencil_idx)] += coeff
            if aux_features is not None and len(aux_features) > 0:
                for i, features in enumerate(aux_features):
                    stencil_idx = [u_var_idx] + [(i - j + 1) // 2 for i, j in zip(imax, imin)]
                    stencil_idx.append(n_vars + aux_feature_count)
                    aux_feature_count += 1
                    del stencil_idx[1]
                    stencil[tuple(stencil_idx)] += features['c']
        self.backend.stencil = stencil


class EmitISLInitializationPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, has_conv=True):
        super().__init__("emit_isl_initialization",
                         "Emits code for initializing the ISL context (i.e., before the loop starts).",
                         backend, enabled)
        self.has_conv = has_conv

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.isl_code
        code.emit("@torch.jit.script")
        var_args = ', '.join([f"_{str(arg)}: torch.Tensor" for arg in self.backend.ir['pdes'].free_symbols])
        code.emit(f"def _isl(_ics: torch.Tensor, _N: torch.Tensor, {var_args}) -> torch.Tensor:")
        code.increase_indent()
        code.emit(f"__feature_maps_shape = [{len(self.backend.ir['order'])}] + [int(i) for i in _N[1:]]")
        code.emit(f"__feature_maps = torch.zeros(__feature_maps_shape, dtype={self.backend.tensor_type})")
        if self.has_conv:
            code.emit(f"__conv_kernel = torch.tensor({self.backend.stencil.tolist()}, dtype={self.backend.tensor_type})")
            code.emit("__kernel_perm = [0] + list(range(len(_N), 0, -1))")
            # Permute the stencil here too, so that it is in the correct format to reference its shape
            self.backend.stencil = np.transpose(self.backend.stencil,
                                                [0] + list(range(len(self.backend.stencil.shape) - 1, 0, -1)))
            code.emit("__conv_kernel = torch.permute(__conv_kernel, __kernel_perm)")
        code.emit("for _i in range(int(_ics.size(0))):")
        code.increase_indent()
        code.emit(f"__feature_maps[_i] = _ics[_i]")
        code.decrease_indent()


class EmitISLCodePhase(BackEndPhase):
    def __init__(self, backend, enabled=True, has_conv=True):
        super().__init__("emit_isl_code", "Emits code for the ISL loop.", backend, enabled)
        self.has_conv = has_conv

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.isl_code
        code.emit("for _i in range(int(_N[0])):")
        code.increase_indent()
        for j, unk in enumerate(self.backend.ir['pdes'].unknowns):
            code.emit(f"_{unk.name} = __feature_maps[{j}, ...]")
        for i, ft in enumerate(self.backend.feature_funcs):
            code.emit(f"__feature_maps[{i}, ...] = __feature_{i}({', '.join([f'_{str(arg)}' for arg in ft.args])})")
        if self.has_conv:
            code.emit(
                f"_tmp = F.conv{len(self.backend.ir['pdes'].free_symbols) - 1}d"
                f"(__feature_maps[None, ...], __conv_kernel)")


class EmitBoundaryConditionsPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, has_conv=True):
        super().__init__("emit_boundary_conditions", "Emits code for the boundary conditions.", backend, enabled)
        self.has_conv = has_conv

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.isl_code
        if self.has_conv:
            for j in range(len(self.backend.ir['pdes'].unknowns)):
                code.emit(f"_tmp_{j} = _tmp[0, {j}, ...]")
                padding_shape = [(self.backend.stencil.shape[i] // 2, self.backend.stencil.shape[i] // 2) for i in
                                 range(2, len(self.backend.stencil.shape))]
                padding_shape = list(itertools.chain(*padding_shape))
                code.emit(f"_tmp_{j} = F.pad(_tmp_{j}, {padding_shape}, value=0.)")
                code.emit(f"__feature_maps[{j}, ...] = _tmp_{j}")
        else:
            # Add together the different aux features for each variable and assign them to the feature maps
            for unk in self.backend.ir['pdes'].unknowns:
                # Collect the features for the unknown
                unk_index = self.backend.ir['pdes'].unknowns.index(unk)
                max_order_in_time = max([d.order[0] for d in self.backend.ir['disc'] if d.base_func.name == unk.name])
                feat_indices = [(j, c) for j, (u, c, expr) in enumerate(self.backend.ir['order']) if u == unk.name]
                feat_indices = feat_indices[max_order_in_time:]
                code.emit(
                    f"__feature_maps[{unk_index}, ...] = {' + '.join([f'({c} * __feature_maps[{i}, ...])' for i, c in feat_indices])}")
        code.decrease_indent()
        code.emit(f"return __feature_maps[:{len(self.backend.ir['pdes'].unknowns)}, ...]")
        code.decrease_indent()


class EmitMainFunctionPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, ics='external', plot=False, sol_hypercube=None):
        super().__init__("emit_main_function", "Emits code for the main function.", backend, enabled)
        self.ics = ics
        self.plot = plot
        self.sol_hypercube = sol_hypercube

    def __call__(self):
        if not self.enabled:
            return
        code = self.backend.main_func_code
        code.emit("def main():")
        code.increase_indent()
        if self.sol_hypercube is not None:
            code.emit(f"_N = torch.tensor({list(self.sol_hypercube)}, dtype=torch.int32)")
        else:
            code.emit(
                f"_N = torch.tensor([{', '.join([str(65)] * len(self.backend.ir['pdes'].free_symbols))}], dtype=torch.int32)")
        code.emit(f"_dt = 1 / (_N - 1)")
        for i, v in enumerate(self.backend.ir['pdes'].free_symbols):
            code.emit(f"_{v} = torch.linspace(0, 1, int(_N[{i}]), dtype={self.backend.tensor_type})")
        code.emit(
            f"_ic = torch.zeros(({len(self.backend.ir['pdes'].unknowns)}, "
            f"{', '.join([f'_N[{i}]' for i in range(1, len(self.backend.ir['pdes'].free_symbols))])}), "
            f"dtype={self.backend.tensor_type})")
        if self.ics == 'external':
            for i, _ in enumerate(self.backend.ir['pdes'].unknowns):
                code.emit(f"_ic[{i}, ...] = torch.tensor(np.load(sys.argv[{i + 1}]), dtype={self.backend.tensor_type})")
        code.emit(f"result = _isl(_ic, _N, {', '.join([f'_{i}' for i in self.backend.ir['pdes'].free_symbols])})")
        if self.plot:
            dims = len(self.backend.ir['pdes'].free_symbols) - 1
            plot_type = len(self.backend.ir['pdes'].unknowns)
            if dims == 3:
                code.emit("\033[93mWarning:\033[0m 3D plots are not supported.")
            if plot_type == 1:
                if dims == 1:
                    code.emit(f"plt.plot(_{self.backend.ir['pdes'].free_symbols[1]}.numpy(), result[0, ...].numpy())")
                elif dims == 2:
                    code.emit(f"plt.imshow(result[0, ...].numpy())")
                    code.emit("plt.colorbar()")
            elif plot_type == 2:
                if dims == 1:
                    code.emit(f"plt.plot(_{self.backend.ir['pdes'].free_symbols[1]}.numpy(), result[0, ...].numpy())")
                    code.emit(f"plt.plot(_{self.backend.ir['pdes'].free_symbols[1]}.numpy(), result[1, ...].numpy())")
                elif dims == 2:
                    code.emit(
                        f"plt.quiver(_{self.backend.ir['pdes'].free_symbols[1]}.numpy(), _{self.backend.ir['pdes'].free_symbols[2]}.numpy(), "
                        f"result[0, ...].numpy(), result[1, ...].numpy())")
            code.emit("plt.show()")
        code.reset_indent()
        code.emit("\n")
        code.emit("main()")


class PytorchPDEBackEnd(object):
    def __init__(self, ir, **kwargs):
        self.ir = ir
        self.stencil = None
        self.tensor_type = self._get_acc(kwargs.get('num_acc', 'f64'))
        self.feature_funcs = []
        self.preamble_code = CodeEmitter()
        self.free_function_code = CodeEmitter()
        self.aux_feature_code = CodeEmitter()
        self.isl_code = CodeEmitter()
        self.main_func_code = CodeEmitter()
        self.phases = [
            EmitPreambleCodePhase(self, True, kwargs.get('plot', True), kwargs.get('ics', 'external')),
            EmitFreeFunctionsPhase(self, True, kwargs.get('funcs', {}), kwargs.get('torch_jit', True)),
            EmitAuxiliaryFeaturesPhase(self, True),
            CalculateStencilShapePhase(self, kwargs.get('use_conv', True)),
            CalculateStencilCoeffPhase(self, kwargs.get('use_conv', True)),
            EmitISLInitializationPhase(self, True, kwargs.get('use_conv', True)),
            EmitISLCodePhase(self, True, kwargs.get('use_conv', True)),
            EmitBoundaryConditionsPhase(self, True, kwargs.get('use_conv', True)),
            EmitMainFunctionPhase(self, True, kwargs.get('ics', 'external'), kwargs.get('plot', True),
                                  kwargs.get('sol_hypercube', None))
        ]

    def execute(self):
        [phase() for phase in self.phases]
        return (f"{self.preamble_code}\n\n{self.free_function_code}\n"
                f"{self.aux_feature_code}\n{self.isl_code}\n{self.main_func_code}")

    def _get_acc(self, acc_str):
        match acc_str:
            case 'f16':
                return "torch.float16"
            case 'f32':
                return "torch.float32"
            case 'f64':
                return "torch.float64"
            case 'b16':
                return "torch.bfloat16"
            case 'e4m3':
                return "torch.float8_e4m3fn"
            case 'e5m2':
                return "torch.float8_e5m2"
            case _:
                raise ValueError(f"Unknown floating-point accuracy {acc_str} provided to 'pde_compile'."
                                 f" Supported types are: [f16, f32, f64, b16, e4m3, e5m2].")



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
