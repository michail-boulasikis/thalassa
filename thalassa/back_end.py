import pprint
from abc import ABC, abstractmethod
from multipledispatch import dispatch
import numpy as np
import sympy
import itertools
import thalassa
import time


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
            return True, f"{func.name}"


class CodeEmitter(object):
    def __init__(self):
        self.code = ""
        self.ir = {}
        self.indent = 0
        self.symbol_stack = []
        self.symbol_id = 0

    def make_temp(self):
        name = f"t{self.symbol_id}"
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
        gen_code, name = _resolve_func_name(func)
        if not gen_code:
            res_name = self.make_temp()
            for arg in func.args:
                self.emit(arg)
            args = []
            for i in range(len(func.args)):
                args.append(f"{self.symbol_stack.pop()}")
            self.emit(f"{res_name} = {name}({', '.join(args)})")
        else:
            res_name = self.make_temp()
            self.emit(f"{res_name} = {func.name}{'[time_step]' if 'self.' + self.ir['pdes'].free_symbols[0].name in map(str, func.args) else ''}")

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
            self.emit(f"{res_name} = {unk.base.name}")
        else:
            indices = [int(i) for i in unk.indices]
            paddings = [(
                -i if i < 0 else 0,
                i if i > 0 else 0
            ) for i in indices[1:]]
            paddings = paddings[::-1]
            slices = [f"{j if j != 0 else ''}:{-i if i != 0 else ''}" for i, j in paddings[::-1]]
            slices = ', '.join(slices)
            paddings = list(itertools.chain(*paddings))
            self.emit(f"{res_name} = F.pad({unk.base.name}, {str(paddings)}, value=0.)[{slices}]")

    @dispatch((int, float, sympy.core.numbers.NegativeOne, sympy.core.numbers.One, sympy.core.numbers.Zero,
               sympy.core.numbers.Number, sympy.core.numbers.Pi, sympy.core.numbers.Exp1))
    def emit(self, num):
        res_name = self.make_temp()
        self.emit(f"{res_name} = {str(sympy.N(num, 20))}")

    @dispatch(sympy.Symbol)
    def emit(self, symbol):
        res_name = self.make_temp()
        self.emit(f"{res_name} = {symbol.name}{'[time_step]' if 'self.' + self.ir['pdes'].free_symbols[0].name == symbol.name else ''}")

    def emit_import(self, lib, lib_as):
        if lib_as is None:
            self.emit(f"import {lib}")
        else:
            self.emit(f"import {lib} as {lib_as}")

    def emit_feature(self, feat_name, unknown_vars, other_vars, feat_expr, has_time=False):
        self.emit(f"class {feat_name}(torch.nn.Module):")
        self.increase_indent()
        feat_var_init_list = [f'{var.name}: torch.Tensor' for var in other_vars]
        feat_var_forward_list = [f'{var}: torch.Tensor' for var in unknown_vars]
        if has_time:
            feat_var_forward_list.append('time_step: int')
        self.emit(f"def __init__(self, {', '.join(feat_var_init_list)}):")
        self.increase_indent()
        self.emit("super().__init__()")
        for var in other_vars:
            self.emit(f"self.{var.name} = {var.name}")
        self.decrease_indent()
        self.emit("")
        self.emit(f"def forward(self, {', '.join(feat_var_forward_list)}):")
        self.increase_indent()
        res_name = self.make_temp()
        # Substitute other_vars with self.other_vars
        feat_expr = feat_expr.subs({var: sympy.Function(f"self.{var.name}")(*var.args) for var in other_vars if
                                    isinstance(var, sympy.Function)})
        feat_expr = feat_expr.subs(
            {var: sympy.Symbol(f"self.{var}") for var in other_vars if isinstance(var, sympy.Symbol)})
        self.emit(feat_expr)
        self.emit(f"return {self.symbol_stack.pop()}")
        self.decrease_indent()
        self.emit("")
        self.decrease_indent()

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


def _get_acc(acc_str):
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


class ResolveImportsPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, output=None, ics='external', device='cpu'):
        super().__init__("resolve_imports", "Resolves import statements depending on input options.", backend, enabled)
        self.plot = output == 'plot'
        self.ics = ics
        self.device = device

    def __call__(self):
        if self.enabled:
            # Always needed
            code = self.backend.code
            code.emit_import("torch", None)
            code.emit_import("torch.nn.functional", "F")
            if self.device == 'dev':
                code.emit_import("torch_xla.core.xla_model", "xm")
                code.emit_import("torch_xla.distributed.xla_multiprocessing", "xmp")
            # Needed for loading initial conditions and any other external data
            if self.ics == 'external' or self.backend.ir['pdes'].free_functions:
                code.emit_import("numpy", "np")
                code.emit_import("sys", None)
            # Needed only for plotting
            if self.plot:
                code.emit_import("matplotlib.pyplot", "plt")
            if self.device == 'dev':
                code.emit("")
                code.emit("dev = xm.xla_device()")
            code.emit("")


class CodeGenAuxiliaryFeaturesPhase(BackEndPhase):
    def __init__(self, backend, enabled=True):
        super().__init__("codegen_auxiliary_features",
                         "Generates code that computes the auxiliary features in torch.nn.Module classes.", backend,
                         enabled)

    def __call__(self):
        if self.enabled:
            code = self.backend.code
            for feat_base, (idx, feat_expr) in self.backend.ir['features'].items():
                unknown_vars = list(set(map(lambda x: x.base.name, feat_expr.atoms(sympy.Indexed))))
                other_vars = list(filter(lambda x: x in self.backend.symbol_total_order,
                                         feat_expr.atoms(sympy.Function, sympy.Symbol)))
                unknown_vars = self.backend.order_symbols_by_total_order(unknown_vars)
                other_vars = self.backend.order_symbols_by_total_order(other_vars)
                has_time = any(list(map(lambda x: x == self.backend.ir['pdes'].free_symbols[0], other_vars)))
                self.backend.ir['features'][feat_base].append(unknown_vars)
                self.backend.ir['features'][feat_base].append(other_vars)
                self.backend.ir['features'][feat_base].append(has_time)
                feat_name = f"PDEFeature{idx}"
                code.emit_feature(feat_name, unknown_vars, other_vars, feat_expr, has_time=has_time)


class CodeGenStencilApplicationPhase(BackEndPhase):
    def __init__(self, backend, enabled=True):
        super().__init__("codegen_stencil_application",
                         "Generates code that computes one step of the stencil application that solves the PDEs.",
                         backend,
                         enabled)

    def __call__(self):
        if self.enabled:
            code = self.backend.code
            code.emit("class PDEISL(torch.nn.Module):")
            code.increase_indent()
            set_all_symbols = _get_all_used_symbols(self.backend, order_them=True)
            code.emit(f"def __init__(self, time_step: int, conv_kernel: torch.Tensor, {', '.join(set_all_symbols)}):")
            code.increase_indent()
            code.emit("super().__init__()")
            code.emit("self.time_step = time_step")
            dims = len(self.backend.ir['pdes'].free_symbols) - 1
            n_unknowns = len(self.backend.ir['pdes'].unknowns)
            code.emit(f"self.conv_layer = torch.nn.Conv{dims}d(")
            code.increase_indent()
            code.emit(
                f"{self.backend.n_features}, {n_unknowns}, kernel_size={str(self.backend.ir['stencil'].shape[2:])}, stride=1, padding=0")
            code.decrease_indent()
            code.emit(")")
            code.emit(f"self.conv_layer.weight = torch.nn.Parameter(conv_kernel, requires_grad=False)")
            biases = [(k, v) for k, v in self.backend.ir['biases'].items()]
            # Sort biases by the total order of symbols
            biases = sorted(biases, key=lambda x: self.backend.symbol_total_order_str.index(x[0].base.name))
            biases = list(map(lambda x: x[1], biases))
            code.emit(
                f"self.conv_layer.bias = torch.nn.Parameter(torch.tensor({str(biases)}, dtype={self.backend.tensor_type}).to(device={self.backend.device}), requires_grad=False)")
            # Generate code that instantiates feature layers
            for idx, (k, (_, _, unknown_vars, other_vars, _)) in enumerate(self.backend.ir['features'].items()):
                feat_name = f"PDEFeature{idx}"
                feat_args = ', '.join([f"{var.name}" for var in other_vars])
                code.emit(f"self.{k.name} = {feat_name}({feat_args})")
            code.decrease_indent()
            code.emit("")
            code.emit("def forward(self, ics: torch.Tensor):")
            code.increase_indent()
            # Step 1: Extract unknowns from ics tensor
            for unk in self.backend.ir['pdes'].unknowns:
                code.emit(f"{unk.name} = ics[{self.backend.ir['pdes'].unknowns.index(unk)}, ...]")
            # Step 2: Apply boundary conditions in the form of padding (zeros=dirichlet)
            for unk in self.backend.ir['pdes'].unknowns:
                # Pad all dimensions from both sides except the first one (batch size)
                paddings = [(1, 1) for _ in range(dims)]
                paddings = list(itertools.chain(*paddings))
                # TODO: Other types of boundary conditions
                code.emit(f"{unk.name} = F.pad({unk.name}, {str(paddings)}, value=0.)")
            # Step 3: Compute the features
            for idx, (k, (_, _, unknown_vars, other_vars, has_time)) in enumerate(self.backend.ir['features'].items()):
                feat_name = f"{k.name}"
                feat_args = ', '.join([f"{str(var)}" for var in unknown_vars])
                feat_args = (f"{'self.time_step' if has_time else ''}{',' if feat_args and has_time else ''}"
                             f"{feat_args}")
                feat_call = f"f_{feat_name} = self.{feat_name}({feat_args})"
                code.emit(f"{feat_call}")
            # Step 4: Concatenate the features and the unknowns
            code.emit("features = torch.stack([")
            code.increase_indent()
            for idx, (k, _) in enumerate(self.backend.ir['features'].items()):
                code.emit(f"f_{k.name},")
            code.decrease_indent()
            code.emit("], dim=0)")
            # Step 5: Apply the convolutional kernel
            code.emit(f"return self.conv_layer(features)")
            code.reset_indent()
            code.emit("")


def _get_all_used_symbols(backend, tt=True, order_them=False):
    set_all_symbols = set()
    for idx, (_, (_, _, _, other_vars, _)) in enumerate(backend.ir['features'].items()):
        feat_args = list(map(lambda x: f"{x.name}", other_vars))
        set_all_symbols = set_all_symbols.union(set(feat_args))
    set_all_symbols = list(set_all_symbols)
    if order_them:
        set_all_symbols = backend.order_symbols_by_total_order(set_all_symbols)
    if tt:
        set_all_symbols = list(map(lambda x: f"{x}: torch.Tensor", set_all_symbols))
    return set_all_symbols


class CodeGenMainNetworkPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, time_steps=20):
        super().__init__("codegen_main_network",
                         "Generates code that constructs the main network that solves the PDEs.", backend,
                         enabled)
        self.time_steps = time_steps

    def __call__(self):
        if self.enabled:
            code = self.backend.code
            code.emit("class PDENetwork(torch.nn.Module):")
            code.increase_indent()
            set_all_symbols = _get_all_used_symbols(self.backend, tt=False, order_them=True)
            set_all_symbols_args = ', '.join(f'{x}' for x in _get_all_used_symbols(self.backend, tt=True, order_them=True))
            code.emit(f"def __init__(self, {set_all_symbols_args}):")
            code.increase_indent()
            code.emit("super().__init__()")
            for x in set_all_symbols:
                code.emit(f"self.{x} = {x}")
            code.emit(f"self.time_steps = {self.time_steps}")
            code.emit(f"self.conv_kernel = torch.tensor({np.array2string(self.backend.ir['stencil'], separator=',', floatmode='unique')}, dtype={self.backend.tensor_type}).to(device={self.backend.device})")
            code.decrease_indent()
            code.emit("")
            code.emit("def forward(self, ics: torch.Tensor, i: int):")
            code.increase_indent()
            for i in range(self.time_steps):
                code.emit(
                    f"ics = PDEISL(self.time_steps * i + {i}, self.conv_kernel, {', '.join(f'self.{x}' for x in set_all_symbols)})(ics)")
            code.emit("return ics")
            code.reset_indent()
            code.emit("")


def _get_device(device):
    match device:
        case 'cpu':
            return '\'cpu\''
        case 'gpu':
            return '\'cuda:0\''
        case 'tpu':
            return 'dev'
        case _:
            raise ValueError(f"Unknown device '{device}' provided to 'pde_compile'. Supported devices are: [cpu, gpu, tpu]")

class CodeGenMainFunctionPhase(BackEndPhase):
    def __init__(self, backend, enabled=True, ics='external', outcome='external', hypercube=None, loop_iterations=20):
        super().__init__("codegen_main_function",
                         "Generates the code that executes when the generated script is run (the main function).",
                         backend,
                         enabled)
        self.outcome = outcome if outcome is not None else 'external'
        self.ics = ics if ics is not None else 'external'
        self.hypercube = hypercube if hypercube is not None else [100] * len(self.backend.ir['pdes'].free_symbols)
        self.loop_iterations = loop_iterations

    def __call__(self):
        if self.enabled:
            code = self.backend.code
            code.emit("")
            code.emit("if __name__ == '__main__':")
            code.increase_indent()
            code.emit('import time')
            for i, (sym, dim) in enumerate(zip(self.backend.ir['pdes'].free_symbols, self.hypercube)):
                code.emit(f"{sym.name} = torch.linspace(0, 1, {dim * (self.loop_iterations if i == 0 else 1)}, dtype={self.backend.tensor_type}).to(device={self.backend.device})")

            if len(self.hypercube) > 2:
                code.emit(f"{', '.join([x.name for x in self.backend.ir['pdes'].free_symbols[1:]])} = torch.meshgrid({', '.join([x.name for x in self.backend.ir['pdes'].free_symbols[1:]])}, indexing='ij')")
            i = 1
            if self.ics == 'external':
                for sym in self.backend.ir['pdes'].unknowns:
                    code.emit(f"{sym.name} = torch.tensor(np.load(sys.argv[{i}]), dtype={self.backend.tensor_type}).to(device={self.backend.device})")
                    i += 1
            for sym in self.backend.ir['pdes'].free_functions:
                if (funcs := self.backend.kwargs.get('funcs', None)) is not None:
                    if funcs[sym] == 'external':
                        code.emit(
                            f"{sym.name} = torch.tensor(np.load(sys.argv[{i}]), dtype={self.backend.tensor_type}).to(device={self.backend.device})")
                        i += 1
            ics = ', '.join([f"{x.name}" for x in self.backend.ir['pdes'].unknowns])
            code.emit(f"initial_conditions = torch.stack([{ics}])")
            set_all_symbols = _get_all_used_symbols(self.backend, tt=False, order_them=True)
            code.emit(f"network = PDENetwork({', '.join(set_all_symbols)}).to(device={self.backend.device})")
            code.emit(f"t_start = time.time()")
            code.emit(f"for i in range({self.loop_iterations}):")
            code.increase_indent()
            code.emit("initial_conditions = network(initial_conditions, i)")
            code.decrease_indent()
            code.emit(f"t_end = time.time()")
            code.emit("print(f'Time elapsed: {t_end - t_start} seconds')")
            call_to_cpu = ".cpu()" if self.backend.device != '\'cpu\'' else ""
            code.emit(f"initial_conditions = initial_conditions{call_to_cpu}.detach().numpy()")
            match self.outcome:
                case 'external':
                    code.emit(f"np.save(sys.argv[{i}], initial_conditions)")
                    # code.emit(f"np.savetxt(sys.argv[{i}] + '.txt', initial_conditions, delimiter='\\n')")
                case 'plot':
                    n_unknowns = len(self.backend.ir['pdes'].unknowns)
                    n_dims = len(self.backend.ir['pdes'].free_symbols) - 1
                    # Plot with n_unknown subplots, with plot if n_dims = 1, otherwise imshow
                    if n_dims == 1:
                        code.emit("for i in range(initial_conditions.shape[0]):")
                        code.increase_indent()
                        code.emit("plt.plot(initial_conditions[i])")
                        code.emit("plt.show()")
                    else:
                        code.emit("plt.imshow(initial_conditions)")
                        code.emit("plt.show()")
                case 'onnx':
                    raise NotImplementedError("ONNX target is not supported yet.")
                case _:
                    raise ValueError(f"Unknown outcome '{self.outcome}' provided to 'pde_compile'")
            code.decrease_indent()
            code.emit("")


class PytorchPDEBackEnd(object):
    def __init__(self, ir, **kwargs):
        self.ir = ir
        self.stencil = None
        self.tensor_type = _get_acc(kwargs.get('num_acc', 'f64'))
        self.device = _get_device(kwargs.get('device', 'cpu'))
        self.code = CodeEmitter()
        self.code.ir = self.ir
        self.symbol_total_order = [
            *self.ir['pdes'].unknowns,
            *self.ir['pdes'].free_functions,
            *self.ir['pdes'].free_symbols
        ]
        self.symbol_total_order_str = list(map(lambda x: x.name, self.symbol_total_order))
        self.n_features = len(self.ir['features'])
        self.kwargs = kwargs
        self.phases = [
            ResolveImportsPhase(self, True, kwargs.get('output', False), kwargs.get('ics', 'external'), self.device),
            CodeGenAuxiliaryFeaturesPhase(self, True),
            CodeGenStencilApplicationPhase(self, True),
            CodeGenMainNetworkPhase(self, True, kwargs.get('sol_hypercube', [20])[0]),
            CodeGenMainFunctionPhase(self, True, kwargs.get('ics', 'external'), kwargs.get('output', 'external'),
                                     kwargs.get('sol_hypercube', None), kwargs.get('loop_iterations', 20))
        ]

    def execute(self):
        timing = {}
        for phase in self.phases:
            t_start = time.time()
            phase()
            t_end = time.time()
            timing[phase.name] = t_end - t_start
        timing['total'] = sum(timing.values())
        if self.kwargs.get('timeit', False):
            import pandas as pd
            pd.DataFrame(timing, index=[0]).to_csv("generated-data/" + self.ir['pdes'].name + '.timing.backend.csv')
        return f"{self.code}"

    def order_symbols_by_total_order(self, symbols):
        if all(isinstance(s, str) for s in symbols):
            return sorted(symbols, key=lambda x: self.symbol_total_order_str.index(x))
        else:
            return sorted(symbols, key=lambda x: self.symbol_total_order.index(x))
