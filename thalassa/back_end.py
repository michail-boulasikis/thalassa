from abc import ABC, abstractmethod

import thalassa


class CodeEmitter(object):
    def __init__(self):
        self.code = ""
        self.indent = 0

    def emit(self, statement):
        self.code += f"{'    ' * self.indent}{statement}\n"

    def increase_indent(self):
        self.indent += 1

    def decrease_indent(self):
        self.indent = (self.indent - 1) if self.indent > 0 else 0

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
        super().__init__("emit_preamble_phase",
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


class PytorchPDEBackEnd(object):
    def __init__(self, ir, disc, **kwargs):
        self.ir = ir
        self.disc = disc
        self.preamble_code = CodeEmitter()
        self.free_function_code = CodeEmitter()
        self.aux_feature_code = CodeEmitter()
        self.isl_code = CodeEmitter()
        self.main_func_code = CodeEmitter()
        self.phases = [
            EmitPreambleCodePhase(self, True, kwargs.get('plot', False)),

        ]

    def execute(self):
        [phase() for phase in self.phases]
        return (f"{self.preamble_code}\n{self.free_function_code}\n"
                f"{self.aux_feature_code}\n{self.isl_code}\n{self.main_func_code}")


def pde_compile(pde, disc, **kwargs):
    front_end = thalassa.PDEFrontEnd(pde, disc, **kwargs)
    target_lang = kwargs.get('target', 'pytorch')
    match target_lang:
        case 'pytorch':
            return PytorchPDEBackEnd(front_end.execute(), disc, **kwargs).execute()
        case 'tensorflow':
            raise ValueError("Tensorflow target is not supported yet.")
        case 'mlir':
            raise ValueError("MLIR target is not supported yet.")
        case _:
            raise ValueError(f"Unknown target {target_lang} provided to 'pde_compile'")

