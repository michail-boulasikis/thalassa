import torch
import torch.nn as nn
import torch.nn.functional as F


class PDEFeature0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u


class PDEFeature1(nn.Module):
    def __init__(self): 
        super().__init__()
    
    def forward(self, u_1: torch.Tensor) -> torch.Tensor:
        return u_1


class PDEStepLayer(nn.Module):
    def __init__(self, kernel: torch.Tensor, input_shape: torch.Tensor, N: torch.Tensor, t: torch.Tensor, x: torch.Tensor):
        super().__init__()
        self.kernel = kernel
        self.input_shape = input_shape
        self.N = N
        self.t = t
        self.x = x

    def forward(self, ics: torch.Tensor) -> torch.Tensor:
        feature_maps_shape = [2] + [int(i) for i in self.N[1:]]
        feature_maps = torch.zeros(feature_maps_shape, dtype=torch.float64)
        #kernel_perm = [0] + list(range(len(self.N), 0, -1))
        #conv_kernel = torch.permute(self.kernel, kernel_perm)
        #feature_maps[0] = ics[0]
        #feature_maps[1] = ics[1]
        #tmp = F.conv1d(feature_maps[None, ...], conv_kernel)
        #tmp_0 = tmp[0, 0, ...]
        #tmp_0 = F.pad(tmp_0, [1, 1], value=0.)
        #feature_maps[0, ...] = tmp_0
        #tmp_1 = tmp[0, 1, ...]
        #tmp_1 = F.pad(tmp_1, [1, 1], value=0.)
        #feature_maps[1, ...] = tmp_1
        return feature_maps[:2, ...]


class PDESolver(nn.Module):
    def __init__(self, input_shape: torch.Tensor):
        super().__init__()
        self.input_shape = input_shape
        self.N = torch.tensor(self.input_shape, dtype=torch.int32)
        self.dx = 1 / (self.N - 1)
        self.t = torch.linspace(0, 1, int(self.N[0]), dtype=torch.float64)
        self.x = torch.linspace(0, 1, int(self.N[1]), dtype=torch.float64)

    def forward(self, ics: torch.Tensor) -> torch.Tensor:
        ics = PDEStepLayer(torch.tensor([[[0.0, 0.0], [1.0, 1e-05], [0.0, 0.0]], [[0.1, 0.0], [-0.2, 1.0], [0.1, 0.0]]], dtype=torch.float64), self.input_shape, self.N, self.t, self.x)(ics)
        ics = PDEStepLayer(torch.tensor([[[0.0, 0.0], [1.0, 1e-05], [0.0, 0.0]], [[0.1, 0.0], [-0.2, 1.0], [0.1, 0.0]]], dtype=torch.float64), self.input_shape, self.N, self.t, self.x)(ics)
        ics = PDEStepLayer(torch.tensor([[[0.0, 0.0], [1.0, 1e-05], [0.0, 0.0]], [[0.1, 0.0], [-0.2, 1.0], [0.1, 0.0]]], dtype=torch.float64), self.input_shape, self.N, self.t, self.x)(ics)
        ics = PDEStepLayer(torch.tensor([[[0.0, 0.0], [1.0, 1e-05], [0.0, 0.0]], [[0.1, 0.0], [-0.2, 1.0], [0.1, 0.0]]], dtype=torch.float64), self.input_shape, self.N, self.t, self.x)(ics)
        return ics


sz = torch.tensor([2, 100], dtype=torch.int32)
model = PDESolver(sz)
tensor_x = torch.rand((2, 100), dtype=torch.float64)
res = model(tensor_x)
print(res)
print("\n\n")
onnx_program = torch.onnx.dynamo_export(model, tensor_x)
onnx_program.save("thalassa_wave.onnx")
