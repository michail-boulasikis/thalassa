import torch
import torch.nn as nn
import torch.nn.functional as F


class PDEFeature0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _u: torch.Tensor) -> torch.Tensor:
        _temp_4 = _u
        _temp_5 = F.pad(_u, [1, 0], value=0.)[:-1]
        _temp_3 = _temp_5 * _temp_4
        _temp_7 = -1.0
        _temp_8 = _u
        _temp_9 = F.pad(_u, [0, 1], value=0.)[1:]
        _temp_6 = _temp_9 * _temp_8 * _temp_7
        _temp_2 = _temp_6 + _temp_3
        return _temp_2


class PDEFeature1(nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, u_1: torch.Tensor) -> torch.Tensor:
        return u_1


class PDEStep(nn.Module):
    def __init__(self, conv_kernel: torch.Tensor):
        super().__init__()
        self.conv_kernel = conv_kernel

    def forward(self, ics):
        u0 = PDEFeature0()(ics[0, :])
        u1 = PDEFeature1()(ics[1, :])
        tmp = torch.stack((u0, u1))
        t_next = F.conv1d(tmp, self.conv_kernel)
        return F.pad(t_next, [1, 1], value=0.0)


class PDESolver(nn.Module):
    def __init__(self, conv_kernel: torch.Tensor):
        super().__init__()
        self.conv_kernel = conv_kernel

    def forward(self, ics: torch.Tensor) -> torch.Tensor:
        ics = PDEStep(self.conv_kernel)(ics)
        ics = PDEStep(self.conv_kernel)(ics)
        ics = PDEStep(self.conv_kernel)(ics)
        ics = PDEStep(self.conv_kernel)(ics)
        return ics


sz = torch.tensor([2, 100], dtype=torch.int32)
model = PDESolver(conv_kernel=torch.tensor([[[0.0, -1.0, 0.0], [0.2, 0.4, 0.2]], [[-0.2, 0.1, -0.2], [0.0, -1.0, 0.0]]], dtype=torch.float64))
tensor_x = torch.rand((2, 100), dtype=torch.float64)
res = model(tensor_x)
onnx_program = torch.onnx.dynamo_export(model, tensor_x)
onnx_program.save("thalassa_wave.onnx")
