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
print(res)
print("\n\n")
onnx_program = torch.onnx.dynamo_export(model, tensor_x)
onnx_program.save("thalassa_wave.onnx")
