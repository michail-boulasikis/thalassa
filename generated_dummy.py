import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.jit.script
def __f(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x)


@torch.jit.script
def __auxiliary_feature_1(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _u_padded_0_1 = F.pad(u, (0, 1), value=0.)
    _u_padded_1_0 = F.pad(u, (1, 0), value=0.)
    _temp_0 = _u_padded_1_0[:-1] * __f(x)
    _temp_1 = _u_padded_0_1[1:] * __f(x)
    _temp_2 = _temp_1 - _temp_0
    return _temp_2


@torch.jit.script
def _integrate(ic: torch.Tensor, N: torch.Tensor, dt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    __feature_maps = torch.zeros((2, int(N[1])))
    __conv_kernel_0 = torch.tensor([[[0., 0.], [1.0, -0.25], [0., 0.]]])
    __conv_kernel_0 = torch.permute(__conv_kernel_0, (0, 2, 1))
    __feature_maps[0, :] = ic[0, :]
    for _i in range(int(N[0])):
        __feature_maps[1, :] = __auxiliary_feature_1(__feature_maps[0, :], x)
        _temp_0 = F.conv1d(__feature_maps[None, ...], __conv_kernel_0)
        _temp_1 = _temp_0[0, 0, :]
        _temp_2 = F.pad(_temp_1, (1, 1), value=0.)
        __feature_maps[0, :] = _temp_2
    return __feature_maps[0, :]


def main():
    _N = torch.tensor([101, 201], dtype=torch.int32)
    _dt = 1 / (_N - 1)
    _t = torch.linspace(0, 1, int(_N[0]))
    _x = torch.linspace(0, 1, int(_N[1]))

    _ic = torch.zeros((2, _N[1]))
    _ic[0, :] = __f(_x)
    result = _integrate(_ic, _N, _dt, _x)

    plt.plot(_x.numpy(), result.numpy())
    plt.show()


main()
