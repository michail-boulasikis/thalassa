import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.jit.script
def __ic_u(x: torch.Tensor) -> torch.Tensor:
    # X, Y = torch.meshgrid(x, y)
    # return torch.sin(2 * 3.141592653589793 * X) * torch.sin(2 * 3.141592653589793 * Y)
    return torch.sin(3.141592653589793 * x)


@torch.jit.script
def __ic_v(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    X, Y = torch.meshgrid(x, y)
    return -torch.sin(2 * 3.141592653589793 * X) * torch.sin(2 * 3.141592653589793 * Y)


@torch.jit.script
def __feature_0(u: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    tmp = u
    return tmp


@torch.jit.script
def __feature_1(u: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    tmp_0_0 = u
    tmp_0_1_0 = F.pad(u, [0, 1], value=0.)
    tmp_0_1_1 = tmp_0_1_0[1:]
    tmp_0 = tmp_0_0 * tmp_0_1_1
    tmp_1_0 = -1.0
    tmp_1_1 = u
    tmp_1_2_0 = F.pad(u, [1, 0], value=0.)
    tmp_1_2_1 = tmp_1_2_0[:-1]
    tmp_1 = tmp_1_0 * tmp_1_1 * tmp_1_2_1
    tmp = tmp_0 + tmp_1
    return tmp


@torch.jit.script
def _integrate(ic: torch.Tensor, N: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    __feature_maps_shape = [2] + [int(i) for i in N[1:]]
    __feature_maps = torch.zeros(__feature_maps_shape)
    __conv_kernel = torch.tensor([[[0.05, 0.0], [0.9, -0.1], [0.05, 0.0]]])
    __kernel_perm = [0] + list(range(len(N), 0, -1))
    __conv_kernel = torch.permute(__conv_kernel, __kernel_perm)
    for _i in range(int(ic.size(0))):
        __feature_maps[_i] = ic[_i]
    for _i in range(int(N[0])):
        __feature_maps[0, ...] = __feature_0(__feature_maps[0], t, x)
        __feature_maps[1, ...] = __feature_1(__feature_maps[0], t, x)
        _tmp = F.conv1d(__feature_maps[None, ...], __conv_kernel)
        _tmp_0 = _tmp[0, 0, ...]
        _tmp_0 = F.pad(_tmp_0, [1, 1], value=0.)
        __feature_maps[0, ...] = _tmp_0
    return __feature_maps[:1, ...]


def main():
    _N = torch.tensor([41, 41], dtype=torch.int32)
    _dt = 1 / (_N - 1)
    _t = torch.linspace(0, 1, int(_N[0]))
    _x = torch.linspace(0, 1, int(_N[1]))

    _ic = torch.zeros((1, _N[1]))
    _ic[0, :] = __ic_u(_x)
    result = _integrate(_ic, _N, _t, _x)
    plt.plot(_x.numpy(), result[0, ...].numpy())
    plt.show()


main()

