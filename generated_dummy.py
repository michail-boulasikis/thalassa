import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.jit.script
def __ic_u(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    X, Y = torch.meshgrid(x, y)
    return torch.sin(2 * 3.141592653589793 * X) * torch.sin(2 * 3.141592653589793 * Y)


@torch.jit.script
def __ic_v(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    X, Y = torch.meshgrid(x, y)
    return -torch.sin(2 * 3.141592653589793 * X) * torch.sin(2 * 3.141592653589793 * Y)


@torch.jit.script
def __feature_0(u: torch.Tensor, v: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tmp = u
    return tmp


@torch.jit.script
def __feature_1(u: torch.Tensor, v: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tmp = v
    return tmp


@torch.jit.script
def __feature_2(u: torch.Tensor, v: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tmp_0_0 = u
    tmp_0_1_0 = F.pad(u, [1, 0, 0, 0], value=0.)
    tmp_0_1_1 = tmp_0_1_0[0:, 0:-1]
    tmp_0 = tmp_0_0 * tmp_0_1_1
    tmp_1_0_0 = F.pad(u, [0, 0, 1, 0], value=0.)
    tmp_1_0_1 = tmp_1_0_0[0:-1, 0:]
    tmp_1_1 = v
    tmp_1 = tmp_1_0_1 * tmp_1_1
    tmp_2_0 = -1.0
    tmp_2_1 = u
    tmp_2_2_0 = F.pad(u, [0, 1, 0, 0], value=0.)
    tmp_2_2_1 = tmp_2_2_0[0:, 1:]
    tmp_2 = tmp_2_0 * tmp_2_1 * tmp_2_2_1
    tmp_3_0 = -1.0
    tmp_3_1_0 = F.pad(u, [0, 0, 0, 1], value=0.)
    tmp_3_1_1 = tmp_3_1_0[1:, 0:]
    tmp_3_2 = v
    tmp_3 = tmp_3_0 * tmp_3_1_1 * tmp_3_2
    tmp = tmp_0 + tmp_1 + tmp_2 + tmp_3
    return tmp


@torch.jit.script
def __feature_3(u: torch.Tensor, v: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tmp_0_0 = u
    tmp_0_1_0 = F.pad(v, [1, 0, 0, 0], value=0.)
    tmp_0_1_1 = tmp_0_1_0[0:, 0:-1]
    tmp_0 = tmp_0_0 * tmp_0_1_1
    tmp_1_0 = v
    tmp_1_1_0 = F.pad(v, [0, 0, 1, 0], value=0.)
    tmp_1_1_1 = tmp_1_1_0[0:-1, 0:]
    tmp_1 = tmp_1_0 * tmp_1_1_1
    tmp_2_0 = -1.0
    tmp_2_1 = u
    tmp_2_2_0 = F.pad(v, [0, 1, 0, 0], value=0.)
    tmp_2_2_1 = tmp_2_2_0[0:, 1:]
    tmp_2 = tmp_2_0 * tmp_2_1 * tmp_2_2_1
    tmp_3_0 = -1.0
    tmp_3_1 = v
    tmp_3_2_0 = F.pad(v, [0, 0, 0, 1], value=0.)
    tmp_3_2_1 = tmp_3_2_0[1:, 0:]
    tmp_3 = tmp_3_0 * tmp_3_1 * tmp_3_2_1
    tmp = tmp_0 + tmp_1 + tmp_2 + tmp_3
    return tmp


@torch.jit.script
def _integrate(ic: torch.Tensor, N: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    __feature_maps_shape = [4] + [int(i) for i in N[1:]]
    __feature_maps = torch.zeros(__feature_maps_shape)
    __conv_kernel = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                                   [[0.1, 0.0, 0.0, 0.0], [0.6, 0.0, 0.1, 0.0], [0.1, 0.0, 0.0, 0.0]],
                                   [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                                   [[0.0, 0.1, 0.0, 0.0], [0.0, 0.6, 0.0, 0.1], [0.0, 0.1, 0.0, 0.0]],
                                   [[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]])
    __kernel_perm = [0] + list(range(len(N), 0, -1))
    __conv_kernel = torch.permute(__conv_kernel, __kernel_perm)
    for _i in range(int(ic.size(0))):
        __feature_maps[_i] = ic[_i]
    for _i in range(int(N[0])):
        __feature_maps[0, ...] = __feature_0(__feature_maps[0], __feature_maps[1], t, x, y)
        __feature_maps[1, ...] = __feature_1(__feature_maps[0], __feature_maps[1], t, x, y)
        __feature_maps[2, ...] = __feature_2(__feature_maps[0], __feature_maps[1], t, x, y)
        __feature_maps[3, ...] = __feature_3(__feature_maps[0], __feature_maps[1], t, x, y)
        _tmp = F.conv2d(__feature_maps[None, ...], __conv_kernel)
        _tmp_0 = _tmp[0, 0, ...]
        _tmp_0 = F.pad(_tmp_0, [1, 1, 1, 1], value=0.)
        __feature_maps[0, ...] = _tmp_0
        _tmp_1 = _tmp[0, 1, ...]
        _tmp_1 = F.pad(_tmp_1, [1, 1, 1, 1], value=0.)
        __feature_maps[1, ...] = _tmp_1
    return __feature_maps[:2, ...]


def main():
    _N = torch.tensor([701, 41, 41], dtype=torch.int32)
    _dt = 1 / (_N - 1)
    _t = torch.linspace(0, 1, int(_N[0]))
    _x = torch.linspace(0, 1, int(_N[1]))
    _y = torch.linspace(0, 1, int(_N[2]))

    _ic = torch.zeros((2, _N[1], _N[2]))
    _ic[0, :] = __ic_u(_x, _y)
    _ic[1, :] = __ic_v(_x, _y)
    result = _integrate(_ic, _N, _t, _x, _y)
    plt.quiver(_x.numpy(), _y.numpy(), result[0, ...].numpy(), result[1, ...].numpy())
    plt.show()


main()
