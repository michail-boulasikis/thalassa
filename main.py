import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim

N = 101
T = 200
dx = 1 / (N - 1)
dt = 0.001

x = torch.linspace(0, 1, N)

cx = 10 * x * (1 - x)

u = torch.sin(2 * torch.pi * x)
conv_data = torch.zeros(3, N)
conv_kernel = torch.tensor([[0, 1, 0], [0, -dt / (2 * dx), 0], [0, dt / (2 * dx), 0]])
all_data = torch.zeros(T, N)
all_data[0] = u

for i in range(1, T):
    conv_data[0] = u
    conv_data[1] = cx * torch.nn.functional.pad(u[1:], (0, 1))
    conv_data[2] = cx * torch.nn.functional.pad(u[:-1], (1, 0))
    u = torch.nn.functional.pad(
        torch.nn.functional.conv1d(conv_data.view(1, 3, N), conv_kernel.view(1, 3, -1), padding="valid").view(-1),
        (1, 1))
    all_data[i] = u

# Animate the solution
fig, ax = plt.subplots()
line, = ax.plot(x, all_data[0])
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)


def update(i):
    line.set_ydata(all_data[i])
    return line,


ani = anim.FuncAnimation(fig, update, frames=T, interval=50, blit=True)
plt.show()
