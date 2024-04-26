import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Grid size, time steps, and grid spacing
N = 101
T = 200
# The minus one is because I use the number of intervals, not the number of points
# This is common in finite difference methods
dx = 1 / (N - 1)
# Time step, needs to be small to ensure stability (not needed if the method is implicit, but then we need to solve a
# linear system --- future work maybe!)
dt = 0.001

# Convective velocity, this is m(x) in c_t + m(x) c_x = 0
x = torch.linspace(0, 1, N)
cx = 10 * x * (1 - x)
# Initial condition
u = torch.sin(2 * torch.pi * x)

# I allocate some memory for the initial condition as well as the auxiliary features
conv_data = torch.zeros(3, N)

# Convolution kernel, utilizing the auxiliary features, note the zeros in the kernel. Typically, we would have
# a stencil of 3 non-zero points, but here we "roll" the stencil in the feature dimension by using auxiliary features.
# Despite the zeroes technically not being needed, the stencil is still 3 points wide.
# We need it so in order to respect the boundary conditions.
conv_kernel = torch.tensor([[0, 1, 0], [0, -dt / (2 * dx), 0], [0, dt / (2 * dx), 0]])

# I allocate memory for the solution, used for the animation
all_data = torch.zeros(T, N)
all_data[0] = u

for i in range(1, T):
    conv_data[0] = u
    # Computing the auxiliary features
    conv_data[1] = cx * torch.nn.functional.pad(u[1:], (0, 1))
    conv_data[2] = cx * torch.nn.functional.pad(u[:-1], (1, 0))
    # Computing the solution
    u = torch.nn.functional.pad(
        torch.nn.functional.conv1d(conv_data.view(1, 3, N), conv_kernel.view(1, 3, -1), padding="valid").view(-1),
        (1, 1))
    # Storing the solution
    all_data[i] = u

# -------------------- Animation --------------------
# Code essentially copy-pasted from the matplotlib documentation
fig, ax = plt.subplots()
line, = ax.plot(x, all_data[0])
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
def update(i):
    line.set_ydata(all_data[i])
    return line,
ani = anim.FuncAnimation(fig, update, frames=T, interval=50, blit=True)
plt.show()
