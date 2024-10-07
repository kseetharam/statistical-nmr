import torch
from torch import nn

import numpy as np
from matplotlib import pyplot as plt

from stNMR.volterra.volterra_nn import VolterraNetwork
from stNMR.utils.runtime import seed_everything
from stNMR.utils.simulate import simulate
from stNMR.utils.io import read_mat


# Set seed
seed_everything(42)

# Simulate some data
mat_path = "data/DFG_TwoSpinStochasticData.mat"
fids, ux, uy = read_mat(mat_path)

ux = ux[:-1]
y = fids[0, :-1]

# Throw away the imaginary part
ux = np.real(ux)
y = np.real(y)

# plt.plot(y)

# ux = ux[::10000][:-1]
ux = ux[:1001][::2]
y = y[::2]

plt.plot(y)
plt.show()
exit()

# Scale y
y = (y - y.min())/(y.max() - y.min())

# Adding channels dimension
x = torch.from_numpy(ux.T).float()
y = torch.from_numpy(y).float().unsqueeze(0)

# Set training params
n_epochs = 500
learning_rate = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

net = VolterraNetwork(order=2, kernel_size=x.size(1))
net.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

loss_histroy = []

for epoch in range(n_epochs):

    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()

    pred = net(x)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}/{n_epochs}. Loss: {loss.item()}")
    loss_histroy.append(loss.item())


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
axes = axes.flatten()

axes[0].plot(loss_histroy)
axes[0].set_title("Training loss")

print(net.get_kernel(order=0))

# axes[1].plot(net.get_kernel(order=0))
# axes[1].set_title("Kernel (order=0)")
axes[1].plot(pred.squeeze().detach().cpu().numpy())
axes[1].plot(y.squeeze().detach().cpu().numpy())
axes[1].set_title("Final prediction")

axes[2].plot(net.get_kernel(order=1))
axes[2].set_title("Kernel (order=1)")

axes[3].imshow(net.get_kernel(order=2))
axes[3].set_title("Kernel (order=2)")

plt.savefig("experiments/figures/plot_2.png", dpi=199)
