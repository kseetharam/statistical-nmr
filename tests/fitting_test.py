import torch
from torch import nn
from matplotlib import pyplot as plt

from stNMR.volterra.volterra_nn import VolterraNetwork
from stNMR.utils.runtime import seed_everything
from stNMR.utils.simulate import simulate


# Set seed
seed_everything(42)

# Simulate some data
x, y = simulate(t=51, dt=1)
x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

# Adding channels dimension
x = x.unsqueeze(0)
y = y.unsqueeze(0)

# Set training params
n_epochs = 100
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

net = VolterraNetwork(order=2, kernel_size=51)
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


fig, axes = plt.subplots(nrows=2, ncols=2)
axes = axes.flatten()

axes[0].plot(loss_histroy)
axes[0].set_title("Training loss")

print(net.get_kernel(order=0))

axes[1].plot(net.get_kernel(order=0))
axes[1].set_title("Kernel (order=0)")

axes[2].plot(net.get_kernel(order=1))
axes[2].set_title("Kernel (order=1)")

axes[3].imshow(net.get_kernel(order=2))
axes[3].set_title("Kernel (order=2)")

plt.show()
