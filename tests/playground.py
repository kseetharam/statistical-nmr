import torch
from torch import nn

x = torch.randn((1, 11,))

x = x.unsqueeze(1)
print(x.shape)

conv = nn.Conv2d(1, 1, (11, 11), padding=(11 - 1, 5))
print(conv.weight.shape)

out = conv(x)

print(out.detach().numpy().shape)
exit()

from matplotlib import pyplot as plt


fig, axes = plt.subplots(ncols=1, nrows=3)


axes[0].plot(x.squeeze().detach().numpy())
axes[1].imshow(conv.weight.squeeze().detach().numpy())
axes[2].imshow(out.squeeze().detach().numpy())

plt.show()