import torch
from torch import nn

from stNMR.volterra.volterra_1d import Volterra1D
from stNMR.volterra.volterra_2d import Volterra2D


class VolterraNetwork(nn.Module):
    def __init__(self, kernel_size: int, order: int = 2) -> None:
        super(VolterraNetwork, self).__init__()

        assert order <= 2 and order >= 1, f"order must be an integer in range [1, 2]. Recieved: {order}"
        self.order = order

        layers = []

        self.kernel_1 = Volterra1D(in_channels=1, kernel_size=kernel_size, stride=1)
        layers.append(self.kernel_1)

        if order == 2:
            self.kernel_2 = Volterra2D(in_channels=1, kernel_size=kernel_size)
            layers.append(self.kernel_2)
        
        # self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.kernel_1(x)

        if self.order == 2:
            out += self.kernel_2(x)
        
        return out

    def get_kernel(self, order: int):

        assert order <= self.order and order >= 0, f"order must be in range [0, {self.order}]. Recieved: {order}"

        if order == 0:
            return self.kernel_1.conv.bias.detach().squeeze().cpu().numpy()

        elif order == 1:
            return self.kernel_1.conv.weight.detach().squeeze().cpu().numpy()

        elif order == 2:
            return self.kernel_2.conv.weight.detach().squeeze().cpu().numpy()
