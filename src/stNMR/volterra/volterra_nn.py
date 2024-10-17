from typing import Union

import torch
from torch import nn

import numpy as np
from scipy import signal

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

    # TODO: this can only be implemented after we can do complex convlution in Torch!
    @classmethod
    def from_process(cls, y: Union[torch.Tensor, np.ndarray], x: Union[torch.Tensor, np.ndarray], order: int = 2):
        """
        Fits the model given an input-output pair by calculating the kernels
        """
        kernel_size = x.shape[0]
        net = cls(kernel_size, order=order)

        if isinstance(y, torch.Tensor):
            y_arr = y.detach().cpu().numpy()
        else:
            y_arr = y

        if isinstance(y, torch.Tensor):
            x_arr = y.detach().cpu().numpy()
        else:
            x_arr = x

        variance = y_arr.var()

        # 0-order term
        net.kernel_1.conv.bias = torch.from_numpy(y_arr).float()

        # 1-order term
        net.kernel_1.conv.weight = signal.convolve(y.detach().numpy())

        return net


if __name__ == "__main__":

    net = VolterraNetwork.from_process(torch.randn(10), torch.randn(10), 2)
    print(net)
