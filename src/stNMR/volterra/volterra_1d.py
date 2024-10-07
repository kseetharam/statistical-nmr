from torch import nn, Tensor


class Volterra1D(nn.Module):
    """A Volterra 1D convolution layer.

    This layer performs a 1D convolution on input data, allowing for customizable
    parameters such as the number of input channels, output channels, kernel size,
    stride, and padding.

    Attributes:
        conv (nn.Conv1d): The underlying 1D convolution layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Amount of zero-padding added to both sides of the input. Defaults to 0.

    Notes:
        This class is functionally equivalent to PyTorch's `nn.Conv1d` and is 
        primarily intended for demonstration and customization purposes.
    """
    def __init__(self, in_channels, kernel_size, stride=1, padding=0) -> None:
        super(Volterra1D, self).__init__()

        # Calculate padding to maintain the same input and output length
        if padding is None:
            # Default padding for same length: (kernel_size - 1) // 2
            padding = (kernel_size - 1) // 2
        
        # Using PyTorch's built-in Conv1d layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:

        # Make sure the input is 1D
        assert len(x.squeeze().shape) == 1, f"expected input to Volterra1D layer to be 1D. Recieved: {x.shape}"

        # Take the covolution
        return self.conv(x)


if __name__ == "__main__":

    import torch

    batch_size = 8
    in_channels = 1
    kernel_size = 11
    T = 11

    # Create random input data
    input_data = torch.randn(in_channels, T)

    # Initialize and apply the custom Conv2D layer
    volterra_1d = Volterra1D(in_channels, kernel_size)
    output_data = volterra_1d(input_data)

    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
