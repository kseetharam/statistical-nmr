from torch import nn, Tensor


class Volterra2D(nn.Module):
    """A Volterra 2D convolution layer. This is the second-order kernel of the series:

    The equation for the area of a circle is given by:

    # TODO: add the equation

    This layer performs a 2D convolution on input data, allowing for customizable
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
    def __init__(self, in_channels, kernel_size):
        super(Volterra2D, self).__init__()
        self.kernel_size = kernel_size
        
        # First convolution to generate a T x T output
        # self.conv1 = nn.Conv2d(in_channels, in_channels, (kernel_size, kernel_size), padding=(kernel_size // 2, kernel_size // 2))
        self.conv = nn.Conv2d(in_channels, in_channels, (kernel_size, kernel_size), padding=(kernel_size-1, kernel_size//2), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Input x should be of shape (batch_size, in_channels, T)
        # Reshape x to (batch_size, in_channels, T, 1) for 2D convolution
        x = x.unsqueeze(1)  # Shape: (batch_size, in_channels, T, 1)

        # First convolution to get a T x T output
        out1 = self.conv(x)  # Shape: (batch_size, in_channels, T, T)

        # Reshape out1 to be used as kernels for the second convolution
        # Shape: (batch_size, in_channels, T, T) -> (batch_size * in_channels, kernel_size, kernel_size)
        # out1_reshaped = out1.view(-1, self.kernel_size, self.kernel_size)
        out1_reshaped = out1.transpose(1, 2)

        # Perform convolution using the reshaped output as kernels
        # Need to create an input for the conv2d where each kernel is applied to x
        if len(x.shape) == 4:  # batched:
            batch_size, _, T, _ = x.shape
        elif len(x.shape) == 3:  # unbatched:
            _, T, _ = x.shape

        # x_repeated = x.repeat(1, 1, 1, self.kernel_size)  # Shape: (batch_size, in_channels, T, T)

        # Perform the 2D convolution with reshaped kernels
        output = nn.functional.conv2d(input=x, weight=out1_reshaped.unsqueeze(0), padding=(self.kernel_size-1, self.kernel_size//2))

        # Get the diagonal elements of the resulting output
        diagonal = output.diagonal(dim1=-2, dim2=-1)  # Shape: (batch_size, in_channels, T)

        return diagonal


if __name__ == "__main__":

    import torch

    batch_size = 8
    in_channels = 1
    kernel_size = 11
    T = 11

    # Create random input data
    input_data = torch.randn(in_channels, T)

    # Initialize and apply the custom Conv2D layer
    volterra_2d = Volterra2D(in_channels, kernel_size)
    output_data = volterra_2d(input_data)

    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
