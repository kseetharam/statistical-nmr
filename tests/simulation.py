import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
T = 50  # Total time duration
dt = 0.1  # Time step
time = np.arange(0, T, dt)

# Generate a random input signal (white noise)
input_signal = np.sin(2 * np.pi * 0.1 * time) + 0.5 * np.random.normal(size=len(time))

# Define the first-order kernel (linear)
def linear_kernel(x):
    return 2 * x  # Example: simple doubling

# Define the second-order kernel (non-linear)
def second_order_kernel(x, y):
    return 0.5 * x * y  # Example: product of inputs

# Compute output using the Volterra series (up to second-order)
output_signal = np.zeros_like(input_signal)

# First-order contribution
for i in range(len(input_signal)):
    output_signal[i] += linear_kernel(input_signal[i])

# Second-order contribution
for i in range(1, len(input_signal)):
    for j in range(i):
        output_signal[i] += second_order_kernel(input_signal[i], input_signal[j])

# Plotting
plt.figure(figsize=(15, 10))

# Input signal
plt.subplot(4, 1, 1)
plt.plot(time, input_signal, label='Input Signal', color='blue')
plt.title('Input Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Output signal
plt.subplot(4, 1, 2)
plt.plot(time, output_signal, label='Output Signal', color='orange')
plt.title('Output Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Kernels visualization
plt.subplot(4, 1, 3)
x = np.linspace(-3, 3, 100)
linear_output = linear_kernel(x)
second_order_output = second_order_kernel(x[:, np.newaxis], x)  # outer product

plt.plot(x, linear_output, label='Linear Kernel (First Order)', color='green')
plt.title('Kernels Visualization')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.imshow(second_order_output)
plt.title('Kernels Visualization (2nd order)')

plt.tight_layout()
plt.show()
