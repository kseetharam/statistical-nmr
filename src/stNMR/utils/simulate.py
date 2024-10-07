import numpy as np
import matplotlib.pyplot as plt


# Define the first-order kernel (linear)
def linear_kernel(x):
    return 2 * x  # Example: simple doubling

# Define the second-order kernel (non-linear)
def second_order_kernel(x, y):
    return 0.5 * x * y  # Example: product of inputs

def simulate(t: int, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:

    # Parameters
    T = t  # Total time duration
    dt = dt  # Time step
    time = np.arange(0, T, dt)

    # Generate a random input signal (white noise)
    input_signal = np.sin(2 * np.pi * 0.1 * time) + 0.5 * np.random.normal(size=len(time))

    output_signal = np.zeros_like(input_signal)

    # First-order contribution
    for i in range(len(input_signal)):
        output_signal[i] += linear_kernel(input_signal[i])

    # Second-order contribution
    for i in range(1, len(input_signal)):
        for j in range(i):
            output_signal[i] += second_order_kernel(input_signal[i], input_signal[j])
    
    return input_signal, output_signal
