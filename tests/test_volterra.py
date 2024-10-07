import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.optimize import minimize

# Generate synthetic data
np.random.seed(0)
# t = np.linspace(0, 10, 100)
# x = np.sin(t) + 0.1 * np.random.normal(size=t.shape)  # Input signal
# y = np.cos(t) + 0.2 * np.random.normal(size=t.shape)  # Output signal

from stNMR.utils.io import read_mat

mat_path = "data/DFG_TwoSpinStochasticData.mat"
fids, x, y = read_mat(mat_path)
x = x.squeeze()
y = fids[0].squeeze()
x = x[::1000]
y = y[::1000]
t = np.linspace(0, 10, len(x))

def volterra_model(params, x, K):
    """ Generate output from the Volterra series model """
    y_hat = np.zeros_like(x)
    
    # First-order kernel (linear part)
    h1 = params[:len(params)//K]
    y_hat += convolve(x, h1, mode='same')
    
    return y_hat

def loss_function(params, x, y, K):
    """ Loss function to minimize """
    y_hat = volterra_model(params, x, K)
    return np.mean((y - y_hat) ** 2)  # Mean squared error

def fit_volterra(x, y, K):
    """ Fit Volterra series to data """
    # Initial guess for the kernel parameters (for simplicity, all zeros)
    initial_params = np.zeros(len(x)//K)
    
    # Minimize the loss function
    result = minimize(loss_function, initial_params, args=(x, y, K))
    
    return result.x

# Fit Volterra series
K = 2  # Up to first order
fitted_params = fit_volterra(x, y, K)

# Visualize the first-order kernel
h1 = fitted_params[:len(fitted_params)//K]

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t, x, label='Input Signal', color='blue')
plt.plot(t, y, label='Output Signal', color='orange')
plt.title('Input and Output Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h1, label='First-order Kernel', color='green')
plt.title('First-order Kernel')
plt.xlabel('Lag')
plt.ylabel('Kernel Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
