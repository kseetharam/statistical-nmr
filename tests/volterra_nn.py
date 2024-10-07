import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the learnable kernel as a PyTorch module
class LearnableKernel(nn.Module):
    def __init__(self):
        super(LearnableKernel, self).__init__()
        self.a = nn.Parameter(torch.randn(1))  # Learnable parameter

    def forward(self, t, t1, t2):
        return self.a * torch.exp(-((t - t1) ** 2 + (t - t2) ** 2))

# Function to compute output
def compute_output(inputs, kernel, t_values):
    output = torch.zeros_like(t_values)
    
    for i, t in enumerate(t_values):
        for j, t1 in enumerate(t_values):
            for k, t2 in enumerate(t_values):
                output[i] += kernel(t, t1, t2) * inputs[j] * inputs[k]

    return output

# Generate synthetic data
t_values = torch.linspace(0, 10, 100)  # Time values
inputs = torch.sin(t_values)  # Example input signal
true_output = 0.5 * inputs + 0.1 * torch.sin(2 * t_values)  # Some true output for simulation

# Initialize the learnable kernel
kernel = LearnableKernel()
optimizer = torch.optim.Adam(kernel.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Compute the output
    output = compute_output(inputs, kernel, t_values)
    
    # Compute loss (mean squared error)
    loss = nn.MSELoss()(output, true_output)
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, Kernel Parameter a: {kernel.a.item()}')

# Final output
final_output = compute_output(inputs, kernel, t_values)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(t_values.numpy(), inputs.numpy(), label='Input Signal (sin)')
plt.plot(t_values.numpy(), true_output.numpy(), label='True Output', linestyle='dashed')
plt.plot(t_values.numpy(), final_output.detach().numpy(), label='Learned Output', alpha=0.7)
plt.title('Learnable Second-Order Kernel Output')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.legend()
plt.grid()
plt.show()
