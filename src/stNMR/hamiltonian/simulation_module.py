import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp
import numpy as np
from tqdm import tqdm


class NMRModel(nn.Module):
    def __init__(self, ux, uy, Nsamp, dt=1e-5, device=None):
        super(NMRModel, self).__init__()

        # Register ux and uy as non-trainable buffers
        self.register_buffer('ux', torch.tensor(ux, dtype=torch.float32))
        self.register_buffer('uy', torch.tensor(uy, dtype=torch.float32))

        # Sampling parameters
        self.Nsamp = Nsamp
        self.dt = dt

        # Trainable parameters
        self.v1 = nn.Parameter(torch.tensor(1000.0 + np.random.normal(0, 10)))  # Initial guess for v1
        self.v2 = nn.Parameter(torch.tensor(-500.0 +  + np.random.normal(0, 10)))  # Initial guess for v2
        self.J = nn.Parameter(torch.tensor(30.0 +  + np.random.normal(0, 1)))  # Initial guess for J

        # Define the Pauli matrices (scaled)
        self.sx = (torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat) / 2).to(device)
        self.sy = (torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat) / 2).to(device)
        self.sz = (torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat) / 2).to(device)
        self.id_matrix = torch.eye(2, dtype=torch.cfloat).to(device)

        # Spin operator matrices
        self.Ix1 = torch.kron(self.sx, self.id_matrix).to(device)
        self.Ix2 = torch.kron(self.id_matrix, self.sx).to(device)
        self.IHx = self.Ix1 + self.Ix2

        self.Iy1 = torch.kron(self.sy, self.id_matrix).to(device)
        self.Iy2 = torch.kron(self.id_matrix, self.sy).to(device)
        self.IHy = self.Iy1 + self.Iy2

        self.Iz1 = torch.kron(self.sz, self.id_matrix).to(device)
        self.Iz2 = torch.kron(self.id_matrix, self.sz).to(device)
        self.IHz = self.Iz1 + self.Iz2

        # Operator for measuring Mxy
        self.Op = self.IHx + 1j * self.IHy

    def forward(self):
        # Construct the static Hamiltonian H0
        H0 = 2 * np.pi * (self.v1 * self.Iz1 + self.v2 * self.Iz2 + self.J * (self.Ix1 @ self.Ix2 + self.Iy1 @ self.Iy2 + self.Iz1 @ self.Iz2))
        
        # Initialize rho and Mxy
        rho = self.IHz
        Mxy = torch.zeros(self.Nsamp, dtype=torch.cfloat)
        
        # Time evolution loop
        for i in range(self.Nsamp):
            # Time-dependent Hamiltonian with control fields ux, uy
            Hrf = self.ux[i] * self.IHx + self.uy[i] * self.IHy
            # Unitary evolution operator U
            U = matrix_exp(-1j * self.dt * (H0 + Hrf))
            # Update rho by evolving it
            rho = U @ rho @ U.conj().T
            # Compute Mxy as the trace of the operator Op
            Mxy[i] = 0.25 * torch.trace(self.Op.conj().T @ rho)
        
        return Mxy


if __name__ == "__main__":

    import torch
    import torch.optim as optim
    import numpy as np

    # Define parameters for the target model
    v1_target = 1000.0
    v2_target = -500.0
    J_target = 30.0
    Nsamp = int(1e4)  # Reduced sample size for faster optimization
    dt = 1e-5

    # Generate random control fields for ux and uy
    ux = 2 * np.pi * 100 * np.random.randn(Nsamp)
    ux = ux - np.mean(ux)
    uy = 2 * np.pi * 100 * np.random.randn(Nsamp)
    uy = uy - np.mean(uy)

    # Define the target model
    class TargetNMRModel(NMRModel):
        def __init__(self, ux, uy, Nsamp, dt=1e-5):
            super(TargetNMRModel, self).__init__(ux, uy, Nsamp, dt)
            # Set v1, v2, and J to target values and freeze them
            self.v1 = torch.nn.Parameter(torch.tensor(v1_target), requires_grad=False)
            self.v2 = torch.nn.Parameter(torch.tensor(v2_target), requires_grad=False)
            self.J = torch.nn.Parameter(torch.tensor(J_target), requires_grad=False)

    # Instantiate the target model and generate target Mxy data
    target_model = TargetNMRModel(ux, uy, Nsamp, dt)
    Mxy_target = target_model().detach()  # Run the target model

    # Step 2: Fit a new model to match the target data

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Instantiate a new model to be fitted
    fitted_model = NMRModel(ux, uy, Nsamp, dt, device=device).to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(fitted_model.parameters(), lr=0.001)
    loss_fn = lambda pred, target: torch.mean(
        ((pred - target) * (pred - target).conj()).real
    )

    # Training loop
    num_epochs = 200
    for epoch in tqdm(range(num_epochs)):

        Mxy_target = Mxy_target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: generate Mxy predictions
        Mxy_pred = fitted_model()

        # Compute the loss (difference between target and predicted Mxy)
        loss = loss_fn(Mxy_pred, Mxy_target)

        # Backpropagation: compute gradients
        loss.backward()

        # Update the parameters (v1, v2, J)
        optimizer.step()

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
            print(f"v1: {fitted_model.v1.item():.2f}, v2: {fitted_model.v2.item():.2f}, J: {fitted_model.J.item():.2f}")

    # Final fitted values
    print("\nFinal fitted values:")
    print(f"v1: {fitted_model.v1.item():.2f}")
    print(f"v2: {fitted_model.v2.item():.2f}")
    print(f"J: {fitted_model.J.item():.2f}")

    # Compare to target values
    print("\nTarget values:")
    print(f"v1: {v1_target}")
    print(f"v2: {v2_target}")
    print(f"J: {J_target}")
