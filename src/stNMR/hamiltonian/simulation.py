import torch
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

# Assuming Sx, Sy, Sz are defined similarly using torch
from stNMR.hamiltonian.utils import Sx, Sy, Sz

Sx, Sy, Sz = torch.from_numpy(Sx).float(), torch.from_numpy(Sy).float(), torch.from_numpy(Sz).float()

class NMRSimulationTorch:
    def __init__(self, v1, v2, J, Nsamp, dt, ux=None, uy=None):
        # Use torch.nn.Parameter for v1, v2, and J to make them optimizable
        self.v1 = torch.nn.Parameter(torch.tensor([v1], dtype=torch.double))
        self.v2 = torch.nn.Parameter(torch.tensor([v2], dtype=torch.double))
        self.J = torch.nn.Parameter(torch.tensor([J], dtype=torch.double))
        self.Nsamp = Nsamp
        self.dt = dt
        
        # Pauli matrix setup (converted to torch)
        self.id_matrix = torch.eye(2, dtype=torch.cdouble)
        self.IHx, self.IHy, self.IHz = self._create_spin_matrices()

        # Initialize control fields
        if ux is None and uy is None:
            self.ux, self.uy = self._generate_random_fields()
        else:
            self.ux, self.uy = ux.clone(), uy.clone()

        # Initialize rho and Op
        self.rho = self.IHz
        self.Op = self.IHx + 1j * self.IHy

    def _create_spin_matrices(self):
        Ix1 = torch.kron(Sx, self.id_matrix)
        Ix2 = torch.kron(self.id_matrix, Sx)
        IHx = Ix1 + Ix2

        Iy1 = torch.kron(Sy, self.id_matrix)
        Iy2 = torch.kron(self.id_matrix, Sy)
        IHy = Iy1 + Iy2

        Iz1 = torch.kron(Sz, self.id_matrix)
        Iz2 = torch.kron(self.id_matrix, Sz)
        IHz = Iz1 + Iz2

        return IHx, IHy, IHz

    def _generate_random_fields(self):
        ux = torch.tensor(2 * np.pi * 100 * np.random.randn(self.Nsamp), dtype=torch.double)
        uy = torch.tensor(2 * np.pi * 100 * np.random.randn(self.Nsamp), dtype=torch.double)

        # Center fields to zero mean
        ux -= ux.mean()
        uy -= uy.mean()

        return ux, uy

    def _create_hamiltonian(self, v1, v2, J):
        Ix1 = torch.kron(Sx, self.id_matrix)
        Ix2 = torch.kron(self.id_matrix, Sx)
        Iy1 = torch.kron(Sy, self.id_matrix)
        Iy2 = torch.kron(self.id_matrix, Sy)
        Iz1 = torch.kron(Sz, self.id_matrix)
        Iz2 = torch.kron(self.id_matrix, Sz)

        # Hamiltonian definition
        H0 = 2 * np.pi * (v1 * Iz1 + v2 * Iz2 + J * (Ix1 @ Ix2 + Iy1 @ Iy2 + Iz1 @ Iz2))
        return H0

    def calculate_time_evolution(self):
        Mxy = torch.zeros(self.Nsamp, dtype=torch.cdouble)

        H0 = self._create_hamiltonian(self.v1, self.v2, self.J)

        rho = self.rho
        for i in range(self.Nsamp):
            Hrf = self.ux[i] * self.IHx + self.uy[i] * self.IHy
            # Ensure torch is used for the matrix exponential and complex numbers
            # U = torch.from_numpy(torch.matrix_exp(-1j * self.dt * (H0 + Hrf))).to(torch.cdouble)
            U = torch.matrix_exp(-1j * self.dt * (H0 + Hrf)).to(torch.cdouble)
            rho = U @ rho @ U.conj().T
            Mxy[i] = 0.25 * torch.trace(self.Op.conj().T @ rho)

        return Mxy


# Define loss function based on target Mxy
def loss_fn(predicted_Mxy, target_Mxy):
    # return torch.mean(torch.abs(predicted_Mxy - target_Mxy))
    return torch.mean(((predicted_Mxy - target_Mxy).conj() * (predicted_Mxy - target_Mxy)).real)
    # return torch.mean((predicted_Mxy - target_Mxy).real**2) + torch.mean((predicted_Mxy - target_Mxy).imag**2)

# Create optimizer function
def optimize_nmr_simulation(v1_init, v2_init, J_init, Nsamp, dt, target_Mxy, learning_rate=1e-2, epochs=100):
    # Initialize the simulation object with initial guesses for v1, v2, and J
    # simulation = NMRSimulationTorch(v1_init, v2_init, J_init, Nsamp, dt)

    # Define optimizer
    optimizer = torch.optim.Adam([v1_init, v2_init, J_init], lr=learning_rate)

    # Training loop
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Calculate predicted Mxy
        predicted_Mxy = NMRSimulationTorch(v1_init, v2_init, J_init, Nsamp, dt).calculate_time_evolution()

        # Calculate loss
        loss = loss_fn(predicted_Mxy, target_Mxy)

        # Backpropagate the loss
        loss.backward()

        # Update the parameters (v1, v2, J)
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    return v1_init, v2_init, J_init

# Parameters
# v1_init = 1000.0 #+ np.random.normal(0, 1)  # Initial guess with noise
# v2_init = -500.0 #+ np.random.normal(0, 1)  # Initial guess with noise
# J_init = 30.0 #+ np.random.normal(0, 1)  # Initial guess with noise
# v1_init = np.random.normal(0, 10)  # Initial guess with noise
# v2_init = np.random.normal(0, 10)  # Initial guess with noise
# J_init = np.random.normal(0, 2)  # Initial guess with noise

v1_init = torch.nn.Parameter(torch.tensor([1000.0], dtype=torch.double))
v2_init = torch.nn.Parameter(torch.tensor([-500.0], dtype=torch.double))
J_init = torch.nn.Parameter(torch.tensor([30.0], dtype=torch.double))

Nsamp = int(1e3)
dt = 1e-5

# Assuming you have some target Mxy from previous simulations or experimental data
# target_Mxy = torch.tensor(np.random.randn(Nsamp), dtype=torch.cdouble)  # Placeholder for target data
target_Mxy = NMRSimulationTorch(v1_init, v2_init, J_init, Nsamp, dt).calculate_time_evolution()

# Optimize to find the best values for v1, v2, and J
v1_fit, v2_fit, J_fit = optimize_nmr_simulation(v1_init, v2_init, J_init, Nsamp, dt, target_Mxy, learning_rate=0.01)

print(f'Fitted v1: {v1_fit.item()}, v2: {v2_fit.item()}, J: {J_fit.item()}')
