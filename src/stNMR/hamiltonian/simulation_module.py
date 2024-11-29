from __future__ import annotations
from typing import Union

import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_exp

from stNMR.hamiltonian.utils import create_params_from_csv


class NMRModelTorch(nn.Module):
    def __init__(
            self, ux, uy, n_samp, h_mat,
            n_spins: int = 2, dt=1e-5,
            B0: float = 80.0, t2: float=1.0,
            phase: float=0.0,
            SW: float=1000, AQ=None, 
            control: bool=True, device=None
    ):
        super(NMRModelTorch, self).__init__()

        # Register ux and uy as non-trainable buffers
        if ux is not None:
            self.register_buffer('ux', torch.tensor(ux, dtype=torch.float32))
        if uy is not None:
            self.register_buffer('uy', torch.tensor(uy, dtype=torch.float32))

        self.control = control

        self.h_mat = h_mat

        # Sampling parameters
        self.n_samp = n_samp
        self.dt = dt

        # Spin system
        self.n_spins = n_spins
        dim = 2 ** self.n_spins
        self.B0 = B0

        self.t2 = t2
        self.phase = phase
        self.SW = SW
        self.AQ = AQ if AQ is None else self.n_samp/self.SW

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
        self.Ix = torch.zeros((dim, dim, self.n_spins), dtype=torch.cfloat)
        self.Iy = torch.zeros((dim, dim, self.n_spins), dtype=torch.cfloat)
        self.Iz = torch.zeros((dim, dim, self.n_spins), dtype=torch.cfloat)

        # Construct spin operator matrices
        for i in range(self.n_spins):
            left_identity = torch.eye(2**i) if i > 0 else torch.tensor(1)
            right_identity = torch.eye(2**(self.n_spins - i - 1)) if (self.n_spins - i - 1) > 0 else torch.tensor(1)
            self.Ix[:, :, i] = torch.kron(torch.kron(left_identity, self.sx), right_identity)
            self.Iy[:, :, i] = torch.kron(torch.kron(left_identity, self.sy), right_identity)
            self.Iz[:, :, i] = torch.kron(torch.kron(left_identity, self.sz), right_identity)

        # Summed spin operators
        self.IHx: torch.Tensor = torch.sum(self.Ix, axis=2)
        self.IHy: torch.Tensor = torch.sum(self.Iy, axis=2)
        self.IHz: torch.Tensor = torch.sum(self.Iz, axis=2)

        # Operator for measuring Mxy
        self.Op = self.IHx + 1j * self.IHy  # this is S_+ total

    def calc_hamiltonian(self, h_mat, B0, Ix, Iy, Iz, nspins):
        dim = 2**nspins
        H0 = torch.zeros((dim, dim), dtype=torch.cfloat)
        for i in range(nspins):
            H0 += 2 * torch.pi * B0 * h_mat[i, i] * Iz[:, :, i]
            for j in range(i + 1, nspins):
                H0 += 2 * torch.pi * h_mat[i, j] * (
                    torch.matmul(Ix[:, :, i], Ix[:, :, j])
                    + torch.matmul(Iy[:, :, i], Iy[:, :, j])
                    + torch.matmul(Iz[:, :, i], Iz[:, :, j])
                )
        return H0

    def calc_spec(self, OP: torch.Tensor, H0: torch.Tensor, rho: torch.Tensor):

        P: torch.Tensor = matrix_exp(-1j * H0 * self.dt)
        idx = torch.arange(self.n_samp)
        apod = torch.exp(-self.dt / self.t2 * idx)
        FID = torch.zeros(self.n_samp, dtype=torch.cfloat)
        for i in range(self.n_samp):
            FID[i] = torch.trace(OP @ rho) * apod[i]
            rho = P @ rho @ P.T.conj()
        spec = torch.fft.fftshift(torch.fft.fft(FID, 2 * self.n_samp))
        time_series = torch.linspace(0, self.AQ, self.n_samp)
        freq_series = torch.linspace(-self.SW / 2, self.SW / 2, 2 * self.n_samp)
        FTspec = torch.exp(torch.tensor(1j * self.phase)) * spec
        return FID, time_series, FTspec, freq_series

    # TODO: replace these with diffrax ODE solver instead of matrix exp
    # allows for a differentiable ODE solver => makes things more efficient for us!
    def forward(self):
        # Construct the static Hamiltonian H0
        # NOTE: change this to a nested for loop for(i from 1 to N) {for (j from i+1 to N)}
        """
        hiList = [[paramMat[i, i], i] for i in np.arange(N)]  # extracts hi from parameter matrix (puts in form for QuSpin)
        JijList = [[2 * paramMat[i, j], i, j] for i in np.arange(N) for j in np.arange(N) if (i != j) and (i < j) if not np.isclose(paramMat[i, j], 0)] 
            # extracts Jij from parameter matrix (puts in form for QuSpin); this list combines the Jij and Jji terms (Hermitian conjugates) into a single term
        """
        # H0 = 2 * np.pi * (self.v1 * self.Iz1 + self.v2 * self.Iz2 + self.J * (self.Ix1 @ self.Ix2 + self.Iy1 @ self.Iy2 + self.Iz1 @ self.Iz2))
        H0 = self.calc_hamiltonian(self.h_mat, self.B0, self.Ix, self.Iy, self.Iz, self.n_spins)

        # Initialize rho and Mxy
        rho = self.IHz  # initial state
        U90y: torch.Tensor = matrix_exp(-1j * torch.pi / 2 * self.IHy)
        rho = U90y @ rho @ U90y.T.conj()

        FID, time_series, FTspec, freq_series = self.calc_spec(self.Op, H0, rho)
        return FID, time_series, FTspec, freq_series

    @classmethod
    def from_csv(cls, file_path: Union[str, Path], device=None) -> NMRModelTorch:

        params = create_params_from_csv(file_path)
        nmr_module = cls(
            ux=None, uy=None, n_samp=params["N_TD"], h_mat=params["Hmat"],
            n_spins=4, dt=(params["AQ"] / params["N_TD"]),
            B0=params["LarmorFreq"], t2=params["T2"],
            phase=params["phase"],
            SW=params["SW"], AQ=params["AQ"], 
            control=False, device=device
        )
        return nmr_module


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
    class TargetNMRModel(NMRModelTorch):
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
    fitted_model = NMRModelTorch(ux, uy, Nsamp, dt, device=device).to(device)

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
