from __future__ import annotations
from typing import Union

from pathlib import Path

from jax import numpy as jnp
from jax.scipy.linalg import expm
from diffrax import diffeqsolve, Dopri8, ODETerm, SaveAt, PIDController

from stNMR.hamiltonian.utils import create_params_from_csv


class NMRModelJax:
    def __init__(
            self, ux, uy, n_samp, h_mat,
            n_spins: int = 2, dt=1e-5,
            B0: float = 80.0, t2: float=1.0,
            phase: float=0.0,
            SW: float=1000, AQ=None, 
            control: bool=True, device=None
    ):
        self.ux = jnp.array(ux) if ux is not None else None
        self.uy = jnp.array(uy) if uy is not None else None
        self.control = control
        self.h_mat = h_mat
        self.n_samp = n_samp
        self.dt = dt
        self.n_spins = n_spins
        dim = 2 ** self.n_spins
        self.B0 = B0
        self.t2 = t2
        self.phase = phase
        self.SW = SW
        self.AQ = AQ if AQ is None else self.n_samp/self.SW

        # Define the Pauli matrices (scaled)
        self.sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64) / 2
        self.sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64) / 2
        self.sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64) / 2
        self.id_matrix = jnp.eye(2, dtype=jnp.complex64)

        # Spin operator matrices
        self.Ix = jnp.zeros((dim, dim, self.n_spins), dtype=jnp.complex64)
        self.Iy = jnp.zeros((dim, dim, self.n_spins), dtype=jnp.complex64)
        self.Iz = jnp.zeros((dim, dim, self.n_spins), dtype=jnp.complex64)

        # Construct spin operator matrices
        for i in range(self.n_spins):
            left_identity = jnp.eye(2**i) if i > 0 else jnp.array(1)
            right_identity = jnp.eye(2**(self.n_spins - i - 1)) if (self.n_spins - i - 1) > 0 else jnp.array(1)
            self.Ix = jnp.insert(self.Ix, i, jnp.kron(jnp.kron(left_identity, self.sx), right_identity), axis=2)
            self.Iy = jnp.insert(self.Iy, i, jnp.kron(jnp.kron(left_identity, self.sy), right_identity), axis=2)
            self.Iz = jnp.insert(self.Iz, i, jnp.kron(jnp.kron(left_identity, self.sz), right_identity), axis=2)

        # Summed spin operators
        self.IHx: jnp.ndarray = jnp.sum(self.Ix, axis=2)
        self.IHy: jnp.ndarray = jnp.sum(self.Iy, axis=2)
        self.IHz: jnp.ndarray = jnp.sum(self.Iz, axis=2)

        # Operator for measuring Mxy
        self.Op = self.IHx + 1j * self.IHy  # this is S_+ total

    def __call__(self):
        return self.forward()

    def calc_hamiltonian(self, h_mat, B0, Ix, Iy, Iz, nspins):
        dim = 2**nspins
        H0 = jnp.zeros((dim, dim), dtype=jnp.complex64)
        for i in range(nspins):
            H0 += 2 * jnp.pi * B0 * h_mat[i, i] * Iz[:, :, i]
            for j in range(i + 1, nspins):
                H0 += 2 * jnp.pi * h_mat[i, j] * (
                    jnp.dot(Ix[:, :, i], Ix[:, :, j])
                    + jnp.dot(Iy[:, :, i], Iy[:, :, j])
                    + jnp.dot(Iz[:, :, i], Iz[:, :, j])
                )
        return H0

    def calc_spec(self, OP: jnp.ndarray, H0: jnp.ndarray, rho: jnp.ndarray):
        
        # the vector field (?????????)
        def rhs(t, y, args):
            # Reshape y back to a matrix form
            rho = y.reshape(H0.shape)
            # Commutator: [H0, rho] (???????)
            commutator = -1j * (jnp.matmul(H0, rho) - jnp.matmul(rho, H0.T))
            return commutator.flatten()

        # Initial conditions (rho = IHz)
        rho_flat = self.IHx.flatten()

        # Set up the ODE solver (following examples...)
        term = ODETerm(rhs)
        solver = Dopri8()
        saveat = SaveAt(ts=jnp.linspace(0, self.AQ, self.n_samp))  # Save at each timestep
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

        # Solve the ODE
        print("ODE solver...")
        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.AQ,
            dt0=self.dt,
            y0=rho_flat,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            args=(),
            max_steps=4096*16
        )

        print("Extracting FID...")
        # Extract the FID from the solution
        FID = jnp.zeros(self.n_samp, dtype=jnp.complex64)
        for i, t in enumerate(sol.ts):
            rho_t = sol.ys[i].reshape(H0.shape)  # Reshape to density matrix
            FID = FID.at[i].set(jnp.trace(OP @ rho_t))  # update the FID vector

        # Apodization and Fourier transform to spectrum
        idx = jnp.arange(self.n_samp)
        apod = jnp.exp(-self.dt / self.t2 * idx)
        FID_apod = FID * apod
        spec = jnp.fft.fftshift(jnp.fft.fft(FID_apod, 2 * self.n_samp))
        time_series = jnp.linspace(0, self.AQ, self.n_samp)
        freq_series = jnp.linspace(-self.SW / 2, self.SW / 2, 2 * self.n_samp)

        # Fourier transformed spectrum
        FTspec = jnp.exp(1j * self.phase) * spec
        return FID, time_series, FTspec, freq_series

    def forward(self):
        # Construct the static Hamiltonian H0
        H0 = self.calc_hamiltonian(self.h_mat, self.B0, self.Ix, self.Iy, self.Iz, self.n_spins)

        # Initialize rho and Mxy
        rho = self.IHz  # initial state
        U90y = expm(-1j * jnp.pi / 2 * self.IHy)
        rho = jnp.dot(U90y, jnp.dot(rho, U90y.T.conj()))

        FID, time_series, FTspec, freq_series = self.calc_spec(self.Op, H0, rho)
        return FID, time_series, FTspec, freq_series

    @classmethod
    def from_csv(cls, file_path: Union[str, Path], device=None) -> NMRModelJax:
        params = create_params_from_csv(file_path)
        nmr_module = cls(
            ux=None, uy=None, n_samp=params["N_TD"], h_mat=params["Hmat"],
            n_spins=params["n_spins"], dt=(params["AQ"] / params["N_TD"]),
            B0=params["LarmorFreq"], t2=params["T2"],
            phase=params["phase"],
            SW=params["SW"], AQ=params["AQ"], 
            control=False, device=device
        )
        return nmr_module
