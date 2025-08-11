from typing import Literal

import math
import time
import logging

import torch
import numpy as np
from jax import numpy as jnp

from stNMR.models.base import Model
from stNMR.optimizers import Optimizer, BFGS


class ExplicitFIDModel(Model):
    """
    Model for fitting FIDs through explicit exponentiation.
    """

    def __init__(
            self,
            n_spins: int,
            B0: float,
            sw: float,
            rho: np.ndarray,
            t2: float,
            n_td: int,
            apodize: bool,
            device: str = "cpu",
            backend: Literal["numpy", "torch", "jax"] = "numpy",
    ) -> None:

        self._logger = logging.getLogger("stNMR")
        self._backend = backend

        if self._backend == "numpy":
            from stNMR.numpy.hamiltonian import explicit_exponentiation
            from stNMR.numpy.nmr import hamiltonian_from_vectors, generate_spin_operators

            if device != "cpu":
                self._logger.warning("Numpy backend does not support GPU. Using 'cpu' as device.")

        elif self._backend == "torch":
            from stNMR.torch.hamiltonian import explicit_exponentiation
            from stNMR.torch.nmr import hamiltonian_from_vectors, generate_spin_operators
            self._device = torch.device(device)

        elif self._backend == "jax":
            from stNMR.jax.hamiltonian import explicit_exponentiation
            from stNMR.jax.nmr import hamiltonian_from_vectors, generate_spin_operators

        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

        self._explicit_exponentiation = explicit_exponentiation
        self._hamiltonian_from_vectors = hamiltonian_from_vectors

        self.n_spins = n_spins
        self.B0 = B0
        self.sw = sw

        if self._backend == "torch":
            self.rho = torch.tensor(rho, dtype=torch.complex64).to(self._device)
        elif self._backend == "jax":
            self.rho = jnp.array(rho, dtype=jnp.complex64)
        else:  # numpy
            self.rho = np.array(rho, dtype=np.complex64)
        
        self.t2 = t2
        self.n_td = n_td
        self.apodize = apodize

        self.Ix, self.Iy, self.Iz = generate_spin_operators(n_spins=self.n_spins)
        self.IHx, self.IHy, self.IHz = self.Ix[:, :, :].sum(2), self.Iy[:, :, :].sum(2), self.Iz[:, :, :].sum(2)
        self.OP = self.IHx + 1j * self.IHy

        if self._backend == "torch":
            self.Ix = self.Ix.to(self._device)
            self.Iy = self.Iy.to(self._device)
            self.Iz = self.Iz.to(self._device)
            self.IHx = self.IHx.to(self._device)
            self.IHy = self.IHy.to(self._device)
            self.IHz = self.IHz.to(self._device)
            self.OP = self.OP.to(self._device)

    def forward(self, parameters, ts, dt, n_td):
        """
        Computes the FID using explicit exponentiation of the Hamiltonian.
        
        :param hamiltonian: The Hamiltonian to be exponentiated.
        :return: The computed FID.
        """
        hamiltonian = self._hamiltonian_from_vectors(
            parameters[:self.n_spins], 
            parameters[self.n_spins:], 
            B0=self.B0, 
            Ix=self.Ix, 
            Iy=self.Iy, 
            Iz=self.Iz
        )

        return self._explicit_exponentiation(
            hamiltonian=hamiltonian,
            rho=self.rho,
            op=self.OP,
            ts=ts,
            dt=dt,
            n_td=n_td,
            t2=self.t2,
            apodize=self.apodize,
        )

    def step(self, x, gt_fid, ts, dt, length):
        """
        Performs a single optimization step.
        
        :param x: Current parameters.
        :param gt_fid: Ground truth FID.
        :param ts: Time points for the FID.
        :param dt: Time step size.
        :param length: Length of the FID to compute.
        :return: Loss value.
        """
        fid = self.forward(x, ts[:length], dt, length)

        if self._backend == "torch":
            loss = torch.mean(torch.abs(fid - gt_fid[:length]) ** 2)
        elif self._backend == "jax":
            loss = jnp.mean(jnp.abs(fid - gt_fid[:length]) ** 2)
        else:  # numpy
            loss = np.mean(np.abs(fid - gt_fid[:length]) ** 2)

        return loss

    def fit(self, x: np.ndarray, y: np.ndarray, bootstrap: bool = True, opt: Optimizer = BFGS(backend="numpy")):

        if self._backend == "torch":
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self._device)
            y = torch.tensor(y, dtype=torch.complex64).to(self._device)

        aq = self.n_td / self.sw
        ts = np.linspace(0, aq, self.n_td)
        dt = ts[1] - ts[0]
        gt_fid = self.forward(y, ts, dt, self.n_td)

        if self._backend == "torch":
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self._device)
            # gt_fid = torch.tensor(gt_fid, dtype=torch.complex64).to(self._device)

        if bootstrap:

            N = int(round(math.log2(len(gt_fid)), 1))

            for n in range(8):
                length = 2 ** (n + 1)
                if length > self.n_td:
                    length = self.n_td

                curr_ts = ts[:length]

                start_time = time.time()
                x, loss, loss_history, lr_history = opt.optimize(x, self.step, gt_fid[:length], curr_ts, dt, length)
                elapsed_time = time.time() - start_time

                self._logger.info(f"Iteration {n + 1}/{N}: Time taken: {elapsed_time:.2f} seconds, Length: {length}")

        else:
            x, loss, loss_history, lr_history = opt.optimize(x, self.step, gt_fid, ts, dt, self.n_td)

        self._logger.info("Optimization complete.")

        return x, loss, loss_history, lr_history
