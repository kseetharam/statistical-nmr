import jax
import jax.nn as jnn
import jax.numpy as jnp

import diffrax
import equinox as eqx

from stNMR.jax.nmr import compute_fid, apodization
from stNMR.jax.hamiltonian.utils import recombine_complex, separate_complex


class Func(eqx.Module):

    mlp: eqx.nn.MLP

    def __init__(self, in_features: int, hidden_features: int, num_layers: int, *, key, **kwargs):
        """
        Vector field for the right-hand side of the Diffeq. Enables the ability to do a forward pass
        with the physics-based simulation alone or in combination with an MLP

        Parameters
        ----------
        in_features : int
            number of input features to the model. This is the same as 2 x number of points
            as it is a concatenation of imaginary and real parts
        hidden_features : int
            size of each hidden layer for the MLP
        num_layers : int
            number of hidden layers, including the output layer
        """
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=2*in_features,
            out_size=2*in_features,
            width_size=hidden_features,
            depth=num_layers,
            activation=jnn.sigmoid,
            key=key,
        )

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """
        Forward pass of the model.

        Parameters
        ----------
        t : jnp.ndarray
            JAX array of size (N, ) with the time points
        y : jnp.ndarray
            JAX array of size (2N, ) as a concatenation of imaginary and real parts of rho,
            which is the current state of the system.
        args : tuple
            extra arguments

            index 1: nn_off : bool
                whether to turn off the MLP or not. If True, uses only the physics-based simulation.

        Returns
        -------
        jnp.ndarray:
            JAX array of size (N, ) whih is the output of the model
        """

        if args[1]:  # do NOT use the MLP
            out = (self.simulation(t, y, args))

        else:  # use both simulation + MLP
            out = separate_complex(-1j * recombine_complex(self.mlp(y.flatten()).reshape(y.shape))) + self.simulation(t, y, args)

        return out

    def simulation(self, t, y, args) -> jnp.ndarray:
        """
        Physics-based simulation of the system

        Parameters
        ----------
        t : jnp.ndarray
            JAX array of size (N, ) with the time points
        y : jnp.ndarray
            JAX array of size (2N, ) as a concatenation of imaginary and real parts
        args : tuple
            extra arguments

        Returns
        -------
        jnp.ndarray:
            JAX array of size (N, ) whih is the output of the model
        """
        # Separate real and imaginary parts of rho
        rho_real_imag = y  # state of spins at this time point, so a function of t: ρ(t)
        H0: jnp.ndarray = recombine_complex(args[0])

        # Recombine to form the complex rho matrix
        rho = recombine_complex(rho_real_imag)  # .reshape(H0.shape)

        # Commutator: [H0, rho] (commutator calculation in complex space)
        commutator = -1j * (jnp.matmul(H0, rho) - jnp.matmul(rho, H0.T))

        # Separate the derivative of the real and imaginary parts
        commutator_real_imag = separate_complex(commutator)

        return commutator_real_imag


class NeuralODE(eqx.Module):

    func: Func
    h_mat: jax.Array
    # h_mat: jax.Array = eqx.field(static=False)

    def __init__(self, h_mat, in_features: int, hidden_features: int, num_layers: int, *, key, **kwargs) -> None:
        """
        Neural ODE model

        Parameters
        ----------
        in_features : int
            number of input features to the model. This is the same as 2 x number of points
            as it is a concatenation of imaginary and real parts
        hidden_features : int
            size of each hidden layer for the MLP
        num_layers : int
            number of hidden layers, including the output layer
        """
        super().__init__(**kwargs)
        self.func = Func(in_features, hidden_features, num_layers, key=key)
        self.h_mat = h_mat

    def __call__(
            self,
            # NeuralODE
            ts, y0, op, nn_off: bool,
            # NMR
            t2: float, n_td: int, sw: int, phase: float, apodize: bool,
            calc_hamiltonian_fun, return_solution: bool = False,
    ):
        """
        Forward pass of the model.

        Parameters
        ----------
        ts : jnp.ndarray
            JAX array of size (N, ) containing the time points
        y0 : jnp.ndarray
            JAX array of the hamiltonian matrix
        num_layers : int
            number of hidden layers, including the output layer
        """
        h0 = calc_hamiltonian_fun(self.h_mat)

        # y0 = separate_complex(y0)
        dt = (ts[1] - ts[0])
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            solver=diffrax.Bosh3(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            args=(separate_complex(h0), nn_off),
            max_steps=(4096*2)**2,
        )

        if return_solution:
            return solution.ys

        # assert False, \
        #     f"Trace of the density should be 1.0. Recieved {jnp.trace(recombine_complex(solution.ys), axis1=1, axis2=2)}, {jnp.trace(recombine_complex(y0))}, {recombine_complex(solution.ys).shape}"

        # Vectorize the computation of FID
        FID = jax.vmap(compute_fid, in_axes=(0, 0, None, None))(
            jnp.arange(len(solution.ts)), recombine_complex(solution.ys), recombine_complex(op), h0
        )

        # Apodization and Fourier transform to spectrum
        if apodize:
            FID_apod = apodization(FID, t2, dt)
            return FID_apod
        return FID
