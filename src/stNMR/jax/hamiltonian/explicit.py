from jax import numpy as jnp
from jax.scipy.linalg import expm as jax_expm

from stNMR.jax.nmr.fid import apodization


def explicit_exponentiation(hamiltonian: jnp.ndarray, rho: jnp.ndarray, op:jnp.ndarray, ts: jnp.ndarray, dt: float, n_td: int, t2: float) -> jnp.ndarray:
    """Explicit exponentiation solution to the given system."""
    # Matrix exponential
    P = jax_expm(-1j * hamiltonian * dt)

    # Initialize FID as zeros
    FID = jnp.zeros(n_td, dtype=jnp.complex128)

    # ATTN: need to vectorize this!
    for i in range(n_td):
        FID = FID.at[i].set(jnp.trace(op @ rho))
        rho = P @ rho @ P.T.conj()

    FID_apod = apodization(fid=FID, t2=t2, dt=dt)

    return FID_apod
