import numpy as np
from scipy.linalg import expm

from stNMR.numpy.nmr.fid import apodization


def explicit_exponentiation(hamiltonian: np.ndarray, rho: np.ndarray, op:np.ndarray, ts: np.ndarray, dt: float, n_td: int, t2: float, apodize: bool = False) -> np.ndarray:
    """Explicit exponentiation solution to the given system."""
    # Matrix exponential
    P = expm(-1j * hamiltonian * dt)
    P_conj = P.T.conj()

    # Initialize FID as zeros
    FID = np.zeros(n_td, dtype=np.complex128)

    # ATTN: need to vectorize this!
    for i in range(n_td):
        FID[i] = (np.trace(op @ rho))
        rho = P @ rho @ P_conj

    if apodize:
        FID_apod = apodization(fid=FID, t2=t2, dt=dt)
        return FID_apod
    return FID
