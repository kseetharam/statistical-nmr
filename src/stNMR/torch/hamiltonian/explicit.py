import torch

from stNMR.torch.nmr.fid import apodization


def explicit_exponentiation(
    hamiltonian: torch.Tensor, rho: torch.Tensor, op: torch.Tensor, 
    ts: torch.Tensor, dt: float, n_td: int, t2: float, apodize: bool = False
) -> torch.Tensor:
    """Explicit exponentiation solution to the given system."""
    # Matrix exponential using scipy (still efficient for dense matrices)
    P = torch.matrix_exp(-1j * hamiltonian * dt)

    # Initialize FID as zeros
    FID = torch.zeros(n_td, dtype=torch.cfloat, device=hamiltonian.device)

    # Explicit exponentiation loop
    for i in range(n_td):
        FID[i] = torch.trace(torch.matmul(op, rho))
        rho = torch.matmul(P, torch.matmul(rho, P.conj().T))

    if apodize:
        FID = apodization(fid=FID, t2=t2, dt=dt)

    return FID
