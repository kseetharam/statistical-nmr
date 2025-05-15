import torch

def generate_spin_operators(n_spins: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates spin operators for the system with the given number of spins."""
    dim = 2 ** n_spins
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat) / 2

    Ix = torch.zeros((dim, dim, n_spins), dtype=torch.cfloat)
    Iy = torch.zeros((dim, dim, n_spins), dtype=torch.cfloat)
    Iz = torch.zeros((dim, dim, n_spins), dtype=torch.cfloat)

    for i in range(n_spins):
        # Left and right identity matrices
        left_identity = torch.eye(2**i, dtype=torch.cfloat) if i > 0 else torch.tensor(1, dtype=torch.cfloat)
        right_identity = (
            torch.eye(2 ** (n_spins - i - 1), dtype=torch.cfloat) 
            if (n_spins - i - 1) > 0 else torch.tensor(1, dtype=torch.cfloat)
        )

        # Kronecker products for spin operators
        Ix[:, :, i] = torch.kron(torch.kron(left_identity, sx), right_identity)
        Iy[:, :, i] = torch.kron(torch.kron(left_identity, sy), right_identity)
        Iz[:, :, i] = torch.kron(torch.kron(left_identity, sz), right_identity)

    return Ix, Iy, Iz


def calc_hamiltonian(h_mat: torch.Tensor, B0: float, Ix: torch.Tensor, Iy: torch.Tensor, Iz: torch.Tensor) -> torch.Tensor:
    """Calculates the H0 from the Hamiltonian matrix"""
    nspins = h_mat.shape[0]
    dim = 2 ** nspins
    H0 = torch.zeros((dim, dim), dtype=torch.cfloat, device=h_mat.device)

    for i in range(nspins):
        H0 += 2 * torch.pi * B0 * h_mat[i, i] * Iz[:, :, i]
        for j in range(i + 1, nspins):
            H0 += 2 * torch.pi * h_mat[i, j] * (
                torch.matmul(Ix[:, :, i], Ix[:, :, j]) +
                torch.matmul(Iy[:, :, i], Iy[:, :, j]) +
                torch.matmul(Iz[:, :, i], Iz[:, :, j])
            )
    return H0
