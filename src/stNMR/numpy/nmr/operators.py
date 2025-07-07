import itertools
import numpy as np


def generate_spin_operators(n_spins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates spin operators for the system with the give number of spins."""
    dim = 2 ** n_spins
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex64) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex64) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex64) / 2

    Ix = np.zeros((dim, dim, n_spins), dtype=np.complex64)
    Iy = np.zeros((dim, dim, n_spins), dtype=np.complex64)
    Iz = np.zeros((dim, dim, n_spins), dtype=np.complex64)

    for i in range(n_spins):
        left_identity = np.eye(2**i) if i > 0 else np.array(1)
        right_identity = (
            np.eye(2 ** (n_spins - i - 1)) if (n_spins - i - 1) > 0 else np.array(1)
        )
        Ix[:, :, i] = (np.kron(np.kron(left_identity, sx), right_identity))
        Iy[:, :, i] = (np.kron(np.kron(left_identity, sy), right_identity))
        Iz[:, :, i] = (np.kron(np.kron(left_identity, sz), right_identity))

    return Ix, Iy, Iz


def calc_hamiltonian(h_mat: np.ndarray, B0: float, Ix: np.ndarray, Iy: np.ndarray, Iz: np.ndarray) -> np.ndarray:
    """Calculates the H0 from the Hamiltonian matrix"""
    nspins = h_mat.shape[0]
    dim = 2**nspins
    H0 = np.zeros((dim, dim), dtype=np.complex64)
    # print(h_mat)
    for i in range(nspins):
        H0 += 2 * np.pi * B0 * h_mat[i, i] * Iz[:, :, i]
        for j in range(i + 1, nspins):
            H0 += 2 * np.pi * h_mat[i, j] * (
                np.dot(Ix[:, :, i], Ix[:, :, j])
                + np.dot(Iy[:, :, i], Iy[:, :, j])
                + np.dot(Iz[:, :, i], Iz[:, :, j])
            )
    return H0


def hamiltonian_from_vectors(v: np.ndarray, J: np.ndarray, Ix: np.ndarray, Iy: np.ndarray, Iz: np.ndarray, B0: int) -> np.ndarray:

    nspins = len(v)
    dim = 2 ** nspins
    J_idx = np.array(list(itertools.combinations(range(0, nspins), 2)))

    H0 = np.zeros((dim, dim), dtype=complex)
    for i in range(nspins):
        H0 += 2 * np.pi * B0 * (v[i]-0) * Iz[:,:,i]

    for i in range(J_idx.shape[0]):
        idx_1 = J_idx[i,0]
        idx_2 = J_idx[i,1]
        H0 += 2*np.pi*J[i]*(np.dot(Ix[:,:,idx_1],Ix[:,:,idx_2]) + np.dot(Iy[:,:,idx_1],Iy[:,:,idx_2]) + np.dot(Iz[:,:,idx_1], Iz[:,:,idx_2]))

    return H0
