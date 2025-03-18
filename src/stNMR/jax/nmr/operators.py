import jax
import jax.numpy as jnp


def generate_spin_operators(n_spins: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generates spin operators for the system with the give number of spins."""
    dim = 2**n_spins
    sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64) / 2
    sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64) / 2
    sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64) / 2

    Ix = jnp.zeros((dim, dim, n_spins), dtype=jnp.complex64)
    Iy = jnp.zeros((dim, dim, n_spins), dtype=jnp.complex64)
    Iz = jnp.zeros((dim, dim, n_spins), dtype=jnp.complex64)

    for i in range(n_spins):
        left_identity = jnp.eye(2**i) if i > 0 else jnp.array(1)
        right_identity = (
            jnp.eye(2 ** (n_spins - i - 1)) if (n_spins - i - 1) > 0 else jnp.array(1)
        )
        Ix = Ix.at[:, :, i].set(jnp.kron(jnp.kron(left_identity, sx), right_identity))
        Iy = Iy.at[:, :, i].set(jnp.kron(jnp.kron(left_identity, sy), right_identity))
        Iz = Iz.at[:, :, i].set(jnp.kron(jnp.kron(left_identity, sz), right_identity))

    return Ix, Iy, Iz


@jax.jit
def calc_hamiltonian(h_mat: jnp.ndarray, B0: float, Ix: jnp.ndarray, Iy: jnp.ndarray, Iz: jnp.ndarray) -> jnp.ndarray:
    """Calculates the H0 from the Hamiltonian matrix"""
    nspins = h_mat.shape[0]
    dim = 2**nspins
    H0 = jnp.zeros((dim, dim), dtype=jnp.complex64)
    # print(h_mat)
    for i in range(nspins):
        H0 += 2 * jnp.pi * B0 * h_mat[i, i] * Iz[:, :, i]
        for j in range(i + 1, nspins):
            H0 += 2 * jnp.pi * h_mat[i, j] * (
                jnp.dot(Ix[:, :, i], Ix[:, :, j])
                + jnp.dot(Iy[:, :, i], Iy[:, :, j])
                + jnp.dot(Iz[:, :, i], Iz[:, :, j])
            )
    return H0
