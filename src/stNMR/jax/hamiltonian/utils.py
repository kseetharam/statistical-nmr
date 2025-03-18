import jax.numpy as jnp


def separate_complex(rho_complex):
    """Function to separate real and imaginary parts."""
    real_part = jnp.stack([jnp.real(rho_complex), jnp.imag(rho_complex)], axis=-1)
    return real_part


def recombine_complex(rho_real_imag):
    """Function to recombine real and imaginary parts."""
    return rho_real_imag[..., 0] + 1j * rho_real_imag[..., 1]
