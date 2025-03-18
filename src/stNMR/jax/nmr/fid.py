import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt


def apodization(fid: jnp.ndarray, t2: float, dt: float) -> jnp.ndarray:
    """Applies apodization to the FID."""
    idx = jnp.arange(fid.shape[0])
    apod = jnp.exp(-dt / t2 * idx)
    return fid * apod


def fourier_transform(fid: jnp.ndarray, n_td: int, sw: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fourier transforms the FID."""
    spec = jnp.fft.fftshift(jnp.fft.fft(fid, 2 * n_td))
    time_series = jnp.linspace(0, n_td / sw, n_td)
    freq_series = jnp.linspace(-sw / 2, sw / 2, 2 * n_td)
    return spec, freq_series, time_series


# TODO: implement this!
def inverse_fourier_transform():
    raise NotImplementedError()


@jax.jit
def compute_fid(i: int, rho_t: jnp.ndarray, op: jnp.ndarray, h0: jnp.ndarray) -> jnp.ndarray:
    """Extracts the FID from the solution."""
    return jnp.trace(op @ rho_t.reshape(h0.shape))


def fid_to_spec(fid: jnp.ndarray, n_td: int, sw: int, phase: float, normalize: bool=False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Converts the FID to spectrum."""
    # Perform Fourier transform
    spec, freq_series, time_series = fourier_transform(
        fid, n_td, sw
    )
    # Fourier transformed spectrum
    FTspec = jnp.exp(1j * phase) * spec
    return (fid, time_series, FTspec, freq_series)


def freq_to_ppm(freq: jnp.ndarray, ppm_offset: float, b0: float) -> jnp.ndarray:
    """Converts the freq array to ppm array."""
    return (freq + (b0 * (ppm_offset)))/b0


def plot_spectrum(ppm: jnp.ndarray, ft_values: jnp.ndarray,  *args, **kwargs) -> None:
    """Plots the given spectrum."""
    plt.plot(
        ppm, (ft_values.real)/max(ft_values.real), *args, **kwargs,
    )
