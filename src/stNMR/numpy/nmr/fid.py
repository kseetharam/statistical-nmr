import numpy as np
from matplotlib import pyplot as plt


def apodization(fid: np.ndarray, t2: float, dt: float) -> np.ndarray:
    """Applies apodization to the FID."""
    idx = np.arange(fid.shape[0])
    apod = np.exp(-dt / t2 * idx)
    return fid * apod


def fourier_transform(fid: np.ndarray, n_td: int, sw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fourier transforms the FID."""
    spec = np.fft.fftshift(np.fft.fft(fid, 2 * n_td))
    time_series = np.linspace(0, n_td / sw, n_td)
    freq_series = np.linspace(-sw / 2, sw / 2, 2 * n_td)
    return spec, freq_series, time_series


# TODO: implement this!
def inverse_fourier_transform():
    raise NotImplementedError()


def compute_fid(i: int, rho_t: np.ndarray, op: np.ndarray, h0: np.ndarray) -> np.ndarray:
    """Extracts the FID from the solution."""
    return np.trace(op @ rho_t.reshape(h0.shape))


def fid_to_spec(fid: np.ndarray, n_td: int, sw: int, phase: float, normalize: bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converts the FID to spectrum."""
    # Perform Fourier transform
    spec, freq_series, time_series = fourier_transform(
        fid, n_td, sw
    )
    # Fourier transformed spectrum
    FTspec = np.exp(1j * phase) * spec
    return (fid, time_series, FTspec, freq_series)


def freq_to_ppm(freq: np.ndarray, ppm_offset: float, b0: float) -> np.ndarray:
    """Converts the freq array to ppm array."""
    return (freq + (b0 * (ppm_offset)))/b0


def plot_spectrum(ppm: np.ndarray, ft_values: np.ndarray,  *args, **kwargs) -> None:
    """Plots the given spectrum."""
    plt.plot(
        ppm, (ft_values.real)/max(ft_values.real), *args, **kwargs,
    )
