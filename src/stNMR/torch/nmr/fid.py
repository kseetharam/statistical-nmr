import torch
from matplotlib import pyplot as plt


def apodization(fid: torch.Tensor, t2: float, dt: float) -> torch.Tensor:
    """Applies apodization to the FID."""
    idx = torch.arange(fid.shape[0], dtype=torch.float32, device=fid.device)
    apod = torch.exp(-dt / t2 * idx)
    return fid * apod


def fourier_transform(fid: torch.Tensor, n_td: int, sw: float) -> tuple:
    """Fourier transforms the FID."""
    spec = torch.fft.fftshift(torch.fft.fft(fid, n=2 * n_td))
    time_series = torch.linspace(0, n_td / sw, n_td, dtype=torch.float32)
    freq_series = torch.linspace(-sw / 2, sw / 2, 2 * n_td, dtype=torch.float32)
    return spec, freq_series, time_series


def inverse_fourier_transform():
    """Placeholder for the inverse Fourier transform."""
    raise NotImplementedError("Inverse Fourier transform is not implemented yet.")


def compute_fid(i: int, rho_t: torch.Tensor, op: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Extracts the FID from the solution."""
    return torch.trace(torch.matmul(op, rho_t.reshape(h0.shape)))


def fid_to_spec(fid: torch.Tensor, n_td: int, sw: int, phase: float, normalize: bool=False) -> tuple:
    """Converts the FID to spectrum."""
    # Perform Fourier transform
    spec, freq_series, time_series = fourier_transform(fid, n_td, sw)
    # Fourier transformed spectrum with phase adjustment
    FTspec = torch.exp(torch.tensor(1j * phase).to(fid.device)) * spec
    if normalize:
        FTspec /= torch.max(torch.abs(FTspec))
    return fid, time_series, FTspec, freq_series


def freq_to_ppm(freq: torch.Tensor, ppm_offset: float, b0: float) -> torch.Tensor:
    """Converts the freq array to ppm array."""
    return (freq + (b0 * ppm_offset)) / b0


def plot_spectrum(ppm: torch.Tensor, ft_values: torch.Tensor, *args, **kwargs) -> None:
    """Plots the given spectrum."""
    plt.plot(ppm.numpy(), (ft_values.real).numpy() / torch.max(ft_values.real).numpy(), *args, **kwargs)
    plt.xlabel("PPM")
    plt.ylabel("Intensity")
    plt.title("NMR Spectrum")
    plt.show()
