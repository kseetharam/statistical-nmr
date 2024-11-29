import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pathlib import Path
from tqdm import tqdm


def create_params_from_csv(csv_file):
    # Load the Hamiltonian parameters from the CSV file
    Hmat = pd.read_csv(csv_file, header=None).values
    
    # Construct the params dictionary based on the CSV content
    params = {
        "N_TD": 2**12,
        "SW": 1000,
        "AQ": None,
        "T2": 1,
        "phase": 0,
        "LarmorFreq": 80,
        "Hmat": Hmat  # Add the Hamiltonian matrix to the params
    }
    
    # Set AQ based on N_TD and SW
    params["AQ"] = params["N_TD"] / params["SW"]
    
    return params


def get_exp_spectra(file: Path) -> pd.DataFrame:

    f = list(Path("statistical-nmr/data/metabolites/urine/exp_spectra").glob(f"*_{file.stem}_*.csv"))

    print(f)
    print(file)

    df = pd.read_csv(f[0])
    return df["ppm"], df["val"]


# Define single-spin operators
sx = np.array([[0, 1], [1, 0]]) / 2  # Pauli-X / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2  # Pauli-Y / 2
sz = np.array([[1, 0], [0, -1]]) / 2  # Pauli-Z / 2
identity = np.eye(2)  # Identity matrix

# Parameters
nspins = 4
dim = 2**nspins

# Initialize spin operator matrices
Ix = np.zeros((dim, dim, nspins), dtype=complex)
Iy = np.zeros((dim, dim, nspins), dtype=complex)
Iz = np.zeros((dim, dim, nspins), dtype=complex)

# Construct spin operator matrices
for i in range(nspins):
    left_identity = np.eye(2**i) if i > 0 else 1
    right_identity = np.eye(2**(nspins - i - 1)) if (nspins - i - 1) > 0 else 1
    Ix[:, :, i] = np.kron(np.kron(left_identity, sx), right_identity)
    Iy[:, :, i] = np.kron(np.kron(left_identity, sy), right_identity)
    Iz[:, :, i] = np.kron(np.kron(left_identity, sz), right_identity)

# Summed spin operators
IHx = np.sum(Ix, axis=2)
IHy = np.sum(Iy, axis=2)
IHz = np.sum(Iz, axis=2)

# Parameters dictionary
# params = {
#     "N_TD": 2**12,
#     "SW": 1000,
#     "AQ": None,
#     "T2": 1,
#     "phase": 0,
#     "LarmorFreq": 80,
# }
# params["AQ"] = params["N_TD"] / params["SW"]

folder = Path("statistical-nmr/data/metabolites/urine/params")

for file in tqdm(folder.iterdir()):

    if file.is_file() and file.suffix == ".csv":


        params = create_params_from_csv(file)

        # Define operators and initial state
        OP = IHx + 1j * IHy
        rho = IHz
        U90y = expm(-1j * np.pi / 2 * IHy)  # Use expm for matrix exponentiation
        rho = U90y @ rho @ U90y.T.conj()

        # Random matrix for testing
        # Hmat = np.random.rand(dim, dim)
        Hmat = params["Hmat"]

        # Calculate the Hamiltonian
        def calc_Hamiltonian(Hmat, B0, Ix, Iy, Iz, nspins):
            dim = 2**nspins
            H0 = np.zeros((dim, dim), dtype=complex)
            for i in range(nspins):
                H0 += 2 * np.pi * B0 * Hmat[i, i] * Iz[:, :, i]
                for j in range(i + 1, nspins):
                    H0 += 2 * np.pi * Hmat[i, j] * (
                        np.dot(Ix[:, :, i], Ix[:, :, j])
                        + np.dot(Iy[:, :, i], Iy[:, :, j])
                        + np.dot(Iz[:, :, i], Iz[:, :, j])
                    )
            return H0

        H0 = calc_Hamiltonian(Hmat, params["LarmorFreq"], Ix, Iy, Iz, nspins)

        # Calculate FID and spectrum
        def calc_spec(params, OP, H0, rho):
            dt = params["AQ"] / params["N_TD"]
            P = expm(-1j * H0 * dt)
            idx = np.arange(params["N_TD"])
            apod = np.exp(-dt / params["T2"] * idx)
            FID = np.zeros(params["N_TD"], dtype=complex)
            for i in range(params["N_TD"]):
                FID[i] = np.trace(OP @ rho) * apod[i]
                rho = P @ rho @ P.T.conj()
            spec = np.fft.fftshift(np.fft.fft(FID, 2 * params["N_TD"]))
            time_series = np.linspace(0, params["AQ"], params["N_TD"])
            freq_series = np.linspace(-params["SW"] / 2, params["SW"] / 2, 2 * params["N_TD"])
            FTspec = np.exp(1j * params["phase"]) * spec
            return FID, time_series, FTspec, freq_series

        FID, time_series, FTspec, freq_series = calc_spec(params, OP, H0, rho)

        # Plot the spectrum
        # plt.figure()
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
        axes[0].plot(freq_series, np.real(FTspec))
        axes[0].invert_xaxis()
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].tick_params(axis='both', labelsize=14)
        axes[0].set_title("Simulated")
        
        exp_data = get_exp_spectra(file)

        axes[1].plot(exp_data[0], exp_data[1])
        axes[1].set_title("Experimental")

        # plt.gcf().set_facecolor("w")
        # plt.show()
        plt.savefig((Path("figures") / file.name).with_suffix(".png"))
        plt.clf()

        # print(file)

        break
