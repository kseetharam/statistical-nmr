from typing import Union, Dict, Any

import numpy as np
import pandas as pd
from pathlib import Path


# TODO: customize values
def create_params_from_csv(csv_file: Union[str, Path]) -> Dict[str, Any]:

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
        "Hmat": Hmat,  # Add the Hamiltonian matrix to the params
        "n_spins": Hmat.shape[0],
    }

    # Set AQ based on N_TD and SW
    params["AQ"] = params["N_TD"] / params["SW"]

    return params


# Define the Pauli matrices and identity matrix
Sx = np.array([[0, 1], [1, 0]]) / 2  # Pauli-X / 2
Sy = np.array([[0, -1j], [1j, 0]]) / 2  # Pauli-Y / 2
Sz = np.array([[1, 0], [0, -1]]) / 2  # Pauli-Z / 2
