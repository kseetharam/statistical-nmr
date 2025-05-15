from typing import Union, Optional, Literal

import torch
from torch.utils.data import Dataset

import numpy as np
from jax import numpy as jnp

import pandas as pd
from pathlib import Path


class GissmoDataset(Dataset):

    _DEFAULT_DATA_FOLDER = Path("data/gissmo/data")
    _DEFAULT_SPEC_FOLDER = Path("DB")
    _DEFAULT_EXTENSION = ".csv"

    def __init__(
            self,
            data_folder: Optional[Union[str, Path]] = None,
            spec_folder: Optional[Union[str, Path]] = None,
            n_spins: Optional[int] = None,
            return_type: Literal["torch", "jax", "numpy"] = "torch"
    ) -> None:

        assert return_type in ["torch", "jax", "numpy"],\
            f"{return_type} is not a supported return type. Use one of 'torch', 'jax', or 'numpy'."
        self.n_spins = n_spins

        if data_folder is None:
            self.data_folder = self._DEFAULT_DATA_FOLDER
        else:
            self.data_folder = Path(data_folder)
        
        # self._files = [f for f in self.data_folder.iterdir() if f.is_file() and f.suffix == self._DEFAULT_EXTENSION]
        self._files = [
            f for f in self.data_folder.iterdir()
            if f.is_file() and f.suffix == self._DEFAULT_EXTENSION and sum(1 for _ in f.open()) == (self.n_spins + 1)
        ]  # this is not really ideal, but it does the job

        if spec_folder is None:
            self.spec_folder = self._DEFAULT_SPEC_FOLDER
        else:
            self.spec_folder = Path(spec_folder)

        self.return_type = return_type

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[jnp.ndarray, jnp.ndarray], tuple[np.ndarray, np.ndarray]]:

        hamiltonian_file = self._files[idx]
        spectra_file = self.spec_folder / hamiltonian_file.stem / "simulation_1" / "B0s" / "sim_default.csv"

        h_mat = pd.read_csv(hamiltonian_file, header=0).to_numpy()
        spectra = pd.read_csv(spectra_file).to_numpy()

        if self.return_type == "torch":
            return torch.from_numpy(h_mat), torch.from_numpy(spectra)

        elif self.return_type == "jax":
            return jnp.array(h_mat), jnp.array(spectra)

        else:
            return h_mat, spectra

    def _collate_fn(self, batch):

        if self.return_type == "torch":
            files, spectra = map(list, zip(*batch))
            spectra = torch.stack(spectra)
            return files, spectra

        else:
            raise NotImplementedError(f"`_collate_fn` not implemented for '{self.return_type}' return type yet!")

    def from_file(self, file_path: Union[str, Path]) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[jnp.ndarray, jnp.ndarray], tuple[np.ndarray, np.ndarray]]:
        
        hamiltonian_file = Path(file_path)
        spectra_file = self.spec_folder / hamiltonian_file.stem / "simulation_1" / "B0s" / "sim_default.csv"

        h_mat = pd.read_csv(hamiltonian_file, header=0).to_numpy()
        spectra = pd.read_csv(spectra_file).to_numpy()

        if self.return_type == "torch":
            return torch.from_numpy(h_mat), torch.from_numpy(spectra)

        elif self.return_type == "jax":
            return jnp.array(h_mat), jnp.array(spectra)

        else:
            return h_mat, spectra
