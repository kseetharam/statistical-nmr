from typing import Union

import numpy as np
from scipy.io import loadmat
from pathlib import Path


def read_mat(path: Union[str, Path]) -> np.ndarray:
    """
    Reads a MATLAB file and returns its contents as a NumPy array

    Args:
        path (Union[str, Path]): The path to the MATLAB file to be read. 
            This can be a string or a Path object.
    
    Returns:
        np.ndarray: The contents of the MATLAB file as a NumPy array.

    Example:
        >>> data = read_mat('data.mat')
        >>> print(data.shape)
        (100, 3)
    """
    data = loadmat(path)

    # Access the 'fids', 'ux', and 'uy' arrays from the loaded data
    fids = data['p']['fids'][0, 0]  # fids is stored in a structured array
    ux = data['p']['ux'][0, 0]      # ux is also stored in a structured array
    uy = data['p']['uy'][0, 0]      # uy is also stored in a structured array

    return fids, ux, uy
