from typing import Union

from jax import Array
from torch import Tensor
from numpy import ndarray


array_like = Union[ndarray, Tensor, Array]


def to_tensor(x: array_like) -> Tensor:
    """
    Converts an array-like object to a PyTorch tensor.
    """
    if isinstance(x, Tensor):
        return x
    elif isinstance(x, ndarray):
        from torch import from_numpy
        return from_numpy(x)
    elif isinstance(x, Array):
        from torch import from_dlpack
        return from_dlpack(x)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(x)}")


def to_array(x: array_like) -> ndarray:
    """
    Converts an array-like object to a NumPy ndarray.
    """
    if isinstance(x, ndarray):
        return x
    elif isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, Array):
        from numpy import asarray
        return asarray(x)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(x)}")


def to_jax_array(x: array_like) -> Array:
    """
    Converts an array-like object to a JAX Array.
    """
    if isinstance(x, Array):
        return x
    elif isinstance(x, Tensor):
        from jax import numpy as jnp
        return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, ndarray):
        from jax import numpy as jnp
        return jnp.array(x)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(x)}")
