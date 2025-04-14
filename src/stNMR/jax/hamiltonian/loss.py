from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from stNMR.jax.hamiltonian.node import NeuralODE
from stNMR.jax.hamiltonian.utils import separate_complex


@eqx.filter_value_and_grad()
def mse(
    model: NeuralODE, calc_hamiltonian_f: Callable,
    rho: jnp.ndarray, ts: jnp.ndarray, op: jnp.ndarray, t2: float, apodize: bool, n_td: int, sw: float, phase: float,
    gt: jnp.ndarray, nn_off: bool=True
) -> jax.Array:

    fid = model(
        ts=ts,
        y0=rho,
        # h0=h0,
        op=separate_complex(op),
        nn_off=nn_off,
        t2=t2,
        apodize=apodize,
        n_td=n_td,
        sw=sw,
        phase=phase,
        calc_hamiltonian_fun=calc_hamiltonian_f
    )

    # Normalize the prediction
    spec_y_values = fid

    # Calcualte and return loss
    return jnp.mean((spec_y_values - gt) ** 2)


@eqx.filter_value_and_grad()
def mse_with_partition(
    diff_model: NeuralODE, static_model: NeuralODE, calc_hamiltonian_f: Callable,
    rho: jnp.ndarray, ts: jnp.ndarray, op: jnp.ndarray, t2: float, apodize: bool, n_td: int, sw: float, phase: float,
    gt: jnp.ndarray, nn_off: bool=True
) -> jax.Array:


    # Combine the two model
    model = eqx.combine(diff_model, static_model)

    fid = model(
        ts=ts,
        y0=rho,
        # h0=h0,
        op=separate_complex(op),
        nn_off=nn_off,
        t2=t2,
        apodize=apodize,
        n_td=n_td,
        sw=sw,
        phase=phase,
        calc_hamiltonian_fun=calc_hamiltonian_f
    )

    # Normalize the prediction
    spec_y_values = fid

    # Calcualte and return loss
    return jnp.mean(jnp.abs(spec_y_values - gt) ** 2)


@eqx.filter_value_and_grad
def hellinger_error(
    model: NeuralODE, calc_hamiltonian_f: Callable,
    rho: jnp.ndarray, ts: jnp.ndarray, op: jnp.ndarray, t2: float, n_td: int, sw: float, phase: float,
    gt: jnp.ndarray, nn_off: bool=True
) -> jax.Array:

    # Prediction
    (FID, time_series, FTspec, freq_series), ys = model(
        ts=ts,
        y0=rho,
        op=separate_complex(op),
        nn_off=nn_off,
        t2=t2,
        n_td=n_td,
        sw=sw,
        phase=phase,
        calc_hamiltonian_fun=calc_hamiltonian_f
    )

    # Normalize the prediction (FTspec) and ground truth (gt)
    spec_y_values = (FTspec).real / jnp.max((FTspec).real)
    # spec_y_values = jax.nn.sigmoid((FTspec).real)
    # gt_normalized = gt / jnp.max(gt)

    # Calculate Hellinger distance: H^2(p, q)
    sqrt_pred = jnp.sqrt(spec_y_values)
    sqrt_gt = jnp.sqrt(gt)
    hellinger_dist_squared = 0.5 * jnp.sum((sqrt_pred - sqrt_gt) ** 2)

    return hellinger_dist_squared


@eqx.filter_value_and_grad
def hellinger_error_with_partition(
    diff_model: NeuralODE, static_model: NeuralODE, calc_hamiltonian_f: Callable,
    rho: jnp.ndarray, ts: jnp.ndarray, op: jnp.ndarray, t2: float, n_td: int, sw: float, phase: float,
    gt: jnp.ndarray, nn_off: bool=True
) -> jax.Array:

    # Combine the two model
    model = eqx.combine(diff_model, static_model)

    # Prediction
    (FID, time_series, FTspec, freq_series), ys = model(
        ts=ts,
        y0=rho,
        op=separate_complex(op),
        nn_off=nn_off,
        t2=t2,
        n_td=n_td,
        sw=sw,
        phase=phase,
        calc_hamiltonian_fun=calc_hamiltonian_f
    )

    # Normalize the prediction (FTspec) and ground truth (gt)
    # spec_y_values = jnp.clip((FTspec).real / jnp.max((FTspec).real), min=0)
    spec_y_values = jnp.maximum((FTspec).real / jnp.max((FTspec).real), 0)

    # gt_normalized = gt / jnp.max(gt)

    # Calculate Hellinger distance: H^2(p, q)
    sqrt_pred = jnp.sqrt(spec_y_values)
    sqrt_gt = jnp.sqrt(gt)
    hellinger_dist_squared = jnp.sum((sqrt_pred - sqrt_gt) ** 2)

    return hellinger_dist_squared
