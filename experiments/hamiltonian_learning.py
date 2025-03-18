from typing import Callable
import time

import jax
jax.config.update("jax_enable_x64", True)

import optax
import equinox as eqx

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.linalg import expm as jax_expm

import copy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from stNMR.jax.nmr import generate_spin_operators, calc_hamiltonian, fid_to_spec, freq_to_ppm
from stNMR.jax.hamiltonian import NeuralODE, mse, mse_with_partition, explicit_exponentiation
from stNMR.jax.hamiltonian.utils import separate_complex


@eqx.filter_jit
def make_step(
    rho: jnp.ndarray, ts: jnp.ndarray, op: jnp.ndarray, t2: float,
    n_td: int, sw: int, phase: float, gt: jnp.ndarray, calc_hamiltonian_f: Callable,
    model: NeuralODE, opt_state: optax.OptState, nn_off: bool, filter_spec=None
) -> tuple[jax.Array, NeuralODE, optax.OptState]:
    """Makes a single step in fitting the model."""

    if filter_spec is not None:  # partition the model into differantialble and frozen
        diff_model, static_model = eqx.partition(model, filter_spec)

        # Calculate loss and gradients
        # loss, grads = hellinger_error_with_partition(diff_model, static_model, rho=rho, gt=gt, nn_off=nn_off)
        loss, grads = mse_with_partition(
            diff_model, static_model,
            calc_hamiltonian_f=calc_hamiltonian_f,
            rho=rho, ts=ts, op=op, t2=t2, n_td=n_td,
            sw=sw, phase=phase, gt=gt, nn_off=nn_off
        )

    else:  # proceed without partitioning

        # Calculate loss and gradients
        loss, grads = mse(model, rho=rho, ts=ts, gt=gt, nn_off=nn_off)

    # Get the updates and the OptSate from the optimizer
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))

    # Update the model
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state, grads, updates


def finite_difference_grad(make_step_fn, rho, ts, op, t2, n_td, sw, phase, gt, calc_hamiltonian_f, model, opt_state, nn_off, filter_spec=None, epsilon=1e-5):
    """Computes finite difference gradients for make_step."""
    grads_fd = jnp.zeros(model.h_mat.shape)

    row_idx = list(range(model.h_mat.shape[0]))
    col_idx = list(range(model.h_mat.shape[1]))

    def loss_fn(m):
        loss, _, _, _, _ = make_step_fn(rho, ts, op, t2, n_td, sw, phase, gt, calc_hamiltonian_f, m, opt_state, nn_off, filter_spec)
        return loss

    for row in row_idx:
        for col in col_idx:

            model_plus = copy.deepcopy(model)
            model_minus = copy.deepcopy(model)

            h_mat_plus = model_plus.h_mat.at[row, col].set(model.h_mat[row, col] + epsilon)
            h_mat_minus = model_minus.h_mat.at[row, col].set(model.h_mat[row, col] - epsilon)

            model_plus = jtu.tree_map(lambda x: h_mat_plus if x.shape == model.h_mat.shape else x, eqx.filter(model_plus, eqx.is_array))
            model_minus = jtu.tree_map(lambda x: h_mat_minus if x.shape == model.h_mat.shape else x, eqx.filter(model_minus, eqx.is_array))

            loss_plus = loss_fn(model_plus)
            loss_minus = loss_fn(model_minus)

            grad = (loss_plus - loss_minus) / (2 * epsilon)
            grads_fd = grads_fd.at[row, col].set(grad)

    return grads_fd

###############################################################################################
#                                            MAIN                                             #
###############################################################################################


if __name__ == "__main__":

    import logging
    from tqdm import tqdm
    from stNMR.dataset import GissmoDataset

    ###################################################################
    #                              Setup                              #
    ###################################################################

    ### Basics
    seed = 42
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ### Logger
    logger = logging.getLogger(name="NeuralODE")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f"%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    ### System setup
    n_spins = 3
    Ix, Iy, Iz = generate_spin_operators(n_spins=n_spins)
    IHx, IHy, IHz = Ix[:, :, :].sum(2), Iy[:, :, :].sum(2), Iz[:, :, :].sum(2)
    OP = IHx + 1j * IHy

    ### Dataset
    dataset = GissmoDataset(n_spins=n_spins, return_type="jax")
    # d = dataset[0]
    path = Path("data/gissmo/data/bmse000007.csv")
    d = dataset.from_file(file_path=path)

    ### NMR parameters: {B0, T2, SW, N_TD, AQ, Phase, etc.}
    gissmo_simulation = d[1]
    # sw = gissmo_simulation[:, 0].max() - gissmo_simulation[:, 0].min()
    # sw *= 500

    n_td = int(gissmo_simulation.shape[0]/2)
    logger.info(f"N_TD: {n_td}")
    # aq = (n_td / sw)

    # ppm_ref = (gissmo_simulation[:, 0].max() - gissmo_simulation[:, 0].min())/2

    N = np.log2(n_td).round(1).astype(int)
    # n_td = 2**N
    sw = 10_000
    aq = (n_td / sw)
    ppm_ref = 4.0
    B0 = 500
    t2 = 0.4
    phase = 0
    in_features = 2**n_spins

    ### Initialize the Hamiltonian matrix
    gt_h_mat = d[0] - jnp.diag(jnp.full(d[0].shape[0], ppm_ref))

    h_mat = jnp.triu(
        gt_h_mat + jr.normal(key=key, shape=(n_spins, n_spins)) * 0.1
    )  # random upper triangular matrix
    # h_mat = jr.normal(key=key, shape=(n_spins, n_spins))

    ts = jnp.linspace(0, aq, n_td)
    h0 = calc_hamiltonian(h_mat=h_mat, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz)

    ###################################################################
    #                              Model                              #
    ###################################################################

    model = NeuralODE(
        h_mat=h_mat,
        in_features=h0.ravel().shape[0],
        hidden_features=3,
        num_layers=2,
        key=model_key
    )

    before = model.h_mat  # to keep track of changes

    ### Optimizer
    lr = 1e-4
    optim = optax.adam(learning_rate=lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    ### Initial conditions (rho = IHz); rho is the state of the system
    rho = IHz
    U90y = jax_expm(-1j * jnp.pi / 2 * IHy)
    rho = jnp.dot(U90y, jnp.dot(rho, U90y.T.conj()))
    rho_flat = separate_complex(rho)

    gt_fid = explicit_exponentiation(hamiltonian=calc_hamiltonian(h_mat=gt_h_mat, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz), rho=rho, op=OP, ts=ts, dt=ts[1]-ts[0], n_td=n_td, t2=t2)
    _, _, gt_FTspec, gt_freq_series = fid_to_spec(fid=gt_fid, n_td=n_td, sw=sw, phase=phase)
    gt_ppm = freq_to_ppm(gt_freq_series, ppm_ref, B0)

    ### Spectrum with the randomly initialized matrix, before any fitting is done
    calc_hamiltonian_f = lambda h: calc_hamiltonian(h_mat=h, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz)
    init_fid = model(
        ts=ts,
        y0=rho_flat,
        # h0=h0,
        op=separate_complex(OP),
        nn_off=True,
        t2=t2,
        n_td=n_td,
        sw=sw,
        phase=phase,
        calc_hamiltonian_fun=calc_hamiltonian_f
    )
    _, _, init_FTspec, init_freq_series = fid_to_spec(fid=init_fid, n_td=n_td, sw=sw, phase=phase)
    init_ppm = freq_to_ppm(init_freq_series, ppm_ref, B0)

    ###################################################################
    #                             Fitting                             #
    ###################################################################

    # _loss, _model, _opt_state, _grads, _updates = make_step(
    #     rho=rho_flat, gt=gt_fid.real,
    #     model=model, opt_state=opt_state,
    #     nn_off=True, filter_spec=None
    # )

    ### Fit the hamiltonian matrix
    n_epochs = {
        # "Warm-up": {"num_steps": 100, "nn_off": False, "h_mat_off": False},
        "Hamiltonian": {"num_steps": 200, "nn_off": True, "h_mat_off": False},
        # "Neural Net": {"num_steps": 100, "nn_off": False, "h_mat_off": True},
    }
    total_epochs = 12
    # loss_history = np.zeros(sum([phase["num_epochs"] for phase in n_epochs.values()]))
    loss_history = {
        phase_name: np.zeros(phase_config["num_steps"]) for phase_name, phase_config in n_epochs.items()
    }
    logger.debug("Starting training...")

    frozen_h_mat_state = jtu.tree_map(lambda x: x if x.shape == model.h_mat.shape else None, opt_state)

    autodiff_grads = []
    fd_grads = []

    for n in tqdm(range(N), colour="green"):

        for phase_name, phase_config in n_epochs.items():

            model_before = model

            logger.debug(f"Starting '{phase_name}' phase with {phase_config["num_steps"]} steps")

            if phase_config["h_mat_off"]:  # freeze the Hamiltonian paramters
                # Store the state of `h_mat` when freezing it
                frozen_h_mat_state = jtu.tree_map(lambda x: x if x.shape == model.h_mat.shape else None, opt_state)

                # Freeze `h_mat`
                filter_spec = jtu.tree_map(lambda _: True, model)
                filter_spec = eqx.tree_at(
                    lambda tree: tree.h_mat,
                    filter_spec,
                    replace=(False),
                )

            else:  # unfreeze the Hamiltonian paramters
                # Allow gradients on `h_mat`
                filter_spec = jtu.tree_map(lambda _: True, model)
                filter_spec = eqx.tree_at(
                    lambda tree: tree.h_mat,
                    filter_spec,
                    replace=(True),
                )
                # filter_spec = None
                # Replace the state of `h_mat` with the last trainable state we stored
                opt_state = eqx.combine(opt_state, frozen_h_mat_state)

            phase_start_time = time.time()
            for step in tqdm(range(phase_config["num_steps"]), leave=False, colour="red"):

                # if n < (total_epochs/2):
                    # length = int(n_td*0.01)
                length = int(2 ** (n+1))
                # else:
                #     length = None

                step_start_time = time.time()
                loss, model, opt_state, grads, updates = make_step(
                    rho=rho_flat, ts=ts[:length], op=OP, t2=t2, n_td=n_td, sw=sw, phase=phase,
                    gt=gt_fid.real[:length], calc_hamiltonian_f=calc_hamiltonian_f,
                    model=model, opt_state=opt_state,
                    nn_off=phase_config["nn_off"], filter_spec=filter_spec
                )

                fd_grad = finite_difference_grad(
                    make_step_fn=make_step, rho=rho_flat, ts=ts[:length], op=OP,
                    t2=t2, n_td=n_td, sw=sw, phase=phase,
                    gt=gt_fid.real[:length], calc_hamiltonian_f=calc_hamiltonian_f,
                    model=model, opt_state=opt_state,
                    nn_off=phase_config["nn_off"], filter_spec=filter_spec, epsilon=1e-6
                )

                autodiff_grads.append(grads.h_mat)
                fd_grads.append(fd_grad)

                # TODO: log FIDs at each step!
                logger.info(f"Grads:\n{grads.h_mat}\n{model.h_mat}\n{updates.h_mat}")
                step_end_time = time.time()

                loss_history[phase_name][step] = loss
                logger.debug(msg=f"Step: {step}, Loss: {loss}, Computation time: {step_end_time - step_start_time:.3f}, FID length: {ts[:length].shape[0]}")

            phase_end_time = time.time()
            logger.debug(msg=f"{phase_name} computation time: {phase_end_time - phase_start_time:.3f}")

            model_after = model

    jnp.savez(f"{path.stem}_AutoDiff.npz", jnp.asarray(autodiff_grads))
    jnp.savez(f"{path.stem}_FiniteDiff.npz", jnp.asarray(fd_grads))

    logger.debug("Training finished.")
    logger.debug(f"Initial:\n{before}")
    logger.debug("-"*30)
    logger.debug(f"Fitted:\n{model.h_mat}")
    logger.debug("-"*30)
    logger.debug(f"GT:\n{d[0]}")
    logger.debug("-"*30)
    logger.debug(f"GT - scaled:\n{d[0] - jnp.diag(jnp.full(d[0].shape[0], ppm_ref))}")

    ### Spectrum with the fitted matrix
    final_fid = model(
        ts=ts,
        y0=rho_flat,
        # h0=h0,
        op=separate_complex(OP),
        nn_off=True,
        t2=t2,
        n_td=n_td,
        sw=sw,
        phase=phase,
        calc_hamiltonian_fun=calc_hamiltonian_f
    )
    # ppm_series = (freq_series + (500 * (ppm_ref)))/500
    _, _, final_FTspec, final_freq_series = fid_to_spec(fid=final_fid, n_td=n_td, sw=sw, phase=phase)
    final_ppm = freq_to_ppm(final_freq_series, ppm_ref, B0)


    ###################################################################
    #                              Plots                              #
    ###################################################################

    ### Visualize the spectra
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(12, 6), sharex=True)

    ################################ (1) Final fitted ################################
    axes[0].plot(final_ppm, (final_FTspec).real/jnp.max((final_FTspec).real), label="Simulation")
    axes[0].set_xlabel("Chemical Shift (PPM)")
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].legend()

    ################################ (2) Ground truth ################################
    axes[1].plot(gt_ppm, (gt_FTspec).real/jnp.max((gt_FTspec).real), label="GISSMO")
    axes[1].set_xlabel("Chemical Shift (PPM)")
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].legend()

    ############################ (3) Initial random matrix ###########################
    axes[2].plot(init_ppm, (init_FTspec).real/jnp.max((init_FTspec).real), label="Simulation - Initial")
    axes[2].set_xlabel("Chemical Shift (PPM)")
    axes[2].tick_params(axis='both', labelsize=14)
    axes[2].legend()

    # plt.show()
    plt.savefig(f"figures/{path.stem}_HL-classical.png")
