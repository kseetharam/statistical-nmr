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

import numpy as np
from pathlib import Path

from stNMR.jax.nmr import generate_spin_operators, calc_hamiltonian, fid_to_spec, freq_to_ppm
from stNMR.jax.hamiltonian import NeuralODE, mse, mse_with_partition, explicit_exponentiation
from stNMR.jax.hamiltonian.utils import separate_complex


@eqx.filter_jit
def make_step(
    rho: jnp.ndarray, ts: jnp.ndarray, op: jnp.ndarray, t2: float, apodize: bool,
    n_td: int, sw: int, phase: float, gt: jnp.ndarray, calc_hamiltonian_f: Callable,
    model: NeuralODE, opt_state: optax.OptState, nn_off: bool, filter_spec=None
) -> tuple[jax.Array, NeuralODE, optax.OptState]:
    """Makes a single step in fitting the model."""

    if filter_spec is not None:  # partition the model into differantialble and frozen
        diff_model, static_model = eqx.partition(model, filter_spec)

        # Calculate loss and gradients
        loss, grads = mse_with_partition(
            diff_model, static_model,
            calc_hamiltonian_f=calc_hamiltonian_f,
            rho=rho, ts=ts, op=op, t2=t2, apodize=apodize,
            n_td=n_td, sw=sw, phase=phase, gt=gt, nn_off=nn_off
        )

    else:  # proceed without partitioning

        # Calculate loss and gradients
        loss, grads = mse(
            model, calc_hamiltonian_f=calc_hamiltonian_f,
            rho=rho, ts=ts, op=op, t2=t2, apodize=apodize,
            n_td=n_td, sw=sw, phase=phase, gt=gt, nn_off=nn_off
        )

    # Get the updates and the OptSate from the optimizer
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))

    # Update the model
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state, grads, updates

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

    ################ Basics ################
    seed = 42
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ################ Logger ################
    logger = logging.getLogger(name="NeuralODE")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    ############# System setup #############
    n_spins = 3
    Ix, Iy, Iz = generate_spin_operators(n_spins=n_spins)
    IHx, IHy, IHz = Ix[:, :, :].sum(2), Iy[:, :, :].sum(2), Iz[:, :, :].sum(2)
    OP = IHx + 1j * IHy

    ################ Dataset ###############
    dataset = GissmoDataset(n_spins=n_spins, return_type="jax")
    path = Path("data/gissmo/data/bmse000007.csv")
    d = dataset.from_file(file_path=path)  # or can also use `d = dataset[0]`

    ############ NMR parameters ############
    gissmo_simulation = d[1]  #
    n_td = int(gissmo_simulation.shape[0]/2)  # or can set as `n_td = 2**16`
    N = np.log2(n_td).round(1).astype(int)
    sw = 10000
    aq = (n_td / sw)
    ppm_ref = 4.0
    B0 = 500
    t2 = 0.4
    phase = 0
    apodize = False
    in_features = 2**n_spins  # this is for the MLP

    logger.info(f"N_TD: {n_td}")

    ######## Initialize Hamiltonian ########
    gt_h_mat = d[0] - jnp.diag(jnp.full(d[0].shape[0], ppm_ref))  # ground truth (offset subtracted)
    h_mat = jnp.triu(jnp.asarray(-10 + 20 * np.random.rand(n_spins, n_spins)))  # random initialization

    ts = jnp.linspace(0, aq, n_td)  # time points
    h0 = calc_hamiltonian(h_mat=h_mat, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz)  # Hamiltonian matrix

    ###################################################################
    #                              Model                              #
    ###################################################################

    ########### Initialize Model ###########
    model = NeuralODE(
        h_mat=h_mat,
        in_features=in_features,
        hidden_features=16,
        num_layers=2,
        key=model_key
    )

    before = model.h_mat  # to keep track of changes

    ############### Optimizer ##############
    lr = 1e-1
    optim = optax.chain(
        optax.scale_by_lbfgs(),
        optax.scale(-lr),  # minus sign to *minimize* the loss.
    )
    max_iter = 1000  # this is specific to BFGS
    gtol = 1e-8  # this is specific to BFGS
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    ############### rho state ##############
    rho = IHz
    U90y = jax_expm(-1j * jnp.pi / 2 * IHy)
    rho = jnp.dot(U90y, jnp.dot(rho, U90y.T.conj()))
    rho_flat = separate_complex(rho)

    ############### FID - GT ###############
    gt_fid = explicit_exponentiation(
        hamiltonian=calc_hamiltonian(h_mat=gt_h_mat, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz),
        rho=rho, op=OP, ts=ts, dt=ts[1]-ts[0], n_td=n_td, t2=t2, apodize=apodize,
    )
    _, _, gt_FTspec, gt_freq_series = fid_to_spec(fid=gt_fid, n_td=n_td, sw=sw, phase=phase)
    gt_ppm = freq_to_ppm(gt_freq_series, ppm_ref, B0)

    ############ FID - Initial ##############
    calc_hamiltonian_f = lambda h: calc_hamiltonian(h_mat=h, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz)
    init_fid = model(
        ts=ts,
        y0=rho_flat,
        # h0=h0,
        op=separate_complex(OP),
        nn_off=True,
        t2=t2,
        apodize=apodize,
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

    ################# Specs #################
    n_epochs = {
        "Hamiltonian": {"num_steps": 1, "nn_off": True, "h_mat_off": False},
    }  # `nn_off = False` means that the neural net is ignored and only the Hamiltonian is used
    loss_history = {
        phase_name: np.zeros(phase_config["num_steps"]) for phase_name, phase_config in n_epochs.items()
    }
    logger.debug("Starting training...")

    ############# Filter spec ##############
    # This is used make sure we allow gradients on `h_mat`.
    # Technically, we don't need this but because the code is developed
    # such that we can freeze `h_mat` and train the neural net, we need to
    # specify here that we want to train `h_mat` as well.
    filter_spec = jtu.tree_map(lambda _: True, model)
    filter_spec = eqx.tree_at(
        lambda tree: tree.h_mat,
        filter_spec,
        replace=(True),
    )

    logger.info(f"Initial Hamiltonian:\n{model.h_mat}")

    for n in tqdm(range(N), colour="green"):

        for phase_name, phase_config in n_epochs.items():

            model_before = model  # to keep track of changes

            logger.debug(f"Starting '{phase_name}' phase with {phase_config["num_steps"]} steps")

            phase_start_time = time.time()
            for step in tqdm(range(phase_config["num_steps"]), leave=False, colour="red"):

                length = int(2 ** (n+1))  # this is for bootstrapping

                step_start_time = time.time()
                for i in range(max_iter):  # BFGS iterations

                    loss, model, opt_state, grads, updates = make_step(
                        rho=rho_flat, ts=ts[:length], op=OP, t2=t2, apodize=apodize, n_td=n_td, sw=sw, phase=phase,
                        gt=gt_fid[:length], calc_hamiltonian_f=calc_hamiltonian_f,
                        model=model, opt_state=opt_state,
                        nn_off=phase_config["nn_off"], filter_spec=filter_spec
                    )

                    if jnp.linalg.norm(grads.h_mat) < gtol:  # convergence check
                        logger.info(f"Successfully terminated with {i} steps.")
                        break
                
                else:  # if all iterations are exhausted, check if the norm is above the threshold
                    if jnp.linalg.norm(grads.h_mat) >= gtol:
                        logger.warning(f"Grad norm {jnp.linalg.norm(grads.h_mat)} is above the threshold {gtol}.")

                logger.debug(f"\n{model.h_mat}\n{grads.h_mat}\n{updates.h_mat}")
                step_end_time = time.time()

                loss_history[phase_name][step] = loss
                logger.debug(msg=f"Step: {step}, Loss: {loss}, Computation time: {step_end_time - step_start_time:.3f}, FID length: {ts[:length].shape[0]}")

            phase_end_time = time.time()
            logger.debug(msg=f"{phase_name} computation time: {phase_end_time - phase_start_time:.3f}")

            model_after = model

    ############ Log the results ############
    logger.debug("Training finished.")
    logger.debug(f"Initial:\n{before}")
    logger.debug("-"*30)
    logger.debug(f"Fitted:\n{model.h_mat}")
    logger.debug("-"*30)
    logger.debug(f"GT:\n{d[0]}")
    logger.debug("-"*30)
    logger.debug(f"GT - scaled:\n{d[0] - jnp.diag(jnp.full(d[0].shape[0], ppm_ref))}")

    ############# FID - Final ###############
    final_fid = model(
        ts=ts,
        y0=rho_flat,
        op=separate_complex(OP),
        nn_off=True,
        t2=t2,
        apodize=apodize,
        n_td=n_td,
        sw=sw,
        phase=phase,
        calc_hamiltonian_fun=calc_hamiltonian_f
    )
    _, _, final_FTspec, final_freq_series = fid_to_spec(fid=final_fid, n_td=n_td, sw=sw, phase=phase)
    final_ppm = freq_to_ppm(final_freq_series, ppm_ref, B0)

    ############# Save Results ##############
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    jnp.savez(
        results_dir / f"{path.stem}_HL-BFGSX-random-n_td={n_td}-SEED={seed}--RESULTS.npz",
        fitted_ppm=final_ppm,
        fitted_spec=final_FTspec,
        fitted_fid=final_fid,
        gt_ppm=gt_ppm,
        gt_spec=gt_FTspec,
        gt_fid=gt_fid,
        init_ppm=init_ppm,
        init_spec=init_FTspec,
        init_fid=init_fid,
        h_mat=model.h_mat,
        seed=seed,
        **loss_history,
    )
