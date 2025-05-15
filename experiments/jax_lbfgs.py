from typing import Callable

import time
import itertools

import numpy as np

import optax
import optax.tree_utils as otu

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from stNMR.jax.hamiltonian import explicit_exponentiation
from stNMR.jax.nmr import generate_spin_operators, calc_hamiltonian


@jax.jit
def make_hamiltonian(v: jax.Array, J: jax.Array, Ix: jax.Array, Iy: jax.Array, Iz: jax.Array, B0: int) -> jax.Array:

    nspins = len(v)
    dim = 2 ** nspins
    J_idx = jnp.array(list(itertools.combinations(range(0, nspins), 2)))

    H0 = jnp.zeros((dim, dim), dtype=jnp.complex64)
    for i in range(nspins):
        H0 += 2 * jnp.pi * B0 * (v[i]-0) * Iz[:,:,i]

    for i in range(J_idx.shape[0]):
        idx_1 = J_idx[i,0]
        idx_2 = J_idx[i,1]
        H0 += 2*jnp.pi*J[i]*(jnp.dot(Ix[:,:,idx_1],Ix[:,:,idx_2]) + jnp.dot(Iy[:,:,idx_1],Iy[:,:,idx_2]) + jnp.dot(Iz[:,:,idx_1], Iz[:,:,idx_2]))

    return H0


def objective_function(x, length):
    """Calculates the MSE error between the predicted and the GT FID."""

    # fid = explicit_exponentiation(
    #     hamiltonian=calc_hamiltonian(h_mat=x.reshape(n_spins, n_spins), B0=B0, Ix=Ix, Iy=Iy, Iz=Iz),
    #     rho=rho, op=OP, ts=ts[:length], dt=ts[1]-ts[0], n_td=length, t2=t2, apodize=apodize,
    # )
    fid = explicit_exponentiation(
        hamiltonian=make_hamiltonian(x[:n_spins], x[n_spins:], B0=B0, Ix=Ix, Iy=Iy, Iz=Iz),
        rho=rho, op=OP, ts=ts[:length], dt=ts[1]-ts[0], n_td=length, t2=t2, apodize=apodize,
    )

    loss = jnp.mean(jnp.abs(fid - gt_fid[:length]) ** 2)
    return loss


def run_opt(init_params: jax.Array, fun: Callable, opt, max_iter, tol):

    def step(carry):
        params, state = carry
        value, grad = jax.value_and_grad(fun)(params)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )

    return final_params, final_state


if __name__ == "__main__":

    import logging
    from tqdm import tqdm
    from pathlib import Path
    from stNMR.dataset import GissmoDataset

    ###################################################################
    #                              Setup                              #
    ###################################################################

    np.random.seed(42)

    ################ Logger ################
    logger = logging.getLogger(name="SciPy L-BFGS")
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
    dataset = GissmoDataset(n_spins=n_spins, return_type="numpy")
    path = Path("data/gissmo/data/bmse000007.csv")  # 9 spins: 43, 7 spins: 29; 5 spins: 39; 4 spins: 104; 3 spins: 7
    d = dataset.from_file(file_path=path)  # or can also use `d = dataset[0]`

    ############ NMR parameters ############
    gissmo_simulation = d[1]  #
    n_td = int(gissmo_simulation.shape[0]/2)  # or can set as `n_td = 2**16`
    N = np.log2(n_td).round(1).astype(int)
    sw = 5000
    aq = (n_td / sw)
    ppm_ref = 4.0
    B0 = 500
    t2 = 0.4
    phase = 0
    apodize = False

    ######## Initialize Hamiltonian ########
    gt_h_mat = d[0] - np.diag(np.full(d[0].shape[0], ppm_ref))  # ground truth (offset subtracted)

    # h_mat = np.triu(np.random.rand(n_spins, n_spins))  # random initialization

    ts = np.linspace(0, aq, n_td)  # time points

    ############### rho state ##############
    rho = IHz
    U90y = expm(-1j * jnp.pi / 2 * IHy)
    rho = jnp.dot(U90y, jnp.dot(rho, U90y.T.conj()))

    ############### FID - GT ###############
    logger.info("Calculating GT FID")
    _start = time.time()
    gt_fid = explicit_exponentiation(
        hamiltonian=calc_hamiltonian(h_mat=gt_h_mat, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz),
        rho=rho, op=OP, ts=ts, dt=ts[1]-ts[0], n_td=n_td, t2=t2, apodize=apodize,
    )
    logger.info(f"Completed GT FID calculation. Time: {time.time() - _start} seconds")

    initial_guess = jnp.asarray(np.random.rand(n_spins * 2,))

    optim = optax.lbfgs()
    max_iter = 1000  # this is specific to BFGS
    gtol = 1e-8  # this is specific to BFGS

    curr_params = initial_guess.copy()
    for n in range(5):

        _start = time.time()
        length = 2 ** (n + 1)
        if length > n_td:
            length = n_td
        
        def fun(p):
            return objective_function(p, length)

        curr_params, _ = run_opt(init_params=curr_params, fun=fun, opt=optim, max_iter=max_iter, tol=gtol,)

        logger.info(f"Current values: {curr_params}")
        logger.info(f"FID Length: {length}")
        logger.info(f"Time: {time.time() - _start} seconds")
        logger.info("-"*50)

    logger.info(curr_params)
