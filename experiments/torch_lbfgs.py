from typing import Iterable

import time
import math
import itertools
import numpy as np

import torch
import torch.optim as optim

from stNMR.torch.hamiltonian import explicit_exponentiation
from stNMR.torch.nmr import generate_spin_operators, calc_hamiltonian


def make_hamiltonian(v: torch.Tensor, J: torch.Tensor, Ix: torch.Tensor, Iy: torch.Tensor, Iz: torch.Tensor, B0: int) -> torch.Tensor:

    nspins = len(v)
    dim = 2 ** nspins
    J_idx = torch.from_numpy(np.array(list(itertools.combinations(range(0, nspins), 2)))).to(v.device)

    H0 = torch.zeros((dim, dim), dtype=torch.complex64, device=v.device)
    for i in range(nspins):
        H0 += 2 * torch.pi * B0 * (v[i]-0) * Iz[:,:,i]

    for i in range(J_idx.shape[0]):
        idx_1 = J_idx[i,0]
        idx_2 = J_idx[i,1]
        H0 += 2*torch.pi*J[i]*(torch.matmul(Ix[:,:,idx_1],Ix[:,:,idx_2]) + torch.matmul(Iy[:,:,idx_1],Iy[:,:,idx_2]) + torch.matmul(Iz[:,:,idx_1], Iz[:,:,idx_2]))

    return H0


def objective_function(x: torch.Tensor, length: int):
    """Calculates the MSE error between the predicted and the GT FID."""
    # fid = explicit_exponentiation(
    #     hamiltonian=calc_hamiltonian(h_mat=x.reshape(n_spins, n_spins), B0=B0, Ix=Ix, Iy=Iy, Iz=Iz),
    #     rho=rho, op=OP, ts=ts[:length], dt=ts[1]-ts[0], n_td=length, t2=t2, apodize=apodize,
    # )

    fid = explicit_exponentiation(
        hamiltonian=make_hamiltonian(x[:n_spins], x[n_spins:], B0=B0, Ix=Ix, Iy=Iy, Iz=Iz),
        rho=rho, op=OP, ts=ts[:length], dt=ts[1]-ts[0], n_td=length, t2=t2, apodize=apodize,
    )

    loss = torch.mean(torch.abs(fid - gt_fid[:length]) ** 2)
    return loss


def calculate_gradient_norm(parameters: Iterable[torch.nn.Parameter]):
    """Calculate the L2 norm of gradients."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


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
    logger = logging.getLogger(name="PyTorch L-BFGS")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ############# System setup #############
    n_spins = 3
    Ix, Iy, Iz = generate_spin_operators(n_spins=n_spins)
    Ix, Iy, Iz = Ix.to(device), Iy.to(device), Iz.to(device)

    IHx, IHy, IHz = Ix[:, :, :].sum(2), Iy[:, :, :].sum(2), Iz[:, :, :].sum(2)
    OP = IHx + 1j * IHy

    ################ Dataset ###############
    dataset = GissmoDataset(n_spins=n_spins, return_type="torch")
    path = Path("data/gissmo/data/bmse000007.csv")  # 9 spins: 43, 7 spins: 29; 5 spins: 39; 4 spins: 104; 3 spins: 7
    d = dataset.from_file(file_path=path)  # or can also use `d = dataset[0]`

    ############ NMR parameters ############
    gissmo_simulation = d[1]  #
    n_td = int(gissmo_simulation.shape[0]/2)  # or can set as `n_td = 2**16`
    N = int(round(math.log2(n_td), 1))
    sw = 5000
    aq = (n_td / sw)
    ppm_ref = 4.0
    B0 = 500
    t2 = 0.4
    phase = 0
    apodize = False

    ######## Initialize Hamiltonian ########
    gt_h_mat = d[0] - torch.diag(torch.full((d[0].shape[0], ), ppm_ref))  # ground truth (offset subtracted)
    gt_h_mat = gt_h_mat.to(device)

    h_mat = np.random.rand(n_spins + sum(range(1, n_spins)), )
    h_mat = torch.from_numpy(h_mat).to(device)
    h_mat.requires_grad = True

    ts = torch.linspace(0, aq, n_td, device=device)  # time points

    ############### rho state ##############
    rho = IHz
    U90y = torch.matrix_exp(-1j * torch.pi / 2 * IHy)
    rho = torch.matmul(U90y, torch.matmul(rho, U90y.T.conj()))

    ############### FID - GT ###############
    logger.info("Calculating GT FID")
    gt_fid = explicit_exponentiation(
        hamiltonian=calc_hamiltonian(h_mat=gt_h_mat, B0=B0, Ix=Ix, Iy=Iy, Iz=Iz),
        rho=rho, op=OP, ts=ts, dt=ts[1]-ts[0], n_td=n_td, t2=t2, apodize=apodize,
    )
    logger.info("Completed GT FID calculation")

    x_lbfgs = h_mat
    logger.info(f"Initial starting values: {x_lbfgs}")

    lbfgs = optim.LBFGS(
        [x_lbfgs],
        history_size=100, 
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-8,
        line_search_fn="strong_wolfe"
    )

    history_lbfgs = []
    for n in tqdm(range(N)):

        s = time.time()
        length = 2 ** (n + 1)
        if length > n_td:
            length = n_td

        # L-BFGS
        def closure():
            lbfgs.zero_grad()
            objective = objective_function(x_lbfgs, length)
            objective.backward()
            return objective

        curr_loss = lbfgs.step(closure)
        history_lbfgs.append(curr_loss.detach().item())

        logger.info(x_lbfgs.detach())
        logger.info(f"FID Length: {length}")
        logger.info(f"Loss: {curr_loss}")
        logger.info(f"Time: {time.time() - s} seconds")
        logger.info("-"*50)

    logger.info(f"Final fitted parameters: {x_lbfgs.detach()}")
