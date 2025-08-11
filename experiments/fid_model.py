from stNMR.models.fid import ExplicitFIDModel
from stNMR.optimizers import BFGS, ADAM

from stNMR.dataset import GissmoDataset
from stNMR.numpy.nmr import generate_spin_operators

import torch
from pathlib import Path
from scipy.linalg import expm


if __name__ == "__main__":

    import logging
    import numpy as np

    np.random.seed(42)

    # Setup logger
    logger = logging.getLogger("FIDModel")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    n_spins = 6
    Ix, Iy, Iz = generate_spin_operators(n_spins=n_spins)
    IHx, IHy, IHz = Ix[:, :, :].sum(2), Iy[:, :, :].sum(2), Iz[:, :, :].sum(2)
    OP = IHx + 1j * IHy

    # Load dataset
    dataset = GissmoDataset(n_spins=n_spins, return_type="numpy")
    path = Path("data/gissmo/data/bmse000077.csv")  # 9 spins: 43, 7 spins: 29; 6 spins: 77; 5 spins: 39; 4 spins: 104; 3 spins: 7
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

    ts = np.linspace(0, aq, n_td)
    dt = ts[1] - ts[0]

    rho = IHz
    U90y = expm(-1j * np.pi / 2 * IHy)
    rho = np.dot(U90y, np.dot(rho, U90y.T.conj()))

    gt_h_mat = d[0] - np.diag(np.full(d[0].shape[0], ppm_ref))
    chemical_shifts = np.diag(gt_h_mat)
    row_indices, col_indices = np.triu_indices(gt_h_mat.shape[0], k=1)
    j_couplings = gt_h_mat[row_indices, col_indices]
    gt_params = np.concatenate((chemical_shifts, j_couplings))

    x = np.random.rand(n_spins + sum(range(1, n_spins)), )

    max_steps = 100  # Total number of training steps
    initial_lr = 1e-2
    min_lr = 1e-4  # New minimum learning rate

    def cosine_scheduler(step: int) -> float:
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / max_steps))
        lr = min_lr + (initial_lr - min_lr) * cosine_decay
        return lr

    adam_opt = ADAM(
        backend="torch",
        learning_rate=initial_lr,
        lr_scheduler=cosine_scheduler
    )

    bfgs_opt = BFGS(
        backend="numpy",
    )

    # Initialize model and optimizer
    model = ExplicitFIDModel(
        n_spins=n_spins,
        B0=500.0,
        sw=sw,
        rho=rho,
        t2=t2,
        n_td=n_td,
        apodize=apodize,
        backend="numpy",
        device="cuda:0"
    )
    fitted_params, loss, loss_history, lr_history = model.fit(x, gt_params, bootstrap=True, opt=bfgs_opt)

    np.set_printoptions(suppress=True)
    fitted_params = fitted_params.detach().cpu().numpy() if isinstance(fitted_params, torch.Tensor) else fitted_params

    print(fitted_params.round(3))
    print(fitted_params.shape)

    from matplotlib import pyplot as plt

    plt.plot(lr_history, label='LR History')
    plt.savefig("figures/lr_history.png")
