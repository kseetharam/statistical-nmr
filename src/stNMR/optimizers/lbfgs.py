from typing import Literal, List, Tuple
import logging

from stNMR.constants import array_like
from stNMR.optimizers.base import Optimizer


class BFGS(Optimizer):
    """
    Base class for L-BFGS optimizers.
    Returns x, loss, loss_history, and lr_history (None for lr_history in non-lr schedulers).
    """
    def __init__(self, backend: Literal["numpy", "torch", "optax"] = "numpy"):
        self.backend = backend

        if backend == "numpy":
            from scipy.optimize import minimize as ScipyLBFGS
            self._opt = ScipyLBFGS
            self._fit_func = self._fit_scipy

        elif backend == "torch":
            from torch.optim import LBFGS as TorchLBFGS
            self._opt = TorchLBFGS
            self._fit_func = self._fit_torch

        elif backend == "optax":
            raise NotImplementedError("Optax L-BFGS has not been implemented yet!")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self._logger = logging.getLogger("stNMR")

    def optimize(self, x: array_like, objective_function, y, *args) -> Tuple[array_like, float, List[float], List[float]]:
        return self._fit_func(x, objective_function, y, *args)

    def _fit_scipy(self, x, objective_function, y=None, *args):
        """
        Fit using SciPy's L-BFGS.
        """



        loss_history = []

        def wrapped_func(x_, *args_):
            loss = objective_function(x_, y, *args_)
            loss_history.append(float(loss))
            return loss

        result = self._opt(
            fun=wrapped_func,
            x0=x,
            args=args,
            method='L-BFGS-B',
            options={"disp": True, "gtol": 1e-8, "maxiter": 1000, "ftol": 1e-8}
        )

        return result.x, result.fun, loss_history, []

    def _fit_torch(self, x, objective_function, y, *args):
        """
        Fit using PyTorch's L-BFGS.
        """
        optimizer = self._opt([x], lr=1.0)
        loss_history = []

        def closure():
            optimizer.zero_grad()
            loss = objective_function(x, y, *args)
            loss.backward()
            loss_history.append(loss.item())
            return loss

        for _ in range(100):
            optimizer.step(closure)

        final_loss = loss_history[-1] if loss_history else float('nan')
        return x.detach().cpu().numpy(), final_loss, loss_history, []
