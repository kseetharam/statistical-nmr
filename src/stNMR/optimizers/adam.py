from typing import Literal, Optional, Callable, List, Tuple

from stNMR.constants import array_like
from stNMR.optimizers.base import Optimizer


class ADAM(Optimizer):
    """
    Base class for ADAM optimizers with optional learning rate scheduling.
    Now includes tracking of learning and learning rate history.
    """
    def __init__(
        self,
        backend: Literal["torch", "optax"] = "torch",
        learning_rate: float = 1e-3,
        lr_scheduler: Optional[Callable[[int], float]] = None,
    ):
        self.backend = backend
        self.lr = learning_rate
        self.lr_scheduler = lr_scheduler

        if backend == "torch":
            from torch.optim import Adam as TorchAdam
            self._opt_class = TorchAdam
            self._fit_func = self._fit_torch
        elif backend == "optax":
            from optax import adam as JaxAdam
            from optax import apply_updates
            self._opt_class = JaxAdam
            self._apply_updates = apply_updates
            self._fit_func = self._fit_jax
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def optimize(self, x: array_like, objective_function, y, *args) -> Tuple[array_like, float, List[float], List[float]]:
        return self._fit_func(x, objective_function, y, *args)

    def _fit_torch(self, x, objective_function, y, *args):
        """
        Fit using PyTorch's ADAM with loss and learning rate history.
        """
        optimizer = self._opt_class([x], lr=self.lr)
        loss_history = []
        lr_history = []

        for step in range(1000):
            if self.lr_scheduler:
                lr = self.lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = self.lr

            optimizer.zero_grad()
            loss = objective_function(x, y, *args)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            lr_history.append(lr)

        return x, loss.item(), loss_history, lr_history

    def _fit_jax(self, x, objective_function, y):
        """
        Fit using JAX's ADAM with loss and learning rate history.
        """
        def get_optimizer(step):
            lr = self.lr_scheduler(step) if self.lr_scheduler else self.lr
            return self._opt_class(learning_rate=lr), lr

        optimizer, lr = get_optimizer(0)
        state = optimizer.init(x)

        loss_history = []
        lr_history = []

        for step in range(100):
            if self.lr_scheduler:
                optimizer, lr = get_optimizer(step)

            loss, grads = objective_function(x, y)
            updates, state = optimizer.update(grads, state)
            x = self._apply_updates(x, updates)

            loss_history.append(float(loss))
            lr_history.append(lr)

        return x, float(loss), loss_history, lr_history
