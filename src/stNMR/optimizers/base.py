from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):
    """
    Base class for optimizers.
    """
    @abstractmethod
    def optimize(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
