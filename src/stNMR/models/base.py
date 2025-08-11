from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """
    Base class for models.
    """

    @abstractmethod
    def forward(self, hamiltonian):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
