"""The Abstract Base Class for Theia models."""

import abc
import pathlib
import typing

import numpy

from .transformer import Transformer


class Theia(abc.ABC):
    """The base model for Theia.

    All subclasses must implement `save`, `load`, `fit` and `transform`.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        channel_overlap: int,
        kernel_size: int,
        alpha: float,
        beta: float,
    ) -> None:
        """Base class for a Theia model.

        Args:
            num_channels: number of channels in each image.
            channel_overlap: Maximum number of adjacent channels to consider for
             bleed-through removal.
            kernel_size: Side-length of square kernel (convolutional) to use for
             estimating bleed-through.
            alpha: Relative size of l1-penalty in the LASSO loss.
            beta: Relative weighting of target channel in interaction terms.
        """
        self._num_channels = num_channels
        self._channel_overlap = channel_overlap
        self._kernel_size = kernel_size
        self._alpha = alpha
        self._beta = beta

        self._contribution_kernels: dict[tuple[int, int], numpy.ndarray] = {}
        self._interactions_kernels: dict[tuple[int, int], numpy.ndarray] = {}

    @property
    def num_channels(self) -> int:
        """The number of channels in each image."""
        return self._num_channels

    @property
    def channel_overlap(self) -> int:
        """The number of adjacent channels to use for bleed-through estimation."""
        return self._channel_overlap

    @abc.abstractmethod
    def fit_theia(
        self,
        *args: list[typing.Any],
        **kwargs: dict[str, typing.Any],
    ) -> None:
        """Fit Theia to a multi-channel image."""
        pass

    @abc.abstractmethod
    def save(self, path: pathlib.Path) -> None:
        """Save the model to the given `path`."""
        pass

    @staticmethod
    @abc.abstractmethod
    def load(path: pathlib.Path) -> "Theia":
        """Load the model from the given `path`."""
        pass

    @property
    def contribution_kernels(self) -> dict[tuple[int, int], numpy.ndarray]:
        """Return the fitted contribution kernels."""
        if len(self._contribution_kernels) == 0:
            message = "Please call `fit` before using this property."
            raise ValueError(message)
        return self._contribution_kernels

    @property
    def interactions_kernels(self) -> dict[tuple[int, int], numpy.ndarray]:
        """Return the fitted interaction kernels."""
        if len(self._interactions_kernels) == 0:
            message = "Please call `fit` before using this property."
            raise ValueError(message)
        return self._interactions_kernels

    @property
    def transformer(self) -> "Transformer":
        """Provide a thread-safe transformer to apply bleed-through correction."""
        return Transformer(
            num_channels=self.num_channels,
            channel_overlap=self.channel_overlap,
            beta=self._beta,
            contribution_kernels=self.contribution_kernels,
            interactions_kernels=self.interactions_kernels,
        )
