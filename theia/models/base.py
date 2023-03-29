"""The Abstract Base Class for Theia models."""

import abc
import pathlib
import typing

import numpy

from ..utils import constants


class Theia(abc.ABC):
    """The base model for Theia.

    All subclasses must implement `save`, `load`, `fit` and `transform`.
    """

    def __init__(
        self,  # noqa
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

        self._num_pixels = max(
            constants.MAX_DATA_SIZE
            // (4 * (kernel_size**2) * self._channel_overlap * 2),
            constants.MIN_DATA_SIZE,
        )

        self._contribution_kernels: dict[tuple[int, int], numpy.ndarray] = {}
        self._interaction_kernels: dict[tuple[int, int], numpy.ndarray] = {}

    @abc.abstractmethod
    def fit(
        self,  # noqa
        *args: list[typing.Any],
        **kwargs: dict[str, typing.Any],
    ) -> None:
        """Fit Theia to a multi-channel image."""
        pass

    @abc.abstractmethod
    def transform(
        self,  # noqa
        image: numpy.ndarray,
        *,
        remove_interactions: bool = False,
    ) -> numpy.ndarray:
        """Transform and return a multichannel image.

        Args:
            image: A multichannel image with shape (H, W, C) where H is the
             height, W is the width and C is the number of channels.
            remove_interactions: Whether to remove interaction components.

        Returns:
            The corrected image.
        """
        pass

    @abc.abstractmethod
    def save(self, path: pathlib.Path) -> None:  # noqa
        """Save the model to the given `path`."""
        pass

    @staticmethod
    @abc.abstractmethod
    def load(path: pathlib.Path) -> "Theia":
        """Load the model from the given `path`."""
        pass

    @property
    def contribution_kernels(self) -> dict[tuple[int, int], numpy.ndarray]:  # noqa
        """Return the fitted contribution kernels."""
        if len(self._contribution_kernels) == 0:
            message = "Please call `fit` before using this property."
            raise ValueError(message)
        return self._contribution_kernels

    @property
    def interaction_kernels(self) -> dict[tuple[int, int], numpy.ndarray]:  # noqa
        """Return the fitted interaction kernels."""
        if len(self._interaction_kernels) == 0:
            message = "Please call `fit` before using this property."
            raise ValueError(message)
        return self._interaction_kernels
