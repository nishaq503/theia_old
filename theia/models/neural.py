"""Theia implemented using a CNN in Tensorflow."""
import pathlib
import typing

import numpy
import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics

from .base import Theia


class Neural(Theia, keras.Model):  # type: ignore
    """Theia implemented our a Neural Network Architecture.

    The architecture is a generalization of Siamese Networks, and performs the
     equivalent of LASSO regression.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        channel_overlap: int,
        kernel_size: int,
        alpha: float,
        beta: float,
        tile_size: int,
    ) -> None:
        """Initialize and build the Network.

        Args:
            num_channels: Number of channels in each input image.
            channel_overlap: Maximum number of adjacent channels to consider for
             bleed-through removal.
            kernel_size: Side-length of square kernel (convolutional) to use for
             estimating bleed-through.
            alpha: Relative size of l1-penalty in the LASSO loss.
            beta: Relative weighting of target channel in interaction terms.
            tile_size: Side-length of square tiles to use as inputs to the
             network.
        """
        keras.Model.__init__(self)
        Theia.__init__(
            self,
            num_channels=num_channels,
            channel_overlap=channel_overlap,
            kernel_size=kernel_size,
            alpha=alpha,
            beta=beta,
        )

        self._callbacks: list[callbacks.Callback] = []

        self._tile_shape = (tile_size, tile_size, 1)
        self._sources = [
            layers.Input(self._tile_shape, name=f"source_{i}")
            for i in range(num_channels)
        ]

        self._kernel_names = []
        self._num_kernels: list[int] = []
        self._contributions: list[list[_TheiaConv]] = []
        self._aggregators: list[_Aggregator] = []

        for source_index, source in enumerate(self._sources):
            contributions: list[_TheiaConv] = []

            for offset in range(-self._channel_overlap, self._channel_overlap + 1):
                if offset == 0:
                    continue

                neighbor_index = source_index + offset
                if neighbor_index not in range(self._num_channels):
                    continue

                neighbor = self._sources[neighbor_index]
                contribution = _TheiaConv(
                    name=f"contribution_{source_index}_{neighbor_index}",
                    kernel_size=self._kernel_size,
                )(neighbor)
                self._kernel_names.append((True, source_index, neighbor_index))
                contributions.append(contribution)

                interaction_term = tensorflow.pow(neighbor, self._beta)
                interaction_term = tensorflow.multiply(interaction_term, source)
                interaction_term = tensorflow.pow(
                    interaction_term,
                    1 / (1 + self._beta),
                )
                interaction = _TheiaConv(
                    name=f"interactions_{source_index}_{neighbor_index}",
                    kernel_size=self._kernel_size,
                )(interaction_term)
                self._kernel_names.append((False, source_index, neighbor_index))
                contributions.append(interaction)

            self._num_kernels.append(len(contributions))
            self._contributions.append(contributions)
            self._aggregators.append(
                _Aggregator(name=f"aggregation_{source_index}")(
                    [source, *contributions],
                ),
            )

        self.inputs = self._sources
        self.outputs = self._aggregators

        self._model = keras.Model(
            inputs=self.inputs,
            outputs=self.outputs,
        )
        self.loss_tracker = metrics.Mean(name="theia_loss")

    @property
    def tile_size(self) -> int:
        """Side-length of square tiles to use as inputs to the network."""
        return self._tile_shape[0]

    def fit(
        self,
        *args: list[typing.Any],
        **kwargs: dict[str, typing.Any],
    ) -> None:
        """Fit Theia to a multi-channel image."""
        pass

    def early_stopping(
        self,
        *,
        min_delta: float,
        patience: int,
        verbose: int,
        restore_best_weights: bool,
    ) -> None:
        """Set up an EarlyStopping callback for the model.

        Args:
            min_delta: Minimum change in the monitored quantity to qualify as an
             improvement, i.e. an absolute change of less than min_delta, will
             count as no improvement.
            patience: Number of epochs with no improvement after which training
             will be stopped.
            verbose: verbosity mode.
            restore_best_weights: Whether to restore model weights from the
             epoch with the best value of the loss function.
        """
        self._callbacks.append(
            callbacks.EarlyStopping(
                monitor="theia_loss",
                min_delta=min_delta,
                patience=patience,
                verbose=verbose,
                mode="min",
                restore_best_weights=restore_best_weights,
            ),
        )

    def add_callback(self, cb: callbacks.Callback) -> None:  # noqa
        self._callbacks.append(cb)

    def transform(
        self,
        image: numpy.ndarray,
        *,
        remove_interactions: bool = False,
    ) -> numpy.ndarray:
        """Correct and return the input image.

        Args:
            image: to be corrected for bleed-through.
            remove_interactions: whether to remove interaction terms.

        Returns:
            The corrected image.
        """
        pass

    def save(self, path: pathlib.Path) -> None:
        """Save the model to the given `path`."""
        raise NotImplementedError

    @staticmethod
    def load(path: pathlib.Path) -> "Neural":
        """Load the model from the given `path`."""
        raise NotImplementedError

    def call(
        self,
        inputs: list[tensorflow.Tensor],
        training: typing.Optional[typing.Union[bool, tensorflow.Tensor]] = None,
        mask: typing.Optional[
            typing.Union[tensorflow.Tensor, list[tensorflow.Tensor]]
        ] = None,
    ) -> list[tuple[tensorflow.Tensor, tensorflow.Tensor]]:
        """Call the model."""
        return self._model(inputs)  # type: ignore

    def get_config(self) -> None:  # noqa
        raise NotImplementedError

    def _serialize_to_tensors(self) -> None:
        raise NotImplementedError

    def _restore_from_tensors(
        self,
        restored_tensors: dict[str, tensorflow.Tensor],
    ) -> None:
        raise NotImplementedError


class _TheiaConv(layers.Conv2D):  # type: ignore
    def __init__(
        self,
        *,
        name: str,
        kernel_size: int,
    ) -> None:
        super().__init__(
            name=name,
            filters=1,
            padding="same",
            kernel_size=kernel_size,
            use_bias=False,
            kernel_constraint=tensorflow.keras.constraints.NonNeg(),
        )

    def call(
        self,
        bleedthrough_term: tensorflow.Tensor,
    ) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
        conv = super().call(bleedthrough_term)
        return conv, self.kernel

    def _serialize_to_tensors(self) -> None:
        raise NotImplementedError

    def _restore_from_tensors(
        self,
        restored_tensors: dict[str, tensorflow.Tensor],
    ) -> None:
        raise NotImplementedError


class _Aggregator(layers.Layer):  # type: ignore
    def __init__(self, *, name: str) -> None:
        super().__init__(name=name)

    def get_config(self) -> typing.Any:  # noqa
        return super().get_config()

    # noinspection PyMethodOverriding
    def call(
        self,
        inputs: list[tensorflow.Tensor],
    ) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
        [source, *contributors] = inputs

        kernels = tensorflow.stack([k for _, k in contributors])
        filtered = [f for f, _ in contributors]

        bleed_through = tensorflow.add_n(filtered)

        corrected = tensorflow.subtract(source, bleed_through)
        return corrected, kernels

    def _serialize_to_tensors(self) -> None:
        raise NotImplementedError

    def _restore_from_tensors(
        self,
        restored_tensors: dict[str, tensorflow.Tensor],
    ) -> None:
        raise NotImplementedError
