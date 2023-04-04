"""A Tiled Image Data Generator for Theia."""
import random

import numpy
import tensorflow
from tensorflow.python.keras.utils import data_utils as keras_utils

from .utils import constants

__all__ = ["TileGenerator"]


class TileGenerator(keras_utils.Sequence):  # type: ignore
    """Take images as numpy arrays and present tiles to the Neural Network."""

    def __init__(
        self,
        *,
        images: list[numpy.ndarray],
        tile_size: int,
        shuffle: bool = True,
        normalize: bool = True,
    ) -> None:
        """Create a Tile Generator from the given images.

        Let `n` be the number of images, `w` by the width, `h` be the height and
         `c` be the number of channels. `images` must have `n` items and each
         should be an image with shape `(h, w, c)`.

        Args:
            images: to correct for bleed-through.
            tile_size: If `w` or `h` is larger than `tile_size` then the images
             will be tiled. All tiles will be padded to a square of side-length
             `tile_size`.
            shuffle: Whether to shuffle tiles after each epoch.
            normalize: Whether to normalize the image intensities to a [0, 1]
             range.
        """
        [h, w, c] = list(map(int, images[0].shape))
        h_pad, self._h_tiles = _calculate_padding(h, tile_size)
        w_pad, self._w_tiles = _calculate_padding(w, tile_size)
        self._paddings = tensorflow.constant([[0, h_pad], [0, w_pad], [0, 0]])

        if normalize:
            images = list(map(_normalize_image, images))
        else:
            images = [img / (numpy.max(img) + constants.EPSILON) for img in images]
        self._images = [
            tensorflow.convert_to_tensor(image, dtype=tensorflow.float32)
            for image in images
        ]

        self._tile_shape = (tile_size, tile_size, c)
        self._shuffle = shuffle

        self.on_epoch_end()

    @property
    def num_channels(self) -> int:
        """The number of channels in each image."""
        return self._tile_shape[-1]

    @property
    def tile_shape(self) -> tuple[int, int, int]:
        """The shape of tiles: `(h, w, c)`."""
        return self._tile_shape

    def on_epoch_end(self) -> None:
        """Optionally shuffle images after each epoch."""
        if self._shuffle:
            random.shuffle(self._images)

    def __getitem__(self, index: int) -> list[tensorflow.Tensor]:
        """Return a batch of tiles.

        Args:
            index: of the image from which to make tiles.

        Returns:
            A list with `c` `Tensors`, each with shape
             `(num_tiles, tile_size, tile_size, 1)`
        """
        image = tensorflow.pad(self._images[index], self._paddings)
        tiled_image: tensorflow.Tensor = tensorflow.reshape(
            image,
            (self._h_tiles, self._w_tiles, *self._tile_shape),
        )
        tiled_image = tensorflow.reshape(
            tiled_image,
            (-1, *self._tile_shape),
        )
        batch: list[tensorflow.Tensor] = tensorflow.unstack(tiled_image, axis=-1)
        return batch

    def __len__(self) -> int:
        """Number of images."""
        return len(self._images)


def _normalize_image(image: numpy.ndarray) -> numpy.ndarray:
    min_val = numpy.min(image, axis=(0, 1))
    max_val = numpy.max(image, axis=(0, 1))
    image = image.astype(numpy.float32) - min_val
    image /= max_val - min_val + constants.EPSILON
    return image


def _calculate_padding(stop: int, step: int) -> tuple[int, int]:
    if stop <= step:
        return stop - step, 1

    padding = stop % step
    n_tiles = stop // step
    if padding == 0:
        return padding, n_tiles

    return padding, n_tiles + 1
