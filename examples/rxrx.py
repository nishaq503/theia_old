"""Theia for the RxRx1 dataset."""
import concurrent.futures
import pathlib
import random

import imageio.v3 as imageio
import numpy
import tqdm

import theia


def train_theia(i_experiment: int, i_plate: int) -> None:
    """Train Theia on a single plate.

    Args:
        i_experiment: index of experiment
        i_plate: index of plage
    """
    data_root, images_dir, experiments, plates = _get_paths()
    train_images, valid_images = _read_images(
        images_dir=images_dir,
        experiment=experiments[i_experiment],
        plate=plates[i_plate],
    )
    train_gen = theia.data.TileGenerator(images=train_images, tile_size=512)
    valid_gen = theia.data.TileGenerator(images=valid_images, tile_size=512)

    model = theia.models.Neural(
        num_channels=6,
        channel_overlap=2,
        kernel_size=3,
        alpha=1,
        beta=1,
        tile_size=512,
    )
    model.early_stopping(
        min_delta=1e-3,
        patience=4,
        verbose=1,
        restore_best_weights=True,
    )
    model.compile(
        optimizer="adam",
    )
    model.build()
    model.fit_theia(
        train_gen=train_gen,
        valid_gen=valid_gen,
        epochs=128,
        verbose=1,
    )


def _get_paths() -> tuple[pathlib.Path, pathlib.Path, list[str], list[str]]:
    data_root = pathlib.Path(__file__).parents[3].resolve().joinpath("data", "rxrx1")

    images_dir = data_root.joinpath("images")

    experiments = [path.name for path in sorted(images_dir.iterdir()) if path.is_dir()]

    plates = sorted(
        {
            plate_path.name
            for exp in experiments
            for plate_path in images_dir.joinpath(exp).iterdir()
            if plate_path.is_dir()
        },
    )

    return data_root, images_dir, experiments, plates


def _read_images(
    images_dir: pathlib.Path,
    experiment: str,
    plate: str,
) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
    plate_dir = images_dir.joinpath(experiment, plate)
    well_names = sorted(
        {
            "_".join(path.name.split("_")[:2])
            for path in sorted(plate_dir.iterdir())
            if path.name.endswith(".png")
        },
    )
    random.shuffle(well_names)

    train_images = _read_batch(plate_dir, well_names[:512])
    valid_images = _read_batch(plate_dir, well_names[512:])

    return train_images, valid_images


def _read_batch(plate_dir: pathlib.Path, well_names: list[str]) -> list[numpy.ndarray]:
    channel_names = [f"w{i + 1}" for i in range(6)]

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for w in well_names:
            futures.append(
                executor.submit(
                    _read_fov,
                    [plate_dir.joinpath(f"{w}_{c}.png") for c in channel_names],
                ),
            )

    images = []
    for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(well_names)):
        images.append(f.result())

    return images


def _read_fov(paths: list[pathlib.Path]) -> numpy.ndarray:
    channels = [
        numpy.asarray(imageio.imread(path), dtype=numpy.float32) for path in paths
    ]
    return numpy.stack(channels, axis=2)


if __name__ == "__main__":
    train_theia(0, 0)
