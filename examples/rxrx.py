"""Theia for the RxRx1 dataset."""
import concurrent.futures
import pathlib
import random

import imageio.v3 as imageio
import numpy
import tqdm

import theia


def run_theia(i_experiment: int, i_plate: int) -> None:
    """Run Theia on a single plate.

    Args:
        i_experiment: index of experiment
        i_plate: index of plate

    Returns:
        The trained model.
    """
    data_root, images_dir, experiments, plates = _get_paths()
    experiment = experiments[i_experiment]
    plate = plates[i_plate]

    train_images_, valid_images_ = _read_images(images_dir, experiment, plate)
    train_wells, valid_wells = list(train_images_.keys()), list(valid_images_.keys())

    train_images = list(train_images_.values())
    valid_images = list(valid_images_.values())
    model = _train_theia(
        train_gen=theia.data.TileGenerator(images=train_images, tile_size=512),
        valid_gen=theia.data.TileGenerator(images=valid_images, tile_size=512),
    )

    out_dir = data_root.joinpath("theia_out", experiment, plate)
    out_dir.mkdir(exist_ok=True, parents=True)

    bleedthrough_dir = out_dir.joinpath("bleed_through_components")
    bleedthrough_dir.mkdir(exist_ok=True)

    interaction_dir = out_dir.joinpath("interaction_components")
    interaction_dir.mkdir(exist_ok=True)

    images_dir = out_dir.joinpath("images")
    images_dir.mkdir(exist_ok=True)

    for well_name, raw_image in tqdm.tqdm(
        zip(train_wells + valid_wells, train_images + valid_images),
        total=len(train_wells) + len(valid_wells),
    ):
        image = _normalize_image(raw_image)

        bleedthrough = model.bleedthrough_component(image)
        interaction = model.interaction_component(image)
        corrected = numpy.clip(image - bleedthrough, a_min=0, a_max=1.0)

        for c in range(6):
            name = f"{well_name}_w{c + 1}.tiff"
            imageio.imwrite(bleedthrough_dir.joinpath(name), bleedthrough[:, :, c])
            imageio.imwrite(interaction_dir.joinpath(name), interaction[:, :, c])
            imageio.imwrite(images_dir.joinpath(name), corrected[:, :, c])


def _normalize_image(image: numpy.ndarray) -> numpy.ndarray:
    img_min = numpy.min(image, axis=(0, 1))
    img_max = numpy.max(image, axis=(0, 1))
    image -= -img_min
    image /= img_max - img_min + theia.utils.constants.EPSILON
    return image


def _img_to_u8(image: numpy.ndarray) -> numpy.ndarray:
    return (image * 255.0).astype(numpy.uint8)


def _train_theia(
    train_gen: theia.data.TileGenerator,
    valid_gen: theia.data.TileGenerator,
) -> theia.Neural:
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
    return model


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
) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
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


def _read_batch(
    plate_dir: pathlib.Path,
    well_names: list[str],
) -> dict[str, numpy.ndarray]:
    channel_names = [f"w{i + 1}" for i in range(6)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures: list[concurrent.futures.Future[tuple[str, numpy.ndarray]]] = []
        for w in well_names:
            futures.append(
                executor.submit(
                    _read_fov,
                    w,
                    [plate_dir.joinpath(f"{w}_{c}.png") for c in channel_names],
                ),
            )

        images = {}
        for f in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(well_names),
        ):
            w, image = f.result()
            images[w] = image

    return images


def _read_fov(well_name: str, paths: list[pathlib.Path]) -> tuple[str, numpy.ndarray]:
    channels = [
        numpy.asarray(imageio.imread(path), dtype=numpy.float32) for path in paths
    ]
    return well_name, numpy.stack(channels, axis=2)


if __name__ == "__main__":
    run_theia(0, 0)
