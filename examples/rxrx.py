"""Theia for the RxRx1 dataset."""
import concurrent.futures
import pathlib
import random

import imageio.v3 as imageio
import numpy
import streamlit
import tqdm
from matplotlib import pyplot

import theia


def run_theia(i_experiment: int, i_plate: int) -> None:
    """Run Theia on a single plate.

    Args:
        i_experiment: index of experiment
        i_plate: index of plate

    Returns:
        The trained model.
    """
    data_root, images_dir, experiments, plates = _get_inp_paths()
    experiment = experiments[i_experiment]
    plate = plates[i_plate]

    train_images_, valid_images_ = _read_images(images_dir, experiment, plate)
    train_wells, valid_wells = list(train_images_.keys()), list(valid_images_.keys())

    train_images = list(train_images_.values())
    valid_images = list(valid_images_.values())
    transformer = _train_theia(
        train_gen=theia.data.TileGenerator(
            images=train_images,
            tile_size=512,
            normalize=False,
        ),
        valid_gen=theia.data.TileGenerator(
            images=valid_images,
            tile_size=512,
            normalize=False,
        ),
    )

    bleedthrough_dir, interaction_dir, corrected_dir = _get_out_paths(
        data_root,
        experiment,
        plate,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for well_name, raw_image in zip(
            train_wells + valid_wells,
            train_images + valid_images,
        ):
            futures.append(
                executor.submit(
                    _save_one,
                    transformer,
                    raw_image,
                    well_name,
                    (bleedthrough_dir, interaction_dir, corrected_dir),
                ),
            )

        for f in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
        ):
            f.result()


def _save_one(
    transformer: theia.Transformer,
    raw_image: numpy.ndarray,
    well_name: str,
    paths: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
) -> None:
    bleedthrough_dir, interaction_dir, corrected_dir = paths

    # image = _normalize_image(raw_image)
    image = raw_image / (numpy.max(raw_image) + theia.constants.EPSILON)

    bleedthrough = transformer.bleedthrough_component(image)
    interaction = transformer.interaction_component(image)
    corrected = numpy.clip(image - bleedthrough, a_min=0, a_max=numpy.max(image))

    for c in range(6):
        name = f"{well_name}_w{c + 1}.tiff"
        imageio.imwrite(bleedthrough_dir.joinpath(name), bleedthrough[:, :, c])
        imageio.imwrite(interaction_dir.joinpath(name), interaction[:, :, c])
        imageio.imwrite(corrected_dir.joinpath(name), corrected[:, :, c])


def _normalize_image(image: numpy.ndarray) -> numpy.ndarray:
    img_min = numpy.min(image, axis=(0, 1))
    img_max = numpy.max(image, axis=(0, 1))
    image -= -img_min
    image /= img_max - img_min + theia.constants.EPSILON
    return image


def _img_to_u8(image: numpy.ndarray) -> numpy.ndarray:
    return (image * 255.0).astype(numpy.uint8)


def _train_theia(
    train_gen: theia.data.TileGenerator,
    valid_gen: theia.data.TileGenerator,
) -> theia.Transformer:
    model = theia.models.Neural(
        num_channels=6,
        channel_overlap=5,
        kernel_size=5,
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
    return model.transformer


def _get_inp_paths() -> tuple[pathlib.Path, pathlib.Path, list[str], list[str]]:
    data_root = pathlib.Path(__file__).resolve().parents[3].joinpath("data", "rxrx1")

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


def _get_out_paths(
    data_root: pathlib.Path,
    experiment: str,
    plate: str,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    out_dir = data_root.joinpath("theia_out", experiment, plate)
    out_dir.mkdir(exist_ok=True, parents=True)

    bleedthrough_dir = out_dir.joinpath("bleed_through_components")
    bleedthrough_dir.mkdir(exist_ok=True)

    interaction_dir = out_dir.joinpath("interaction_components")
    interaction_dir.mkdir(exist_ok=True)

    corrected_dir = out_dir.joinpath("images")
    corrected_dir.mkdir(exist_ok=True)

    return bleedthrough_dir, interaction_dir, corrected_dir


def _read_images(
    images_dir: pathlib.Path,
    experiment: str,
    plate: str,
) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
    plate_dir = images_dir.joinpath(experiment, plate)

    well_names = _get_well_names(plate_dir)
    random.shuffle(well_names)

    train_images = _read_batch(plate_dir, well_names[:512])
    valid_images = _read_batch(plate_dir, well_names[512:])

    return train_images, valid_images


def _get_well_names(plate_dir: pathlib.Path) -> list[str]:
    return sorted(
        {
            "_".join(path.name.split("_")[:2])
            for path in sorted(plate_dir.iterdir())
            if path.name.endswith(".png")
        },
    )


def _get_channel_paths(
    plate_dir: pathlib.Path,
    well_name: str,
    ext: str,
) -> list[pathlib.Path]:
    return [plate_dir.joinpath(f"{well_name}_w{c + 1}.{ext}") for c in range(6)]


def _read_batch(
    plate_dir: pathlib.Path,
    well_names: list[str],
) -> dict[str, numpy.ndarray]:
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures: list[concurrent.futures.Future[tuple[str, numpy.ndarray]]] = []
        for w in well_names:
            futures.append(
                executor.submit(
                    _read_fov,
                    w,
                    _get_channel_paths(plate_dir, w, "png"),
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
    return well_name, numpy.stack(list(map(_read_single_channel, paths)), axis=2)


def _read_single_channel(path: pathlib.Path) -> numpy.ndarray:
    return numpy.asarray(imageio.imread(path), dtype=numpy.float32)


# noinspection DuplicatedCode
def app() -> None:  # noqa
    """Run the Streamlit app."""
    streamlit.title("Theia: RxRx1 Plate")

    streamlit.write("This dataset contains 6-channel images. The fluorophores are:")
    channel_names = ["Hoechst", "ConA", "Phalloidin", "Syto14", "MitoTracker", "WGA"]
    _ = [streamlit.write(f"    {i}. {n}") for i, n in enumerate(channel_names, start=1)]

    data_root, images_dir, experiments, plates = _get_inp_paths()
    experiment, plate = experiments[0], plates[0]

    bleedthrough_dir, interaction_dir, corrected_dir = _get_out_paths(
        data_root,
        experiment,
        plate,
    )
    plate_dir = images_dir.joinpath(experiment, plate)

    well_row: str = streamlit.selectbox("Well Row:", list("BCDEFGHIJKLMNO"))
    well_col: str = streamlit.selectbox(
        "Well Column:",
        [f"{c:02d}" for c in range(2, 24)],
    )
    well_site: str = streamlit.selectbox("Imaging Site:", ["s1", "s2"])

    well_name = f"{well_row}{well_col}_{well_site}"

    original = list(
        map(
            _read_single_channel,
            _get_channel_paths(plate_dir, well_name, "png"),
        ),
    )
    bleedthrough = list(
        map(
            _read_single_channel,
            _get_channel_paths(bleedthrough_dir, well_name, "tiff"),
        ),
    )
    list(
        map(
            _read_single_channel,
            _get_channel_paths(interaction_dir, well_name, "tiff"),
        ),
    )
    corrected = list(
        map(
            _read_single_channel,
            _get_channel_paths(corrected_dir, well_name, "tiff"),
        ),
    )

    fig: pyplot.Figure
    ax: pyplot.Axes

    streamlit.write("The original six channels")
    fig, axs = pyplot.subplots(2, 3, figsize=(12, 8), dpi=256)
    for i, (ax, img, c) in enumerate(zip(axs.flat, original, channel_names), start=1):
        ax.imshow(_red_channel(img / numpy.max(img) + theia.constants.EPSILON))
        ax.set_title(f"Channel {i}: {c}")
        ax.axis("off")
    streamlit.pyplot(fig)

    streamlit.write("Overlay of Original channel (Red) with Corrected (Cyan) channel")
    [p, q, r, s, t, u] = [
        [
            _red_channel(p / (numpy.max(p) + theia.constants.EPSILON)),
            _overlay(
                p / (numpy.max(p) + theia.constants.EPSILON),
                q / (numpy.max(q) + theia.constants.EPSILON),
            ),
            _cyan_channel(q / (numpy.max(q) + theia.constants.EPSILON)),
        ]
        for p, q in zip(original, corrected)
    ]
    images = [*p, *q, *r, *s, *t, *u]
    prefixes = ["Original", "Overlay", "Corrected"]
    fig, axs = pyplot.subplots(6, 3, figsize=(9, 18), dpi=256)
    for i, (ax, img) in enumerate(zip(axs.flat, images)):
        ax.imshow(img)
        ax.set_title(f"{prefixes[i % 3]} {channel_names[i // 3]}")
        ax.axis("off")
    streamlit.pyplot(fig)

    streamlit.write(
        "Overlay of Original channel (Red) with "
        "Bleed-Through (Cyan) into that channel.",
    )
    [p, q, r, s, t, u] = [
        [
            _red_channel(p / (numpy.max(p) + theia.constants.EPSILON)),
            _overlay(p / (numpy.max(p) + theia.constants.EPSILON), q),
            _cyan_channel(q),
        ]
        for p, q in zip(original, bleedthrough)
    ]
    images = [*p, *q, *r, *s, *t, *u]
    prefixes = ["Original", "Overlay", "Bleed-Through"]
    fig, axs = pyplot.subplots(6, 3, figsize=(9, 18), dpi=256)
    for i, (ax, img) in enumerate(zip(axs.flat, images)):
        ax.imshow(img)
        ax.set_title(f"{prefixes[i % 3]} {channel_names[i // 3]}")
        ax.axis("off")
    streamlit.pyplot(fig)

    streamlit.write(
        "Overlay of Corrected channel (Red) with Bleed-Through (Cyan) removed.",
    )
    [p, q, r, s, t, u] = [
        [
            _red_channel(p / (numpy.max(p) + theia.constants.EPSILON)),
            _overlay(p / (numpy.max(p) + theia.constants.EPSILON), q),
            _cyan_channel(q),
        ]
        for p, q in zip(corrected, bleedthrough)
    ]
    images = [*p, *q, *r, *s, *t, *u]
    prefixes = ["Corrected", "Overlay", "Bleed-Through"]
    fig, axs = pyplot.subplots(6, 3, figsize=(9, 18), dpi=256)
    for i, (ax, img) in enumerate(zip(axs.flat, images)):
        ax.imshow(img)
        ax.set_title(f"{prefixes[i % 3]} {channel_names[i // 3]}")
        ax.axis("off")
    streamlit.pyplot(fig)


def _red_channel(img: numpy.ndarray) -> numpy.ndarray:
    x, y = img.shape
    red = numpy.zeros((x, y, 3), dtype=img.dtype)
    red[:, :, 0] = img
    return red


def _cyan_channel(img: numpy.ndarray) -> numpy.ndarray:
    x, y = img.shape
    cyan = numpy.zeros((x, y, 3), dtype=img.dtype)
    cyan[:, :, 1] = img
    cyan[:, :, 2] = img
    return cyan


def _overlay(red: numpy.ndarray, cyan: numpy.ndarray) -> numpy.ndarray:
    red_ = _red_channel(red)
    cyan_ = _cyan_channel(cyan)
    return red_ + cyan_


if __name__ == "__main__":
    # run_theia(0, 0)
    app()
