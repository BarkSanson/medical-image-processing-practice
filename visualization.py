import os
from typing import Sequence

import matplotlib
import numpy as np
import pyvista as pv
import scipy
from matplotlib import animation
from matplotlib import pyplot as plt

from pixel_array import PixelArray
from projections import max_intensity_projection, median_planes

PLANE_TITLES: tuple[str, str, str] = ("Sagittal", "Coronal", "Axial")


def show_median_planes(
    path: str,
    filename: str,
    pixel_array: PixelArray,
    cmap: str = "bone",
    titles: Sequence[str] = PLANE_TITLES,
    show: bool = True,
):
    """
    Displays the sagittal, coronal and axial median planes of a pixel_array
    :param path: path to results directory
    :param pixel_array: Source ``PixelArray`` to extract the median planes from
    :param cmap: colormap to use
    :param titles: titles to show in the plots
    :param show: whether to show the plots
    :return: Figure and Axes objects of the plot
    """
    planes = median_planes(pixel_array.pixel_array)
    aspects = pixel_array.metadata.plane_aspects
    cmap_obj = matplotlib.colormaps[cmap]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for axis, plane, aspect, title in zip(axes, planes, aspects, titles):
        axis.imshow(plane, cmap=cmap_obj, aspect=aspect)
        axis.set_title(title)
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    plt.savefig(os.path.join(path, f"{filename}.png"))

    if show:
        plt.show()

    return fig, axes


def show_planes_grid(
    path: str,
    pixel_arrays: Sequence[tuple[PixelArray, str, str]],
    aspects_from: PixelArray | None = None,
    show: bool = True,
):
    """
    Display the median planes of one or more ``PixelArray`` objects vertically.
    Aspect ratios are taken from ``aspects_from`` (defaults to the first volume).
    :param path: path to the results directory
    :param pixel_arrays: pixel_arrays to show. Each entry should be a tuple
    of (pixel_array, row_label, colormap).
    :param aspects_from:
    :param show: whether to show the plot
    :return: Figure and Axes objects of the plot
    """
    if not pixel_arrays:
        raise ValueError("At least one volume is required")

    # Defaulting to the plane aspects of the first pixel_array is fine
    aspects = (aspects_from or pixel_arrays[0][0]).metadata.plane_aspects

    fig, axes = plt.subplots(len(pixel_arrays), 3, figsize=(10, 3 * len(pixel_arrays)))
    if len(pixel_arrays) == 1:
        axes = np.array([axes])

    for row, (volume, label, cmap) in enumerate(pixel_arrays):
        cmap_obj = matplotlib.colormaps[cmap]
        for col, (plane, aspect, title) in enumerate(
            zip(median_planes(volume.pixel_array), aspects, PLANE_TITLES)
        ):
            axes[row, col].imshow(plane, cmap=cmap_obj, aspect=aspect)
            axes[row, col].set_title(f"{label} {title.lower()}")
            axes[row, col].set_xticklabels([])
            axes[row, col].set_yticklabels([])

    plt.savefig(os.path.join(path, "grid.png"))

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes


def show_overlay(
    path: str,
    reference: PixelArray,
    input_: PixelArray,
    reference_cmap: str = "bone",
    input_cmap: str = "hot",
    alpha: float = 0.4,
    aspects_from: PixelArray | None = None,
    show: bool = True,
):
    """
    Shows a reference and a moving image overlayed. Basically, this function
    is expected to be used after a co-registration. 
    :param path: path to the results directory
    :param reference: reference image
    :param input_: input image
    :param reference_cmap: colormap of the reference image
    :param input_cmap: colormap of the input image
    :param alpha: transparency of the overlay
    :param aspects_from: ``PixelArray`` to extract the aspect ratios. Defaults to the
    input image
    :param show: whether to show the plot
    :return: Figure and Axes objects of the plot
    """
    aspects = (aspects_from or input_).metadata.plane_aspects
    ref_planes = median_planes(reference.pixel_array)
    mov_planes = median_planes(input_.pixel_array)
    ref_cmap = matplotlib.colormaps[reference_cmap]
    mov_cmap = matplotlib.colormaps[input_cmap]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for axis, rp, mp, aspect, title in zip(axes, ref_planes, mov_planes, aspects, PLANE_TITLES):
        axis.imshow(rp, cmap=ref_cmap, aspect=aspect)
        axis.imshow(mp, cmap=mov_cmap, aspect=aspect, alpha=alpha)
        axis.set_title(f"{title} overlay")
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    plt.savefig(os.path.join(path, "overlay.png"))
    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes


def create_median_gif(pixel_array: PixelArray, path: str, title: str, cmap: str = "bone"):
    """
    Creates a GIF of the median planes of each frame of a dynamic PET. This function expects
    a 4D pixel_array, with the shape (frames, slices, rows, cols).
    :param path: path to the results directory
    :param pixel_array: source ``PixelArray``
    :param title: title of the plot
    :param cmap: colormap to use
    """
    if pixel_array.pixel_array.ndim < 4:
        raise ValueError("create_median_gif requires a 4D volume (frames, slices, rows, cols)")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    aspects = pixel_array.metadata.plane_aspects
    cmap_obj = matplotlib.colormaps[cmap]

    frames = []
    for idx, frame in enumerate(pixel_array.pixel_array):
        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plots = [
            axis.imshow(plane, animated=True, cmap=cmap_obj, aspect=aspect)
            for axis, plane, aspect in zip(axes, median_planes(frame), aspects)
        ]
        plt.savefig(os.path.join(path, f"{title}_{idx}.png"))
        frames.append(plots)

    anim = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    anim.save(os.path.join(path, f"{title}.gif"))


def _rotate_on_axial_plane(pixel_array: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    return scipy.ndimage.rotate(pixel_array, angle=angle_in_degrees, axes=(1, 2), reshape=False)


def create_rotation_gif(
    pixel_array: np.ndarray,
    path: str,
    title: str,
    aspect: float,
    n_frames: int = 30,
    cmap: str = "bone",
):
    """
    Creates a GIF of the Max Intensity Projection (MIP) of a dynamic PET frame. It
    expects only one frame, and thus the pixel_array shape should be (slices, rows, cols).
    :param pixel_array: source ``PixelArray``
    :param path: path to the results directory
    :param title: title of the plot
    :param aspect: aspect ratio of the plot
    :param n_frames: number of frames to generate for the GIF
    :param cmap: colormap to use
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    cmap_obj = matplotlib.colormaps[cmap]

    frames = []
    for angle in np.linspace(0, 360 * (n_frames - 1) / n_frames, num=n_frames):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        rotated = _rotate_on_axial_plane(pixel_array, angle)
        mip = max_intensity_projection(rotated, axis=2)
        frames.append([ax.imshow(mip, animated=True, cmap=cmap_obj, aspect=aspect)])
        plt.savefig(os.path.join(path, f"{title}_{angle}.png"))

    anim = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    anim.save(os.path.join(path, f"{title}.gif"))


def create_pixel_array_gif(pixel_array: PixelArray, path: str, title: str, n_frames: int = 30, cmap: str = "bone"):
    """
    Convenience wrapper around ``create_rotation_gif`` that takes a ``PixelArray``.
    """
    create_rotation_gif(
        pixel_array.pixel_array,
        path,
        title,
        aspect=pixel_array.metadata.sagittal_aspect,
        n_frames=n_frames,
        cmap=cmap,
    )


def show_masked_tumor(
    mr: np.ndarray,
    mask: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    bbox: Sequence[Sequence[float]] | None = None,
    reference_mask: np.ndarray | None = None,
):
    grid = pv.ImageData(
        dimensions=np.array(mr.shape),  # ImageData uses point dims = cell dims + 1
        spacing=spacing,
    )
    grid.point_data["mr"] = mr.flatten(order="F")
    grid.point_data["mask"] = mask.astype(np.uint8).flatten(order="F")

    plotter = pv.Plotter()
    # translucent volume rendering of the MR
    # softer opacity ramp -> MR appears more translucent so the mask shows through
    mr_opacity = [0.0, 0.05, 0.12, 0.20, 0.30, 0.40]
    plotter.add_volume(
        grid,
        scalars="mr",
        cmap="bone",
        opacity=mr_opacity,
        opacity_unit_distance=10.0,
        shade=True,
    )
    # opaque isosurface of the predicted mask
    mask_surface = grid.contour(isosurfaces=[0.5], scalars="mask")
    plotter.add_mesh(
        mask_surface, color="red", opacity=0.7, smooth_shading=True, label="Prediction"
    )

    if reference_mask is not None:
        if reference_mask.shape != mr.shape:
            raise ValueError(
                f"reference_mask shape {reference_mask.shape} does not match mr shape {mr.shape}"
            )
        grid.point_data["reference"] = reference_mask.astype(np.uint8).flatten(order="F")
        reference_surface = grid.contour(isosurfaces=[0.5], scalars="reference")
        plotter.add_mesh(
            reference_surface,
            color="green",
            opacity=0.4,
            smooth_shading=True,
            label="Ground truth",
        )
        plotter.add_legend()

    if bbox is not None:
        # bbox is given in array axis order (axis0, axis1, axis2), matching
        # the ``dimensions`` order passed to ``pv.ImageData`` (i.e. x, y, z
        # in PyVista's local frame). ``pv.Box`` expects bounds in the form
        # (xmin, xmax, ymin, ymax, zmin, zmax).
        (a0_min, a0_max), (a1_min, a1_max), (a2_min, a2_max) = bbox
        bounds = (
            a0_min * spacing[0], a0_max * spacing[0],
            a1_min * spacing[1], a1_max * spacing[1],
            a2_min * spacing[2], a2_max * spacing[2],
        )
        box = pv.Box(bounds=bounds)
        plotter.add_mesh(box, color="yellow", style="wireframe", line_width=2)

    plotter.show()
