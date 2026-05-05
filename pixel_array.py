from dataclasses import dataclass
from typing import Self

import matplotlib
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
from pydicom import FileDataset

from tags import TAGS


def median_sagittal_plane(pixel_array: np.ndarray) -> np.ndarray:
    return pixel_array[:, :, pixel_array.shape[2] // 2]


def median_coronal_plane(pixel_array: np.ndarray) -> np.ndarray:
    return pixel_array[:, pixel_array.shape[1] // 2, :]


def median_axial_plane(pixel_array: np.ndarray) -> np.ndarray:
    return pixel_array[pixel_array.shape[0] // 2, :, :]


def mean_intensity_projection(pixel_array: np.ndarray, axis=0) -> np.ndarray:
    return np.mean(pixel_array, axis=axis)


def max_intensity_projection(pixel_array: np.ndarray, axis=1) -> np.ndarray:
    return np.max(pixel_array, axis=axis)


def alpha_fusion(reference: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    ref_norm = (reference - reference.min()) / (np.ptp(reference) + 1e-9)
    mask_norm = (mask - mask.min()) / (np.ptp(mask) + 1e-9)

    reference_cmapped = matplotlib.colormaps["bone"](ref_norm)
    mask_cmapped = matplotlib.colormaps["hot"](mask_norm)

    return reference_cmapped * (1 - alpha) + mask_cmapped * alpha

def create_gif_rotation(title: str, pixel_array: np.ndarray, aspect: float):
    n = 60

    fig, ax = plt.subplots(figsize=(12, 4))

    animation_data = []
    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        rotated = _rotate_on_axial_plane(pixel_array, alpha)
        mip_sagittal = max_intensity_projection(rotated, axis=2)

        sagittal_plot = ax.imshow(
            mip_sagittal,
            animated=True,
            cmap=matplotlib.colormaps["bone"],
            aspect=aspect,
        )

        animation_data.append([sagittal_plot])

    anim = animation.ArtistAnimation(fig, animation_data, interval=100, blit=True)
    anim.save(f"{title}.gif")


def _rotate_on_axial_plane(pixel_array: np.ndarray, angle_in_degrees: float):
    return scipy.ndimage.rotate(pixel_array, angle=angle_in_degrees, axes=(1, 2), reshape=False)


@dataclass
class PixelArrayMetadata:
    pixel_spacing: tuple[float, float]
    spacing_between_slices: float
    image_position_patient: np.ndarray
    image_orientation_patient: np.ndarray

    @classmethod
    def from_dicom(cls, dicom: FileDataset) -> Self:
        return cls(
            pixel_spacing=dicom[TAGS["PixelSpacing"]].value,
            spacing_between_slices=float(dicom[TAGS["SpacingBetweenSlices"]].value),
            image_position_patient=dicom[TAGS["DetectorInformationSequence"]][0].ImagePositionPatient,
            image_orientation_patient=dicom[TAGS["DetectorInformationSequence"]][0].ImageOrientationPatient,
        )

@dataclass
class PixelArray:
    metadata: PixelArrayMetadata
    pixel_array: np.ndarray

    def create_median_gif(self, title: str):
        if len(self.pixel_array.shape) < 4:
            raise

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        animation_data = []
        for frame in self.pixel_array:
            sagital, coronal, axial = (
                median_sagittal_plane(frame),
                median_coronal_plane(frame),
                median_axial_plane(frame),
            )

            sagittal_plot = ax1.imshow(
                sagital,
                animated=True,
                cmap=matplotlib.colormaps["bone"],
                aspect=self.metadata.spacing_between_slices/self.metadata.pixel_spacing[0]
            )

            coronal_plot = ax2.imshow(
                coronal,
                animated=True,
                cmap=matplotlib.colormaps["bone"],
                aspect=self.metadata.spacing_between_slices/self.metadata.pixel_spacing[0]
            )

            axial_plot = ax3.imshow(
                axial,
                animated=True,
                cmap=matplotlib.colormaps["bone"],
                aspect=self.metadata.spacing_between_slices/self.metadata.pixel_spacing[0]
            )

            animation_data.append([sagittal_plot, coronal_plot, axial_plot])

        anim = animation.ArtistAnimation(fig, animation_data, interval=100, blit=True)
        anim.save(f"{title}.gif")
