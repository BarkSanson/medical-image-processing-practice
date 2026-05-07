from dataclasses import dataclass
from typing import Self

import numpy as np
from pydicom import FileDataset


@dataclass
class PixelArrayMetadata:
    """
    Holds the spatial metadata associated with a ``PixelArray``. Stores only the
    DICOM fields that are needed by the rest of the pipeline (co-registration
    and visualization).
    """
    pixel_spacing: tuple[float, float]
    spacing_between_slices: float
    image_position_patient: np.ndarray
    image_orientation_patient: np.ndarray

    @classmethod
    def from_dicom(cls, dicom: FileDataset) -> Self:
        """
        Builds a ``PixelArrayMetadata`` from a pydicom ``FileDataset``. The image
        position and orientation are read from the first item of the
        ``DetectorInformationSequence``.
        :param dicom: source DICOM dataset
        :return: a new ``PixelArrayMetadata`` populated from the DICOM tags
        """
        detector = dicom.DetectorInformationSequence[0]
        return cls(
            pixel_spacing=dicom.PixelSpacing,
            spacing_between_slices=float(dicom.SpacingBetweenSlices),
            image_position_patient=detector.ImagePositionPatient,
            image_orientation_patient=detector.ImageOrientationPatient,
        )

    @property
    def sagittal_aspect(self) -> float:
        """
        Aspect ratio to use when displaying the sagittal median plane.
        :return: ``spacing_between_slices / pixel_spacing[0]``
        """
        return self.spacing_between_slices / self.pixel_spacing[0]

    @property
    def coronal_aspect(self) -> float:
        """
        Aspect ratio to use when displaying the coronal median plane.
        :return: ``spacing_between_slices / pixel_spacing[1]``
        """
        return self.spacing_between_slices / self.pixel_spacing[1]

    @property
    def axial_aspect(self) -> float:
        """
        Aspect ratio to use when displaying the axial median plane.
        :return: ``pixel_spacing[0] / pixel_spacing[1]``
        """
        return self.pixel_spacing[0] / self.pixel_spacing[1]

    @property
    def plane_aspects(self) -> tuple[float, float, float]:
        """
        Convenience accessor that bundles the three plane aspect ratios.
        :return: ``(sagittal_aspect, coronal_aspect, axial_aspect)``
        """
        return self.sagittal_aspect, self.coronal_aspect, self.axial_aspect


@dataclass
class PixelArray:
    """
    Pairs a numpy pixel array with the spatial metadata needed to interpret
    it. The pixel array can be 3D (slices, rows, cols) or 4D
    (frames, slices, rows, cols) for dynamic PET.
    """
    metadata: PixelArrayMetadata
    pixel_array: np.ndarray

    def with_array(self, pixel_array: np.ndarray) -> "PixelArray":
        """
        Creates a new ``PixelArray`` that reuses the current metadata but wraps
        a different pixel array. Useful to derive intermediate volumes (e.g. a
        single frame or a temporal projection) without rebuilding the metadata.
        :param pixel_array: pixel array of the new ``PixelArray``
        :return: a new ``PixelArray`` sharing the metadata of ``self``
        """
        return PixelArray(metadata=self.metadata, pixel_array=pixel_array)
