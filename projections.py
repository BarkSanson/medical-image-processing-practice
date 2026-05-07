import numpy as np


def median_sagittal_plane(pixel_array: np.ndarray) -> np.ndarray:
    """
    Extracts the sagittal median plane of a 3D pixel array
    :param pixel_array: 3D pixel array of shape ``(slices, rows, cols)``
    :return: 2D array with the sagittal median plane
    """
    return pixel_array[:, :, pixel_array.shape[2] // 2]


def median_coronal_plane(pixel_array: np.ndarray) -> np.ndarray:
    """
    Extracts the coronal median plane of a 3D pixel array
    :param pixel_array: 3D pixel array of shape ``(slices, rows, cols)``
    :return: 2D array with the coronal median plane
    """
    return pixel_array[:, pixel_array.shape[1] // 2, :]


def median_axial_plane(pixel_array: np.ndarray) -> np.ndarray:
    """
    Extracts the axial median plane of a 3D pixel array
    :param pixel_array: 3D pixel array of shape ``(slices, rows, cols)``
    :return: 2D array with the axial median plane
    """
    return pixel_array[pixel_array.shape[0] // 2, :, :]


def median_planes(pixel_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience helper that returns the three median planes of a 3D pixel
    array in a single call.
    :param pixel_array: 3D pixel array of shape ``(slices, rows, cols)``
    :return: a tuple ``(sagittal, coronal, axial)`` with the three median planes
    """
    return (
        median_sagittal_plane(pixel_array),
        median_coronal_plane(pixel_array),
        median_axial_plane(pixel_array),
    )


def mean_intensity_projection(pixel_array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Computes the Mean Intensity Projection of a pixel array along the given
    axis. When applied to a dynamic PET (4D, with ``axis=0``) it produces a
    temporal mean volume.
    :param pixel_array: source pixel array
    :param axis: axis along which the mean is computed
    :return: pixel array with one less dimension than the input
    """
    return np.mean(pixel_array, axis=axis)


def max_intensity_projection(pixel_array: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Computes the Maximum Intensity Projection (MIP) of a pixel array along
    the given axis.
    :param pixel_array: source pixel array
    :param axis: axis along which the maximum is computed
    :return: pixel array with one less dimension than the input
    """
    return np.max(pixel_array, axis=axis)
