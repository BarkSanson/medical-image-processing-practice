import numpy as np
import SimpleITK as sitk


def load_nrrd_mask(path: str, segment_index: int = 0) -> np.ndarray:
    """
    Loads a segmentation stored as an NRRD file (e.g. exported from 3D Slicer)
    and returns it as a boolean ``numpy`` array.

    SimpleITK returns the volume with axes already in ``(slices, rows, cols)``
    order, which matches the layout used by ``pydicom``'s ``pixel_array`` in
    this project.

    If the NRRD contains multiple segments (4D array with a leading channel
    axis, as Slicer writes when several segments share the same node) the
    one indicated by ``segment_index`` is selected.

    :param path: path to the ``.nrrd`` file
    :param segment_index: index of the segment to extract when the file
        contains more than one
    :return: boolean mask
    """
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)

    if array.ndim == 4:
        array = array[segment_index]

    return array.astype(bool)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Sorensen-Dice coefficient between two binary masks.

    ``Dice = 2 * |X n Y| / (|X| + |Y|)``

    Returns ``1.0`` when both masks are empty (degenerate case).

    :param pred: predicted mask, any array-like that can be cast to bool
    :param gt: ground-truth mask, same shape as ``pred``
    :return: Dice coefficient in ``[0, 1]``
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)

    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs gt {g.shape}")

    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0

    intersection = np.logical_and(p, g).sum()
    return 2.0 * intersection / denom
