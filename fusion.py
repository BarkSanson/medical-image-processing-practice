import matplotlib
import numpy as np


def alpha_fusion(
    reference: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.25,
    reference_cmap: str = "bone",
    mask_cmap: str = "hot",
) -> np.ndarray:
    """
    Performs an alpha-fusion between a reference image and a mask image. Both
    images are min-max normalized and mapped through their respective
    colormaps before being linearly blended with weight ``alpha`` on the mask.
    Typically used after a co-registration to overlay a PET ``mask`` onto an
    MR ``reference``.

    :param reference: reference image (e.g. an MR volume)
    :param mask: image to overlay on top of the reference (e.g. a PET volume)
    :param alpha: blending weight applied to the mask; the reference is
    weighted with ``1 - alpha``
    :param reference_cmap: colormap used to render the reference image
    :param mask_cmap: colormap used to render the mask image
    :return: RGBA pixel array with the same spatial shape as the inputs and a
    trailing channel dimension of size 4
    """
    ref_norm = (reference - reference.min()) / (np.ptp(reference) + 1e-9)
    mask_norm = (mask - mask.min()) / (np.ptp(mask) + 1e-9)

    reference_cmapped = matplotlib.colormaps[reference_cmap](ref_norm)
    mask_cmapped = matplotlib.colormaps[mask_cmap](mask_norm)

    return reference_cmapped * (1 - alpha) + mask_cmapped * alpha
