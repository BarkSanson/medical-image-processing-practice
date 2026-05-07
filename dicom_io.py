import pydicom

from pixel_array import PixelArray, PixelArrayMetadata


def load_volume(path: str) -> tuple[pydicom.FileDataset, PixelArray]:
    """
    Reads a DICOM file and wraps it as a 3D ``PixelArray``. The raw
    pydicom dataset is returned as well so the caller can inspect any tag that
    is not part of ``PixelArrayMetadata``.

    :param path: path to the DICOM file
    :return: a tuple ``(dicom, pixel_array)`` with the raw ``FileDataset`` and
    the wrapped ``PixelArray``
    """
    dicom = pydicom.dcmread(path)
    volume = PixelArray(
        metadata=PixelArrayMetadata.from_dicom(dicom),
        pixel_array=dicom.pixel_array,
    )
    return dicom, volume


def load_dynamic_pet(path: str) -> tuple[pydicom.FileDataset, PixelArray]:
    """
    Reads a multi-frame dynamic PET DICOM file and reshapes its pixel array
    into a 4D volume of shape ``(frames, slices, rows, cols)``, where
    ``frames = NumberOfFrames / NumberOfSlices``.

    :param path: path to the dynamic PET DICOM file
    :return: a tuple ``(dicom, pixel_array)`` with the raw ``FileDataset`` and
    the wrapped 4D ``PixelArray``
    """
    dicom = pydicom.dcmread(path)
    n_frames = int(dicom.NumberOfFrames)
    n_slices = int(dicom.NumberOfSlices)
    rearranged = dicom.pixel_array.reshape((
        n_frames // n_slices,
        n_slices,
        int(dicom.Rows),
        int(dicom.Columns),
    ))
    volume = PixelArray(
        metadata=PixelArrayMetadata.from_dicom(dicom),
        pixel_array=rearranged,
    )
    return dicom, volume
