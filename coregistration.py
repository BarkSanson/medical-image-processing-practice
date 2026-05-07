import SimpleITK as sitk

from pixel_array import PixelArray, PixelArrayMetadata


def _to_sitk_image(pixel_array: PixelArray) -> sitk.Image:
    """
    Transforms a `PixelArray` into a `sitk.Image`.
    :param pixel_array: `PixelArray` to be transformed.
    :return: a new `sitk.Image` with the metadata of the pixel_array
    """
    image = sitk.GetImageFromArray(pixel_array.pixel_array)
    image.SetOrigin([float(x) for x in pixel_array.metadata.image_position_patient])

    pixel_spacing = pixel_array.metadata.pixel_spacing
    slice_thickness = pixel_array.metadata.spacing_between_slices

    image.SetSpacing([float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)])
    return image


def coregister(reference: PixelArray, input_: PixelArray) -> PixelArray:
    """
    Co-register a moving or input image with a reference image. For this specific work,
    these images are an MR and a dynamic PET, but this method could be potentially
    used to coregister any other pair of images.

    The method performs a rigid-body transformation, using Mattes Mutual Information as
    loss function and optimizing via a Gradient Descent. Interpolations are performed
    linearly.
    :param reference: reference image that the input image will be co-registered to
    :param input_: input image to be co-registered to the coordinate space of the reference
    image
    :return: a new `PixelArray` with the metadata of the reference image and the
    co-registered pixel array.
    """
    reference_sitk = sitk.Cast(_to_sitk_image(reference), sitk.sitkFloat32)
    moving_sitk = sitk.Cast(_to_sitk_image(input_), sitk.sitkFloat32)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        reference_sitk,
        moving_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    method.SetInitialTransform(initial_transform)

    final_transform = method.Execute(reference_sitk, moving_sitk)

    resampled = sitk.Resample(
        moving_sitk,
        reference_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        reference_sitk.GetPixelID(),
    )

    spacing = resampled.GetSpacing()
    metadata = PixelArrayMetadata(
        pixel_spacing=spacing[0:2],
        spacing_between_slices=spacing[2],
        image_position_patient=resampled.GetOrigin(),
        image_orientation_patient=resampled.GetDirection(),
    )

    return PixelArray(metadata=metadata, pixel_array=sitk.GetArrayFromImage(resampled))
