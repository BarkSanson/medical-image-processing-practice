import os

import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from pixel_array import PixelArray, PixelArrayMetadata, median_sagittal_plane, median_coronal_plane, median_axial_plane, \
    mean_intensity_projection, create_gif_rotation, alpha_fusion
from tags import TAGS

DATA_PATH = "data"
DYNAMIC_PET_FILE = os.path.join(DATA_PATH, "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916")
MR_FILE = os.path.join(DATA_PATH, "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916")


def main():
    dynamic_pet = pydicom.dcmread(DYNAMIC_PET_FILE)
    print("DYNAMIC PET INFORMATION")
    print("-----------------------")
    print(dynamic_pet)
    mr = pydicom.dcmread(MR_FILE)

    mr_pa = mr.pixel_array

    print("MR INFORMATION")
    print("-----------------------")
    print(mr)

    rearranged = dynamic_pet.pixel_array.reshape((
        dynamic_pet[TAGS["NumberOfFrames"]].value // dynamic_pet[TAGS["NumberOfSlices"]].value,
        dynamic_pet[TAGS["NumberOfSlices"]].value,
        dynamic_pet[TAGS["Rows"]].value,
        dynamic_pet[TAGS["Columns"]].value,
    ))

    rearranged = PixelArray(
        metadata=PixelArrayMetadata.from_dicom(dynamic_pet),
        pixel_array=rearranged
    )

    mr = PixelArray(
        metadata=PixelArrayMetadata.from_dicom(mr),
        pixel_array=mr_pa
    )

    last_frame = rearranged.pixel_array[-1, :, :, :]

    sagital_pet_last, coronal_pet_last, axial_pet_last = (
        median_sagittal_plane(last_frame),
        median_coronal_plane(last_frame),
        median_axial_plane(last_frame)
    )

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(sagital_pet_last, cmap=matplotlib.colormaps["bone"], aspect=3.27/1.171875)
    ax[0].set_title("Sagittal")
    ax[1].imshow(coronal_pet_last, cmap=matplotlib.colormaps["bone"], aspect=3.27/1.171875)
    ax[1].set_title("Coronal")
    ax[2].imshow(axial_pet_last, cmap=matplotlib.colormaps["bone"], aspect=3.27/1.171875)
    ax[2].set_title("Axial")
    plt.show()

    sagital_mr, coronal_mr, axial_mr = (
        median_sagittal_plane(mr_pa),
        median_coronal_plane(mr_pa),
        median_axial_plane(mr_pa)
    )

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(sagital_mr, cmap=matplotlib.colormaps["bone"], aspect=1/1.0)
    ax[0].set_title("Sagittal")
    ax[1].imshow(coronal_mr, cmap=matplotlib.colormaps["bone"], aspect=1/1.0)
    ax[1].set_title("Coronal")
    ax[2].imshow(axial_mr, cmap=matplotlib.colormaps["bone"], aspect=1/1.0)
    ax[2].set_title("Axial")
    plt.show()

    #rearranged.create_median_gif("medians")
    dyn_pet_coregistration = mean_intensity_projection(rearranged.pixel_array)
    dyn_pet_coregistration = PixelArray(
        metadata=PixelArrayMetadata.from_dicom(dynamic_pet),
        pixel_array=dyn_pet_coregistration
    )

    coregistered_pet = coregister(mr, dyn_pet_coregistration)

    coreg_pa = coregistered_pet.pixel_array
    coreg_sagittal = median_sagittal_plane(coreg_pa)
    coreg_coronal = median_coronal_plane(coreg_pa)
    coreg_axial = median_axial_plane(coreg_pa)

    coreg_spacing = coregistered_pet.metadata
    sag_aspect = coreg_spacing.spacing_between_slices / coreg_spacing.pixel_spacing[0]
    cor_aspect = coreg_spacing.spacing_between_slices / coreg_spacing.pixel_spacing[1]
    ax_aspect = coreg_spacing.pixel_spacing[0] / coreg_spacing.pixel_spacing[1]

    mr_sagittal = median_sagittal_plane(mr.pixel_array)
    mr_coronal = median_coronal_plane(mr.pixel_array)
    mr_axial = median_axial_plane(mr.pixel_array)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].imshow(mr_sagittal, cmap=matplotlib.colormaps["bone"], aspect=sag_aspect)
    ax[0, 0].set_title("MR sagittal")
    ax[0, 1].imshow(mr_coronal, cmap=matplotlib.colormaps["bone"], aspect=cor_aspect)
    ax[0, 1].set_title("MR coronal")
    ax[0, 2].imshow(mr_axial, cmap=matplotlib.colormaps["bone"], aspect=ax_aspect)
    ax[0, 2].set_title("MR axial")

    ax[1, 0].imshow(coreg_sagittal, cmap=matplotlib.colormaps["hot"], aspect=sag_aspect)
    ax[1, 0].set_title("Coregistered PET sagittal")
    ax[1, 1].imshow(coreg_coronal, cmap=matplotlib.colormaps["hot"], aspect=cor_aspect)
    ax[1, 1].set_title("Coregistered PET coronal")
    ax[1, 2].imshow(coreg_axial, cmap=matplotlib.colormaps["hot"], aspect=ax_aspect)
    ax[1, 2].set_title("Coregistered PET axial")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(mr_sagittal, cmap=matplotlib.colormaps["bone"], aspect=sag_aspect)
    ax[0].imshow(coreg_sagittal, cmap=matplotlib.colormaps["hot"], aspect=sag_aspect, alpha=0.4)
    ax[0].set_title("Sagittal overlay")
    ax[1].imshow(mr_coronal, cmap=matplotlib.colormaps["bone"], aspect=cor_aspect)
    ax[1].imshow(coreg_coronal, cmap=matplotlib.colormaps["hot"], aspect=cor_aspect, alpha=0.4)
    ax[1].set_title("Coronal overlay")
    ax[2].imshow(mr_axial, cmap=matplotlib.colormaps["bone"], aspect=ax_aspect)
    ax[2].imshow(coreg_axial, cmap=matplotlib.colormaps["hot"], aspect=ax_aspect, alpha=0.4)
    ax[2].set_title("Axial overlay")
    plt.tight_layout()
    plt.show()

    #create_gif_rotation(
    #    "reference_mip_rotation",
    #    mr.pixel_array,
    #    mr.metadata.spacing_between_slices / mr.metadata.pixel_spacing[0]
    #)

    #create_gif_rotation(
    #    "coregistered_mip_rotation",
    #    coregistered_pet.pixel_array,
    #    coregistered_pet.metadata.spacing_between_slices / coregistered_pet.metadata.pixel_spacing[0]
    #)

    alpha_fused = alpha_fusion(
        mr.pixel_array,
        coregistered_pet.pixel_array
    )

    create_gif_rotation(
        "alpha_fused",
        alpha_fused,
        mr.metadata.spacing_between_slices / mr.metadata.pixel_spacing[0]
    )

def coregister(
    reference: PixelArray,
    input: PixelArray,
):
    def create_sitk_image(pixel_array: PixelArray) -> sitk.Image:
        new_image = sitk.GetImageFromArray(pixel_array.pixel_array)

        new_image.SetOrigin([float(x) for x in pixel_array.metadata.image_position_patient])

        pixel_spacing = pixel_array.metadata.pixel_spacing
        slice_thickness = pixel_array.metadata.spacing_between_slices
        new_image.SetSpacing([float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)])

        return new_image

    reference_sitk = sitk.Cast(create_sitk_image(reference), sitk.sitkFloat32)
    input_sitk = sitk.Cast(create_sitk_image(input), sitk.sitkFloat32)


    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
    )
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerScalesFromPhysicalShift()

    aligned_transform = sitk.CenteredTransformInitializer(
        reference_sitk,
        input_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(aligned_transform)

    final_transform = R.Execute(reference_sitk, input_sitk)

    resampled = sitk.Resample(
        input_sitk,
        reference_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        reference_sitk.GetPixelID()
    )

    spacing = resampled.GetSpacing()
    origin = resampled.GetOrigin()
    orientation = resampled.GetDirection()
    new_metadata = PixelArrayMetadata(
        pixel_spacing=spacing[0:2],
        spacing_between_slices=spacing[2],
        image_position_patient=origin,
        image_orientation_patient=orientation,
    )

    return PixelArray(
        metadata=new_metadata,
        pixel_array=sitk.GetArrayFromImage(resampled),
    )



if __name__ == '__main__':
    main()
