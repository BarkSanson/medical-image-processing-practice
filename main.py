import os

import numpy as np

from coregistration import coregister
from dicom_io import load_dynamic_pet, load_3d_dicom
from fusion import alpha_fusion
from projections import mean_intensity_projection
from visualization import (
    create_median_gif,
    create_rotation_gif,
    create_pixel_array_gif,
    show_median_planes,
    show_overlay,
    show_planes_grid,
)

DATA_PATH = "data"
RESULTS_PATH = "results"
DYNAMIC_PET_FILE = os.path.join(DATA_PATH, "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916")
MR_FILE = os.path.join(DATA_PATH, "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916")


def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)

    pet_dicom, dynamic_pet = load_dynamic_pet(DYNAMIC_PET_FILE)
    print("DYNAMIC PET INFORMATION")
    print("-----------------------")
    print(pet_dicom)

    mr_dicom, mr = load_3d_dicom(MR_FILE)
    print("MR INFORMATION")
    print("-----------------------")
    print(mr_dicom)

    # Median planes of the last PET frame and the average PET frame
    last_pet_frame = dynamic_pet.with_array(np.flip(dynamic_pet.pixel_array[-1]))
    average_frame = dynamic_pet.with_array(np.flip(np.mean(dynamic_pet.pixel_array, axis=0)))
    show_median_planes(RESULTS_PATH, "last-pet-frame", last_pet_frame, cmap="hot")
    show_median_planes(RESULTS_PATH, "average-pet-frame", average_frame, cmap="hot")

    # Show the MR median planes
    show_median_planes(RESULTS_PATH, "mr-median-planes", mr.with_array(np.flip(mr.pixel_array)), cmap="bone")

    create_median_gif(dynamic_pet.with_array(np.flip(dynamic_pet.pixel_array)), RESULTS_PATH, "medians", cmap="hot")

    # Coregister the temporal mean of the dynamic PET onto the MR.
    pet_mip = dynamic_pet.with_array(mean_intensity_projection(dynamic_pet.pixel_array))
    coregistered_pet = coregister(mr, pet_mip)

    # Side-by-side and overlay views of MR vs. coregistered PET.
    # Flip pixel arrays for better visualization
    flipped_mr = mr.with_array(np.flip(mr.pixel_array))
    flipped_coregistered_pet = coregistered_pet.with_array(np.flip(coregistered_pet.pixel_array))

    show_planes_grid(
        RESULTS_PATH,
        [
            (flipped_mr, "MR", "bone"),
            (flipped_coregistered_pet, "Coregistered PET", "hot")
        ],
        aspects_from=coregistered_pet,
    )
    show_overlay(RESULTS_PATH, flipped_mr, flipped_coregistered_pet, aspects_from=coregistered_pet)

    create_pixel_array_gif(flipped_mr, RESULTS_PATH, "reference_mip_rotation")
    create_pixel_array_gif(flipped_coregistered_pet, RESULTS_PATH, "coregistered_mip_rotation", cmap="hot")

    # Alpha-fused rotation GIF of MR + coregistered PET.
    alpha_fused = alpha_fusion(flipped_mr.pixel_array, flipped_coregistered_pet.pixel_array)
    create_rotation_gif(alpha_fused, RESULTS_PATH, "alpha_fused", aspect=mr.metadata.sagittal_aspect)


if __name__ == "__main__":
    main()
