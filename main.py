import os

from coregistration import coregister
from dicom_io import load_dynamic_pet, load_volume
from fusion import alpha_fusion
from projections import mean_intensity_projection
from visualization import (
    create_median_gif,
    create_rotation_gif,
    create_volume_rotation_gif,
    show_median_planes,
    show_overlay,
    show_planes_grid,
)

DATA_PATH = "data"
DYNAMIC_PET_FILE = os.path.join(DATA_PATH, "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916")
MR_FILE = os.path.join(DATA_PATH, "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916")


def main():
    pet_dicom, dynamic_pet = load_dynamic_pet(DYNAMIC_PET_FILE)
    print("DYNAMIC PET INFORMATION")
    print("-----------------------")
    print(pet_dicom)

    mr_dicom, mr = load_volume(MR_FILE)
    print("MR INFORMATION")
    print("-----------------------")
    print(mr_dicom)

    # Median planes of the last PET frame and of the MR.
    last_pet_frame = dynamic_pet.with_array(dynamic_pet.pixel_array[-1])
    show_median_planes(last_pet_frame, cmap="hot")
    show_median_planes(mr, cmap="bone")

    # Optional: GIF cycling through the median planes of every PET frame.
    create_median_gif(dynamic_pet, "medians")

    # Coregister the temporal mean of the dynamic PET onto the MR.
    pet_mip = dynamic_pet.with_array(mean_intensity_projection(dynamic_pet.pixel_array))
    coregistered_pet = coregister(mr, pet_mip)

    # Side-by-side and overlay views of MR vs. coregistered PET.
    show_planes_grid(
        [(mr, "MR", "bone"), (coregistered_pet, "Coregistered PET", "hot")],
        aspects_from=coregistered_pet,
    )
    show_overlay(mr, coregistered_pet, aspects_from=coregistered_pet)

    # Optional: rotation GIFs of the reference MR and coregistered PET.
    create_volume_rotation_gif(mr, "reference_mip_rotation")
    create_volume_rotation_gif(coregistered_pet, "coregistered_mip_rotation")

    # Alpha-fused rotation GIF of MR + coregistered PET.
    alpha_fused = alpha_fusion(mr.pixel_array, coregistered_pet.pixel_array)
    create_rotation_gif(alpha_fused, "alpha_fused", aspect=mr.metadata.sagittal_aspect)


if __name__ == "__main__":
    main()
