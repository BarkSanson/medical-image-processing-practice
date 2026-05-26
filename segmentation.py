import os
from enum import Enum

import numpy as np
import torch
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = "./weights/nnInteractive"  # Specify the download directory

SEGMENTATION_BBOX = [
    [45, 90],
    [154, 207],
    [150, 193],
]

def segment(pixel_array: np.ndarray) -> np.ndarray | None:
    download_path = _download_weights()

    session = nnInteractiveInferenceSession(
        torch_n_threads=os.cpu_count(),
        device=torch.device("cpu"),
    )
    session.initialize_from_trained_model_folder(download_path)

    input_pixel_array = pixel_array[np.newaxis, ...]
    session.set_image(input_pixel_array)
    target_tensor = torch.zeros(pixel_array.shape)
    session.set_target_buffer(target_tensor)

    session.add_bbox_interaction(SEGMENTATION_BBOX, include_interaction=True)

    results = session.target_buffer.clone()

    return results.numpy()

def _download_weights() -> str:
    download_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
    if os.path.exists(download_path) and len(os.listdir(download_path)) > 0:
        return download_path

    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[
            f"{MODEL_NAME}/*",
        ],
        local_dir=DOWNLOAD_DIR,
    )

    return download_path