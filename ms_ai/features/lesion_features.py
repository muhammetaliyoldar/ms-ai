"""
Lesion feature extraction from NIfTI mask files.

Computes per-case volumetric features using 3D connected-component analysis
with a 26-neighbourhood (full connectivity).
"""

from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
from scipy.ndimage import label

# 26-neighbourhood structuring element (full 3D connectivity)
_STRUCT_26 = np.ones((3, 3, 3), dtype=np.int8)


def compute_lesion_features(mask_path: Path) -> Dict[str, float]:
    """
    Compute lesion features from a binary NIfTI mask.

    Parameters
    ----------
    mask_path : Path
        Path to a ``*_MASK.nii.gz`` (or predicted mask) file.

    Returns
    -------
    dict with keys:
        lesion_voxels, lesion_volume_mm3, lesion_count,
        mean_lesion_size_mm3, max_lesion_size_mm3
    """
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    binary = (data > 0).astype(np.int8)

    # Voxel volume from header zooms (mm per voxel along each axis)
    zooms = img.header.get_zooms()[:3]
    voxel_vol = float(np.prod(zooms))

    lesion_voxels = int(binary.sum())

    # Connected-component labelling (26-connectivity)
    labelled, num_labels = label(binary, structure=_STRUCT_26)

    if num_labels == 0:
        return {
            "lesion_voxels": 0,
            "lesion_volume_mm3": 0.0,
            "lesion_count": 0,
            "mean_lesion_size_mm3": 0.0,
            "max_lesion_size_mm3": 0.0,
        }

    # Size of each connected component (in voxels)
    component_sizes = np.bincount(labelled.ravel())[1:]  # skip background=0
    component_volumes = component_sizes.astype(np.float64) * voxel_vol

    return {
        "lesion_voxels": lesion_voxels,
        "lesion_volume_mm3": round(float(lesion_voxels * voxel_vol), 2),
        "lesion_count": int(num_labels),
        "mean_lesion_size_mm3": round(float(component_volumes.mean()), 2),
        "max_lesion_size_mm3": round(float(component_volumes.max()), 2),
    }
