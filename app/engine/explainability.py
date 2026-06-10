from __future__ import annotations

import numpy as np


def generate_gradcam_summary(tensor: np.ndarray) -> dict[str, float]:
    """Return placeholder Grad-CAM heatmap extrema derived from the input tensor."""
    heatmap = tensor.mean(axis=-1) if tensor.ndim == 3 else tensor
    return {"min": float(np.min(heatmap)), "max": float(np.max(heatmap))}
