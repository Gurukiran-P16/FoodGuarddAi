from __future__ import annotations

import numpy as np


def generate_gradcam_summary(tensor: np.ndarray) -> dict[str, float]:
    heatmap = tensor.mean(axis=-1) if tensor.ndim == 3 else tensor
    return {"min": float(np.min(heatmap)), "max": float(np.max(heatmap))}
