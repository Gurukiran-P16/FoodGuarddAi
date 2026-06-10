from __future__ import annotations

import numpy as np


def run_inference(tensor: np.ndarray) -> dict[str, float | str]:
    """Return placeholder freshness prediction for a normalized (224, 224, 3) tensor."""
    mean_value = float(np.mean(tensor))
    confidence = min(0.99, max(0.01, abs(mean_value)))
    label = "fresh" if mean_value >= 0.5 else "stale"
    return {"label": label, "confidence": confidence}
