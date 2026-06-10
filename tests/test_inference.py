import numpy as np

from app.engine.inference import run_inference


def test_inference_output_values() -> None:
    tensor = np.ones((224, 224, 3), dtype=np.float32)
    result = run_inference(tensor)
    assert result["label"] in {"fresh", "stale"}
    assert 0.0 <= float(result["confidence"]) <= 1.0
