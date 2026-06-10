from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def to_tensor(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array
