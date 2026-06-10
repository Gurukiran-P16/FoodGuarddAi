from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError


def to_tensor(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array
