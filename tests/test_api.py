from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _image_bytes() -> bytes:
    img = Image.new("RGB", (16, 16), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "xai" in payload


def test_predict_endpoint_rejects_non_image_content_type() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only image uploads are supported."


def test_predict_endpoint_rejects_corrupted_image() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("broken.png", b"not-valid-image-bytes", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."
