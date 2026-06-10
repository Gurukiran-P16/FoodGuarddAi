from fastapi import APIRouter, File, UploadFile

from app.engine.explainability import generate_gradcam_summary
from app.engine.inference import run_inference
from app.utils.preprocessing import to_tensor

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    image_bytes = await file.read()
    tensor = to_tensor(image_bytes)
    prediction = run_inference(tensor)
    xai = generate_gradcam_summary(tensor)
    return {"prediction": prediction, "xai": xai}
