from fastapi import APIRouter, File, HTTPException, UploadFile

from app.engine.explainability import generate_gradcam_summary
from app.engine.inference import run_inference
from app.utils.preprocessing import to_tensor

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")
    image_bytes = await file.read()
    try:
        tensor = to_tensor(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    prediction = run_inference(tensor)
    xai = generate_gradcam_summary(tensor)
    return {"prediction": prediction, "xai": xai}
