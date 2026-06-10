# FoodGuardAI

FoodGuardAI is an optimized, containerized computer vision microservice that classifies physical food degradation (freshness) while providing real-time Explainable AI (XAI) visual feedback.

## Project Structure

```text
FoodGuardAI/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   └── explainability.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py
├── model_pipeline/
│   ├── train_resnet.py
│   └── export_to_onnx.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_inference.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```
