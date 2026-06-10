# FoodGuardAI: High-Performance Visual Freshness & XAI Engine

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![ONNX](https://img.shields.io/badge/ONNXRuntime-Optimized-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)

## Overview
FoodGuardAI is a production-ready computer vision microservice designed to classify physical food degradation and freshness. Instead of deploying heavy PyTorch models to production, this engine utilizes an **ONNX-compiled inference pipeline** to minimize latency, served via an asynchronous **FastAPI** backend. 

Crucially, the API does not function as a black box. It implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide Explainable AI (XAI) outputs, returning base64-encoded heatmaps detailing the exact physical regions driving the classification.

## Engineering Focus
* **Algorithmic Inference Optimization:** Compiled base ResNet50 architecture into ONNX format, stripping training graphs to reduce inference latency by [Add % drop later]% compared to native PyTorch execution.
* **Explainable AI (XAI):** Implemented custom spatial gradient extraction to generate visual heatmaps, bridging physical state changes (e.g., surface oxidation, bruising) with model interpretation.
* **Systems Architecture:** Fully containerized backend using Docker, featuring decoupled route handling, preprocessing, and inference engines.

## API Architecture

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Returns 200 OK and model load status. |
| `/api/v1/analyze` | `POST` | Accepts a multipart/form-data image upload. Returns JSON payload with classification confidence and base64 XAI heatmap. |

## Local Development & Deployment

### 1. Model Preparation (Offline Pipeline)
Before running the server, the PyTorch weights must be converted to an optimized ONNX graph.
\`\`\`bash
python model_pipeline/export_to_onnx.py --input weights/best_model.pth --output app/engine/model.onnx
\`\`\`

### 2. Standard Server Run
\`\`\`bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
\`\`\`

### 3. Docker Deployment
\`\`\`bash
docker build -t foodguard-api .
docker run -p 8000:8000 foodguard-api
\`\`\`

## System Diagram
*(Will add Mermaid.js architecture diagram mapping the flow from Client -> FastAPI -> Preprocessor -> ONNX Runtime -> Grad-CAM Extractor -> JSON Response)*
