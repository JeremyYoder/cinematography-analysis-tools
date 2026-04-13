"""
Scalable FastAPI server for cinematography-analysis-tools.

Exposes REST API endpoints for predicting shot types and extracting color palettes natively.
"""

from __future__ import annotations

import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .model import load_model
from .predict import predict_image
from .color import extract_palette

app = FastAPI(
    title="Cinematography Analysis API",
    description="Zero-IO REST API for Shot Type Classification and CIELAB Color Palettes",
    version="1.0.0",
)

# Allow CORS for external web dashboards (e.g. Adobe Premiere extensions)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading PyTorch model into global API context...")
    model, _ = load_model()


@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model is not None}


@app.post("/predict/image")
async def api_predict_image(file: UploadFile = File(...)):
    """Predict the cinematic shot type from an uploaded image in-memory."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = predict_image(model, img)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/colors")
async def api_extract_colors(colors: int = 5, file: UploadFile = File(...)):
    """Extract a dominant CIELAB Color Palette from an image in-memory."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        palette = extract_palette(img, n_colors=colors)
        return {"file": "api-upload", "palette": palette}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def serve(host: str = "127.0.0.1", port: int = 8000):
    """Entry point to start the uvicorn server via CLI."""
    import uvicorn
    print(f"🎬 Starting Cinematography Analysis API on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
