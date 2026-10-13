"""
MTAN-SLR FastAPI REST API & WebSocket Server
============================================
Provides REST endpoints and WebSocket video/keypoint streaming interface
for deploying sign language recognition & translation as a microservice.
"""

import os
import sys
import yaml
import numpy as np
import torch
from typing import List, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models import build_model_from_config
from src.nlp.sign_to_sentence import SignToSentenceTranslator
from src.utils.logger import get_logger

logger = get_logger("MTAN_API")

# Global state variables
MODEL = None
ACTIONS = ["hello", "thanks", "i" , "water"]
TRANSLATOR = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequencePredictRequest(BaseModel):
    sequence: List[List[float]]  # (T, 454) or (T, 226)


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: List[float]
    all_classes: List[str]


class TranslateRequest(BaseModel):
    sign_sequence: List[str]


class TranslateResponse(BaseModel):
    input_signs: List[str]
    translated_sentence: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, ACTIONS, TRANSLATOR

    model_config_path = "configs/model_config.yaml"
    checkpoint_path = "checkpoints/best_model.pt"

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    if os.path.exists(checkpoint_path):
        logger.info(f"Loading model checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        ACTIONS = checkpoint.get("actions", ACTIONS)
        active_model = checkpoint.get("active_model", model_cfg.get("active_model", "bilstm_attention"))
        model_cfg["active_model"] = active_model

        MODEL = build_model_from_config(model_cfg, num_classes=len(ACTIONS))
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.to(DEVICE)
        MODEL.eval()
        logger.info(f"Model '{active_model}' loaded successfully with classes: {ACTIONS}")
    else:
        logger.warning(f"No checkpoint found at '{checkpoint_path}'. Initializing unweighted model.")
        MODEL = build_model_from_config(model_cfg, num_classes=len(ACTIONS)).to(DEVICE)

    TRANSLATOR = SignToSentenceTranslator()
    yield

app = FastAPI(
    title="MTAN-SLR Sign Language Recognition & Translation API",
    description="Research-grade real-time sign language recognition microservice.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "classes_count": len(ACTIONS),
    }


@app.post("/predict/sequence", response_model=PredictionResponse)
def predict_sequence(req: SequencePredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    seq_arr = np.array(req.sequence, dtype=np.float32)
    if seq_arr.ndim != 2:
        raise HTTPException(status_code=400, detail="Sequence must be 2D array (T, features).")

    T, D = seq_arr.shape
    if D != 454:
        padded = np.zeros((T, 454), dtype=np.float32)
        padded[:, :min(D, 454)] = seq_arr[:, :min(D, 454)]
        seq_arr = padded

    seq_tensor = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, T, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        logits = MODEL(seq_tensor, mask=mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    best_idx = int(np.argmax(probs))

    return PredictionResponse(
        predicted_label=ACTIONS[best_idx],
        confidence=float(probs[best_idx]),
        probabilities=probs.tolist(),
        all_classes=ACTIONS,
    )


@app.post("/translate", response_model=TranslateResponse)
def translate_signs(req: TranslateRequest):
    if TRANSLATOR is None:
        raise HTTPException(status_code=503, detail="Translator not initialized.")

    sentence = TRANSLATOR(req.sign_sequence)
    return TranslateResponse(
        input_signs=req.sign_sequence,
        translated_sentence=sentence,
    )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        while True:
            data = await websocket.receive_json()
            raw_seq = data.get("features", [])
            if raw_seq and MODEL is not None:
                seq_arr = np.array(raw_seq, dtype=np.float32)
                T, D = seq_arr.shape
                if D != 454:
                    padded = np.zeros((T, 454), dtype=np.float32)
                    padded[:, :min(D, 454)] = seq_arr[:, :min(D, 454)]
                    seq_arr = padded

                seq_tensor = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                mask = torch.ones(1, T, dtype=torch.bool, device=DEVICE)

                with torch.no_grad():
                    logits = MODEL(seq_tensor, mask=mask)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                best_idx = int(np.argmax(probs))
                await websocket.send_json({
                    "label": ACTIONS[best_idx],
                    "confidence": float(probs[best_idx]),
                    "probabilities": probs.tolist(),
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
