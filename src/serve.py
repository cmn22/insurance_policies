# src/serve.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import os

import pandas as pd
from joblib import load

APP = FastAPI(title="Insurance Model API", version="1.0.0")

# -------- Config / Paths --------
ROOT = Path(__file__).resolve().parents[1]  # repo root (src/..)
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.json"
SCHEMA_PATH = PROCESSED_DIR / "schema.json"  # optional, from prepare.py

# -------- Runtime state --------
_model = None
_feature_names: List[str] = []
_schema: Dict[str, Any] = {}

# -------- I/O Schemas --------
class PredictRequest(BaseModel):
    # Raw feature dict for ONE sample (unencoded). Keys are your original column names.
    # If you already send encoded columns matching feature_names, it will also work.
    features: Dict[str, Any] = Field(..., description="Input feature mapping for one sample")

class PredictBatchRequest(BaseModel):
    # List of raw feature dicts
    rows: List[Dict[str, Any]] = Field(..., description="List of feature mappings for multiple samples")

class PredictResponse(BaseModel):
    prediction: float

class PredictBatchResponse(BaseModel):
    predictions: List[float]

class Metadata(BaseModel):
    model_path: str
    n_features: int
    feature_names: List[str]
    has_schema: bool
    schema: Optional[Dict[str, Any]] = None

# -------- Helpers --------
def _load_artifacts() -> None:
    global _model, _feature_names, _schema
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first (dvc repro).")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found at {FEATURES_PATH}.")
    _model = load(MODEL_PATH)
    _feature_names = json.loads(FEATURES_PATH.read_text())
    if SCHEMA_PATH.exists():
        try:
            _schema = json.loads(SCHEMA_PATH.read_text())
        except Exception:
            _schema = {}

def _ensure_loaded():
    if _model is None or not _feature_names:
        _load_artifacts()

def _coerce_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    # Best-effort numeric coercion where possible (without breaking strings)
    for col in df.columns:
        # Try numeric; if fails, leave as object
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df

def _align_features_from_raw(row_or_rows: List[Dict[str, Any]] | Dict[str, Any]) -> pd.DataFrame:
    """
    Accepts raw dict or list-of-dicts; will:
      - Build a DataFrame
      - Try one-hot encode via get_dummies
      - Align to _feature_names (add missing cols=0, drop extras)
    If input already matches encoded feature space, alignment still works.
    """
    # Normalize input to DataFrame
    if isinstance(row_or_rows, dict):
        df_raw = pd.DataFrame([row_or_rows])
    else:
        df_raw = pd.DataFrame(row_or_rows)

    # Try to detect if already encoded: if most saved feature_names are present, assume encoded.
    saved = set(_feature_names)
    overlap = saved.intersection(df_raw.columns)
    already_encoded = len(overlap) >= max(1, int(0.8 * len(saved)))  # heuristic

    if already_encoded:
        X = df_raw.copy()
    else:
        # One-hot encode on provided rows; base (drop_first) category is implicit (all zeros)
        X = pd.get_dummies(df_raw, drop_first=True)

    X = _coerce_dataframe_types(X)

    # Align to training feature space
    for missing in (saved - set(X.columns)):
        X[missing] = 0
    # Drop any unknown columns
    X = X[[c for c in _feature_names]]
    return X

# -------- Routes --------
@APP.on_event("startup")
def _startup():
    _ensure_loaded()

@APP.get("/health")
def health() -> Dict[str, str]:
    try:
        _ensure_loaded()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@APP.get("/model/metadata", response_model=Metadata)
def model_metadata():
    _ensure_loaded()
    meta = Metadata(
        model_path=str(MODEL_PATH),
        n_features=len(_feature_names),
        feature_names=_feature_names,
        has_schema=bool(_schema),
        schema=_schema or None,
    )
    return meta

@APP.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _ensure_loaded()
    try:
        X = _align_features_from_raw(req.features)
        pred = float(_model.predict(X)[0])
        return PredictResponse(prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@APP.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    _ensure_loaded()
    try:
        X = _align_features_from_raw(req.rows)
        preds = _model.predict(X).tolist()
        return PredictBatchResponse(predictions=[float(p) for p in preds])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")