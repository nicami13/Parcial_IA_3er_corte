from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "occupation_model.joblib"
ENGINEER_CLASS = "Engineer"
MODEL_WARNING = (
    "Modelo experimental: en la evaluación, su rendimiento fue cercano al azar. "
    "No uses estas probabilidades para decisiones reales."
)


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ARTIFACT_PATH.is_file():
        raise RuntimeError(
            f"No se encontró {ARTIFACT_PATH}. "
            "Ejecuta primero: python train_api_model.py"
        )
    app.state.artifact = joblib.load(ARTIFACT_PATH)
    yield


app = FastAPI(
    title="API de clasificación de ocupación",
    description=(
        "Clasifica perfiles de clientes entre ocupaciones. "
        "El modelo es experimental y no es adecuado para decisiones reales."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    artifact = request.app.state.artifact
    return {
        "status": "ok",
        "classes": len(artifact["classes"]),
        "warning": MODEL_WARNING,
    }


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
    artifact = request.app.state.artifact
    expected_columns = artifact["feature_columns"]
    extra_columns = sorted(set(payload.features) - set(expected_columns))
    if extra_columns:
        raise HTTPException(
            status_code=422,
            detail={"message": "Features desconocidas", "columns": extra_columns},
        )

    row = {column: payload.features.get(column) for column in expected_columns}
    for column in artifact["numeric_columns"]:
        value = row[column]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise HTTPException(
                status_code=422,
                detail=f"'{column}' debe ser un número finito o null.",
            )

    for column in artifact["boolean_columns"]:
        value = row[column]
        if value is not None and not isinstance(value, bool):
            raise HTTPException(
                status_code=422,
                detail=f"'{column}' debe ser true, false o null.",
            )

    categorical_columns = set(expected_columns) - set(
        artifact["numeric_columns"]
    ) - set(artifact["boolean_columns"])
    for column in categorical_columns:
        value = row[column]
        if value is not None and not isinstance(value, str):
            raise HTTPException(
                status_code=422,
                detail=f"'{column}' debe ser texto o null.",
            )

    frame = pd.DataFrame([row], columns=expected_columns)
    model = artifact["model"]
    try:
        probabilities = model.predict_proba(frame)[0]
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudieron procesar las features: {error}",
        ) from error

    classes = artifact["classes"]
    predicted_index = int(np.argmax(probabilities))
    engineer_probability = (
        float(probabilities[classes.index(ENGINEER_CLASS)])
        if ENGINEER_CLASS in classes
        else None
    )
    top_indices = np.argsort(probabilities)[::-1][:3]

    return {
        "predicted_occupation": classes[predicted_index],
        "predicted_probability": float(probabilities[predicted_index]),
        "probability_engineer": engineer_probability,
        "top_3": [
            {
                "occupation": classes[int(index)],
                "probability": float(probabilities[index]),
            }
            for index in top_indices
        ],
        "warning": MODEL_WARNING,
    }
