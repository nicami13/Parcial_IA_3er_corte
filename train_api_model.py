from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from Model import DATA_PATH, TARGET, build_preprocessor, load_data, prepare_features


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "occupation_model.joblib"


def main() -> None:
    data = load_data(DATA_PATH)
    features, target = prepare_features(data)
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    tol=1e-2,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features, target)

    artifact = {
        "model": model,
        "feature_columns": features.columns.tolist(),
        "numeric_columns": features.select_dtypes(include="number").columns.tolist(),
        "boolean_columns": features.select_dtypes(include="bool").columns.tolist(),
        "classes": model.named_steps["classifier"].classes_.tolist(),
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ARTIFACT_PATH.with_suffix(".tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(ARTIFACT_PATH)
    print(f"Modelo entrenado con {len(data):,} filas y guardado en {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
