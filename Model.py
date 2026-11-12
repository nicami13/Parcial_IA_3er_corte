from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Dataset" / "customer_data.csv"
TARGET = "occupation"
ENGINEER_CLASS = "Engineer"
RANDOM_STATE = 42

IDENTIFIER_COLUMNS = ["customer_id"]
DATE_COLUMNS = [
    "first_tx",
    "last_tx",
    "last_survey_date",
    "last_transaction_date",
    "first_transaction_date",
]
REDUNDANT_COLUMNS = ["transaction_frequency", "avg_daily_transactions"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the source CSV and validate that the selected target is present."""
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET}' en {path}.")
    return data


def cramers_v(feature: pd.Series, target: pd.Series) -> float:
    """Measure association between two categorical variables."""
    table = pd.crosstab(
        feature.fillna("__MISSING__").astype(str),
        target.astype(str),
    )
    if min(table.shape) < 2:
        return 0.0

    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(
        axis=0, keepdims=True
    ) / total
    chi_squared = np.sum((observed - expected) ** 2 / expected)
    return float(np.sqrt(chi_squared / (total * min(table.shape[0] - 1, table.shape[1] - 1))))


def correlation_ratio(groups: pd.Series, values: pd.Series) -> float:
    """Measure association between a categorical target and a numeric feature."""
    data = pd.DataFrame({"group": groups, "value": values}).dropna()
    if data.empty:
        return 0.0

    grand_mean = data["value"].mean()
    total_sum_squares = ((data["value"] - grand_mean) ** 2).sum()
    if total_sum_squares == 0:
        return 0.0

    group_sum_squares = data.groupby("group", observed=True)["value"].apply(
        lambda values_in_group: len(values_in_group)
        * (values_in_group.mean() - grand_mean) ** 2
    ).sum()
    return float(np.sqrt(group_sum_squares / total_sum_squares))


def flatten_text_column(values: np.ndarray) -> np.ndarray:
    return values.ravel()


def tokenize_requests(value: str) -> list[str]:
    return [request.strip() for request in value.split(",") if request.strip()]


def print_analysis(data: pd.DataFrame) -> None:
    """Print data-quality, distribution, redundancy, and target-association results."""
    print(f"\nDataset: {data.shape[0]:,} filas x {data.shape[1]} columnas")
    print(f"Filas duplicadas: {data.duplicated().sum():,}")

    missing = data.isna().sum()
    missing = missing[missing.gt(0)].sort_values(ascending=False)
    if missing.empty:
        print("\nDatos faltantes: ninguno")
    else:
        missing_report = pd.DataFrame(
            {
                "faltantes": missing,
                "porcentaje": (missing / len(data) * 100).round(2),
            }
        )
        print("\nDatos faltantes:")
        print(missing_report.to_string())

    target_counts = data[TARGET].value_counts().sort_values()
    print(
        f"\nClases de {TARGET}: {target_counts.size}; "
        f"frecuencia por clase: {target_counts.min():,}–{target_counts.max():,}"
    )
    print("Clases menos frecuentes:")
    print(target_counts.head(5).to_string())

    numeric = data.select_dtypes(include="number").drop(
        columns=IDENTIFIER_COLUMNS, errors="ignore"
    )
    correlations = numeric.corr()
    correlated_pairs = []
    for index, first in enumerate(correlations.columns):
        for second in correlations.columns[index + 1 :]:
            value = correlations.loc[first, second]
            if abs(value) >= 0.95:
                correlated_pairs.append((first, second, value))
    print("\nPares numéricos con |correlación de Pearson| >= 0.95:")
    if correlated_pairs:
        for first, second, value in sorted(
            correlated_pairs, key=lambda pair: abs(pair[2]), reverse=True
        ):
            print(f"  {first} / {second}: {value:.5f}")
    else:
        print("  Ninguno")

    print("\nColumnas exactamente duplicadas:")
    duplicates = []
    columns = list(data.columns)
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            if data[first].equals(data[second]):
                duplicates.append((first, second))
    if duplicates:
        for first, second in duplicates:
            print(f"  {first} = {second}")
    else:
        print("  Ninguna")

    associations = []
    excluded = set(
        [TARGET]
        + IDENTIFIER_COLUMNS
        + DATE_COLUMNS
        + REDUNDANT_COLUMNS
    )
    for column in data.columns:
        if column in excluded:
            continue
        feature = data[column]
        if pd.api.types.is_numeric_dtype(feature) and not pd.api.types.is_bool_dtype(
            feature
        ):
            score = correlation_ratio(data[TARGET], feature)
            method = "eta (numérica)"
        else:
            score = cramers_v(feature, data[TARGET])
            method = "V de Cramer"
        associations.append((score, method, column))

    print("\nAsociación exploratoria con occupation (no causal ni importancia definitiva):")
    for score, method, column in sorted(associations, reverse=True)[:10]:
        print(f"  {column}: {score:.4f} ({method})")


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target and remove identifiers, raw dates, and redundant measures."""
    drop_columns = [
        column
        for column in IDENTIFIER_COLUMNS + DATE_COLUMNS + REDUNDANT_COLUMNS
        if column in data.columns
    ]
    features = data.drop(columns=[TARGET, *drop_columns])
    target = data[TARGET].copy()
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build train-only imputing, scaling, and one-hot encoding transformations."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    request_columns = (
        ["feature_requests"] if "feature_requests" in features.columns else []
    )
    categorical_columns = features.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    categorical_columns = [
        column for column in categorical_columns if column not in request_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__SIN_REGISTRO__"),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    request_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__SIN_REGISTRO__"),
            ),
            ("flatten", FunctionTransformer(flatten_text_column, validate=False)),
            (
                "multi_hot",
                CountVectorizer(
                    tokenizer=tokenize_requests,
                    token_pattern=None,
                    lowercase=False,
                    binary=True,
                ),
            ),
        ]
    )

    transformers = [
        ("numericas", numeric_pipeline, numeric_columns),
        ("categoricas", categorical_pipeline, categorical_columns),
    ]
    if request_columns:
        transformers.append(("solicitudes", request_pipeline, request_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def evaluate_classifier(features: pd.DataFrame, target: pd.Series) -> None:
    """Tune three classifiers on training folds and evaluate once on the test set."""
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)
    engineer_class = int(label_encoder.transform([ENGINEER_CLASS])[0])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_predictions = dummy.predict(np.zeros((len(y_test), 1)))
    baseline_probabilities = dummy.predict_proba(np.zeros((len(y_test), 1)))
    baseline_log_loss = log_loss(
        y_test, baseline_probabilities, labels=dummy.classes_
    )
    search_x, _, search_y, _ = train_test_split(
        x_train,
        y_train,
        train_size=min(12_000, len(x_train)),
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    cross_validation = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )
    preprocessor = build_preprocessor(x_train)

    model_grids = [
        (
            "Regresión logística",
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                tol=1e-2,
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__C": [0.01, 0.1],
            },
        ),
        (
            "Bosque aleatorio",
            RandomForestClassifier(
                n_estimators=60,
                max_features="sqrt",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__max_depth": [5, 10],
                "classifier__min_samples_leaf": [10, 30],
            },
        ),
        (
            "Red neuronal MLP",
            MLPClassifier(
                max_iter=60,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                batch_size=256,
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__hidden_layer_sizes": [(32,), (64,)],
                "classifier__alpha": [0.1, 1.0],
            },
        ),
    ]

    print("\nBúsqueda de hiperparámetros")
    print("GridSearchCV: F1 macro, validación cruzada estratificada de 3 pliegues")
    print(
        f"El ajuste usa una muestra estratificada de {len(search_x):,} filas "
        "del conjunto de entrenamiento; luego el mejor modelo se reentrena "
        "con todo el entrenamiento. La prueba queda reservada."
    )
    print(
        "Regularización: L2 en logística; profundidad/tamaño de hoja en bosque; "
        "L2 (alpha) y parada temprana en MLP."
    )

    results = []
    for model_name, classifier, parameter_grid in model_grids:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1_macro",
            cv=cross_validation,
            n_jobs=1,
            refit=True,
            return_train_score=True,
            error_score="raise",
            verbose=1,
        )
        print(f"\nAjustando {model_name}...", flush=True)
        search.fit(search_x, search_y)

        best_model = search.best_estimator_
        best_model.fit(x_train, y_train)
        predictions = best_model.predict(x_test)
        probabilities = best_model.predict_proba(x_test)
        classes = best_model.named_steps["classifier"].classes_
        engineer_index = int(np.flatnonzero(classes == engineer_class)[0])
        engineer_probabilities = probabilities[:, engineer_index]
        engineer_actual = (y_test == engineer_class).astype(int)
        engineer_prevalence = float(np.mean(y_train == engineer_class))
        engineer_baseline_probabilities = np.full(
            len(y_test), engineer_prevalence
        )

        results.append(
            {
                "name": model_name,
                "cv_f1": search.best_score_,
                "train_cv_f1": search.cv_results_["mean_train_score"][
                    search.best_index_
                ],
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "f1_macro": f1_score(y_test, predictions, average="macro"),
                "log_loss": log_loss(y_test, probabilities, labels=classes),
                "top_3_accuracy": top_k_accuracy_score(
                    y_test, probabilities, k=3, labels=classes
                ),
                "engineer_precision": precision_score(
                    y_test,
                    predictions,
                    labels=[engineer_class],
                    average=None,
                    zero_division=0,
                )[0],
                "engineer_recall": recall_score(
                    y_test,
                    predictions,
                    labels=[engineer_class],
                    average=None,
                    zero_division=0,
                )[0],
                "engineer_roc_auc": roc_auc_score(
                    engineer_actual, engineer_probabilities
                ),
                "engineer_pr_auc": average_precision_score(
                    engineer_actual, engineer_probabilities
                ),
                "engineer_brier": brier_score_loss(
                    engineer_actual, engineer_probabilities
                ),
                "engineer_baseline_brier": brier_score_loss(
                    engineer_actual, engineer_baseline_probabilities
                ),
                "best_params": search.best_params_,
            }
        )

    print("\nResultados en el conjunto de prueba (20% reservado):")
    print(
        f"Baseline mayoritario: accuracy={accuracy_score(y_test, baseline_predictions):.4f}, "
        f"F1 macro={f1_score(y_test, baseline_predictions, average='macro'):.4f}"
    )
    class_count = len(label_encoder.classes_)
    print(
        f"Referencia de azar uniforme ({class_count} clases): "
        f"accuracy esperada={1 / class_count:.4f}, "
        f"top-3 esperado={min(3, class_count) / class_count:.4f}"
    )
    print(f"Log loss baseline de prevalencia: {baseline_log_loss:.4f}")
    print(
        "Modelo | CV F1 macro | Test F1 macro | Test balanced acc. | "
        "Test accuracy | Test log loss | Top-3"
    )
    for result in sorted(results, key=lambda item: item["f1_macro"], reverse=True):
        print(
            f"{result['name']} | {result['cv_f1']:.4f} | "
            f"{result['f1_macro']:.4f} | {result['balanced_accuracy']:.4f} | "
            f"{result['accuracy']:.4f} | {result['log_loss']:.4f} | "
            f"{result['top_3_accuracy']:.4f}"
        )
        print(f"  Mejores parámetros: {result['best_params']}")
        print(
            f"  F1 macro entrenamiento CV: {result['train_cv_f1']:.4f}; "
            f"brecha train-CV: "
            f"{result['train_cv_f1'] - result['cv_f1']:.4f}"
        )
        print(
            "  Engineer: "
            f"precision={result['engineer_precision']:.4f}, "
            f"recall={result['engineer_recall']:.4f}, "
            f"ROC AUC={result['engineer_roc_auc']:.4f}, "
            f"PR AUC={result['engineer_pr_auc']:.4f}, "
            f"Brier={result['engineer_brier']:.4f} "
            f"(baseline={result['engineer_baseline_brier']:.4f})"
        )


def main() -> None:
    data = load_data()
    print_analysis(data)
    features, target = prepare_features(data)
    print(
        f"\nPreprocesamiento definido: {features.shape[1]} features antes de codificar; "
        f"target separado = {TARGET}."
    )
    evaluate_classifier(features, target)


if __name__ == "__main__":
    main()
