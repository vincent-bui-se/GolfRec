"""Train and evaluate gradient-boosted models for golf fitting labels."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from preprocess import INPUT_COLUMNS, make_feature_frame
from synthetic_data import DEFAULT_PROFILE_COUNT, predict_rule_labels, save_dataset


LABEL_COLUMNS = [
    "driver_loft",
    "shaft_flex",
    "fairway_wood_loft",
    "iron_category",
    "wedge_bounce",
]

# Labels that are numeric loft degrees rather than category names round-trip
# through CSV as floats (16.5 -> "16.5", but 15 -> "15.0"); the rest are
# already display-ready strings.
_NUMERIC_LABELS = ["driver_loft", "fairway_wood_loft"]


def _clean_numeric_label(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def normalize_label_columns(golfers: pd.DataFrame) -> pd.DataFrame:
    """Keep labels as display-ready strings after CSV round-trips."""
    normalized = golfers.copy()
    for label in _NUMERIC_LABELS:
        normalized[label] = normalized[label].map(_clean_numeric_label)
    for label in ["shaft_flex", "iron_category", "wedge_bounce"]:
        normalized[label] = normalized[label].astype(str)
    return normalized


# 8 combinations per label, 5-fold CV.
_SHARED_GRID: dict[str, list[Any]] = {
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__max_leaf_nodes": [15, 31],
    "classifier__min_samples_leaf": [20, 40],
}
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {label: dict(_SHARED_GRID) for label in LABEL_COLUMNS}


def build_base_pipeline(random_state: int = 42) -> Pipeline:
    """Class-balanced gradient boosting; GridSearchCV handles parallelism.

    early_stopping is set explicitly because sklearn only turns it on by default
    above 10k rows, and small runs (like the tests) would otherwise fit all 500.
    """
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(make_feature_frame, validate=False)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=500,
                    early_stopping=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_models(
    golfers: pd.DataFrame,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Tune balanced boosted-tree models on training rows and evaluate on a held-out 20%.

    Repeated CV describes the selected model on training rows only.
    """
    golfers = normalize_label_columns(golfers)
    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}
    x = golfers[INPUT_COLUMNS]

    for label in LABEL_COLUMNS:
        y = golfers[label].astype(str)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.20,
            random_state=random_state,
            stratify=y,
        )

        base = build_base_pipeline(random_state=random_state)
        grid = GridSearchCV(
            base,
            param_grid=PARAM_GRIDS[label],
            cv=5,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(x_train, y_train)
        best_model: Pipeline = grid.best_estimator_

        y_pred = best_model.predict(x_test)
        rule_pred = predict_rule_labels(x_test)[label]
        labels = sorted(y.unique())

        repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=random_state)
        cv_scores = cross_val_score(
            best_model, x_train, y_train, cv=repeated, scoring="accuracy", n_jobs=-1
        )

        print(
            f"  [{label}] best params: {grid.best_params_}  "
            f"CV balanced-acc: {grid.best_score_:.3f}"
        )

        metrics[label] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "rule_baseline": {
                "accuracy": float(accuracy_score(y_test, rule_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, rule_pred)),
            },
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "random_state": random_state,
            "best_params": grid.best_params_,
            "cv_balanced_accuracy": float(grid.best_score_),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, labels=labels, zero_division=0
            ),
        }
        models[label] = best_model

    return models, metrics


def save_artifacts(
    models: dict[str, Pipeline],
    metrics: dict[str, dict[str, Any]],
    model_dir: str | Path = "models",
) -> None:
    """Persist trained models, metrics JSON, and confusion matrix images."""
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)

    for label, model in models.items():
        joblib.dump(model, output / f"{label}_model.joblib", compress=3)
        label_names = metrics[label]["labels"]
        matrix = np.asarray(metrics[label]["confusion_matrix"])
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names)
        display.plot(xticks_rotation=45, colorbar=False)
        plt.title(f"{label.replace('_', ' ').title()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output / f"{label}_confusion_matrix.png")
        plt.close()

    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("scikit-learn", "numpy", "scipy", "pandas", "joblib", "matplotlib")
        },
    }
    (output / "training_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )


def train_from_csv(
    csv_path: str | Path = "data/golfers.csv",
    model_dir: str | Path = "models",
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Train from an existing CSV, generating it first if needed."""
    path = Path(csv_path)
    if path.exists():
        golfers = pd.read_csv(path)
    else:
        golfers = save_dataset(path, n=DEFAULT_PROFILE_COUNT, seed=random_state)
    models, metrics = train_models(golfers, random_state=random_state)
    save_artifacts(models, metrics, model_dir=model_dir)
    return models, metrics


if __name__ == "__main__":
    print("Loading dataset and running hyperparameter search...")
    _, run_metrics = train_from_csv()
    for target, values in run_metrics.items():
        print(
            f"{target}: accuracy={values['accuracy']:.3f}  "
            f"CV balanced-acc={values.get('cv_balanced_accuracy', 'n/a')}"
        )
