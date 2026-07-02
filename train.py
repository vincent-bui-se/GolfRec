"""Train and evaluate Random Forest models for golf fitting labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from preprocess import INPUT_COLUMNS, make_feature_frame
from synthetic_data import save_dataset


LABEL_COLUMNS = [
    "driver_loft",
    "shaft_flex",
    "iron_category",
]


def _clean_driver_loft(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def normalize_label_columns(golfers: pd.DataFrame) -> pd.DataFrame:
    """Keep labels as display-ready strings after CSV round-trips."""
    normalized = golfers.copy()
    normalized["driver_loft"] = normalized["driver_loft"].map(_clean_driver_loft)
    for label in ["shaft_flex", "iron_category"]:
        normalized[label] = normalized[label].astype(str)
    return normalized


# Per-label hyperparameter search grids.
# Narrower grids keep search fast; each entry is ~16 fits on 75% of 2 000 rows.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "driver_loft": {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [10, 14, None],
        "classifier__min_samples_leaf": [1, 2],
    },
    "shaft_flex": {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [10, 14, None],
        "classifier__min_samples_leaf": [1, 2],
    },
    "iron_category": {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [12, 16, None],
        "classifier__min_samples_leaf": [1, 2],
    },
}


def build_base_pipeline(random_state: int = 42) -> Pipeline:
    """Create the shared preprocessing + Random Forest pipeline (untuned)."""
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(make_feature_frame, validate=False)),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_models(
    golfers: pd.DataFrame,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Train one tuned model per target label and return models plus metrics.

    Uses GridSearchCV with 5-fold stratified cross-validation to find the best
    Random Forest hyperparameters independently for each label.  class_weight
    is set to 'balanced' so minority classes (e.g. X-flex, blade irons) are
    not ignored in favour of the most common labels.
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

        # Apply balanced class weights computed from the FULL training set as
        # per-sample weights so that the refit is not sensitive to fold subsets.
        best_model.named_steps["classifier"].set_params(class_weight=None)
        sample_weights = compute_sample_weight("balanced", y=y_train)
        best_model.fit(x_train, y_train, classifier__sample_weight=sample_weights)

        y_pred = best_model.predict(x_test)
        labels = sorted(y.unique())

        print(
            f"  [{label}] best params: {grid.best_params_}  "
            f"CV balanced-acc: {grid.best_score_:.3f}"
        )

        metrics[label] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
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
        joblib.dump(model, output / f"{label}_model.joblib")
        label_names = metrics[label]["labels"]
        matrix = np.asarray(metrics[label]["confusion_matrix"])
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names)
        display.plot(xticks_rotation=45, colorbar=False)
        plt.title(f"{label.replace('_', ' ').title()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output / f"{label}_confusion_matrix.png")
        plt.close()

    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


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
        golfers = save_dataset(path, n=2000, seed=random_state)
    models, metrics = train_models(golfers, random_state=random_state)
    save_artifacts(models, metrics, model_dir=model_dir)
    return models, metrics


if __name__ == "__main__":
    print("Generating dataset and running hyperparameter search (this may take ~60 s)...")
    _, run_metrics = train_from_csv()
    for target, values in run_metrics.items():
        print(
            f"{target}: accuracy={values['accuracy']:.3f}  "
            f"CV balanced-acc={values.get('cv_balanced_accuracy', 'n/a')}"
        )
