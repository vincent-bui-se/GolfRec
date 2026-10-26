import json

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split

import train

from synthetic_data import generate_golfer_profiles
from train import LABEL_COLUMNS, build_base_pipeline, normalize_label_columns, save_artifacts, train_models


def test_base_pipeline_balances_classes():
    assert build_base_pipeline().named_steps["classifier"].class_weight == "balanced"


def test_cv_evaluates_the_same_training_procedure_as_the_saved_model(monkeypatch):
    monkeypatch.setattr(train, "PARAM_GRIDS", {
        label: {"classifier__n_estimators": [5], "classifier__max_depth": [3]}
        for label in LABEL_COLUMNS
    })
    golfers = normalize_label_columns(generate_golfer_profiles(n=1000, seed=21))
    models, metrics = train_models(golfers, random_state=21)

    for label, model in models.items():
        x_train, _, y_train, _ = train_test_split(
            golfers[train.INPUT_COLUMNS], golfers[label], test_size=0.2,
            random_state=21, stratify=golfers[label],
        )
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="balanced_accuracy")
        assert model.named_steps["classifier"].class_weight == "balanced"
        assert np.isclose(metrics[label]["cv_balanced_accuracy"], cv_scores.mean())
        assert 0 <= metrics[label]["rule_baseline"]["accuracy"] <= 1
        assert 0 <= metrics[label]["balanced_accuracy"] <= 1
        repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=21)
        repeated_scores = cross_val_score(model, x_train, y_train, cv=repeated, scoring="accuracy")
        assert np.isclose(metrics[label]["cv_accuracy_mean"], repeated_scores.mean())
        assert np.isclose(metrics[label]["cv_accuracy_std"], repeated_scores.std())


def test_train_models_returns_one_model_and_metric_bundle_per_label():
    golfers = generate_golfer_profiles(n=1000, seed=21)

    models, metrics = train_models(golfers, random_state=21)

    assert set(models.keys()) == set(LABEL_COLUMNS)
    assert set(metrics.keys()) == set(LABEL_COLUMNS)
    for label in LABEL_COLUMNS:
        assert 0 <= metrics[label]["accuracy"] <= 1
        assert "classification_report" in metrics[label]
        assert "confusion_matrix" in metrics[label]
        predictions = models[label].predict(golfers.head(3))
        assert len(predictions) == 3
        assert isinstance(predictions[0], str)


def test_save_artifacts_writes_metrics_and_confusion_matrix_images(tmp_path):
    golfers = generate_golfer_profiles(n=1000, seed=22)
    models, metrics = train_models(golfers, random_state=22)

    save_artifacts(models, metrics, tmp_path)

    assert (tmp_path / "metrics.json").exists()
    metadata = json.loads((tmp_path / "training_environment.json").read_text())
    assert metadata["packages"]["scikit-learn"]
    for label in LABEL_COLUMNS:
        assert (tmp_path / f"{label}_model.joblib").exists()
        assert (tmp_path / f"{label}_confusion_matrix.png").exists()


def test_normalize_label_columns_cleans_driver_loft_values_after_csv_reload(tmp_path):
    golfers = generate_golfer_profiles(n=1000, seed=23)
    csv_path = tmp_path / "golfers.csv"
    golfers.to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path)

    normalized = normalize_label_columns(reloaded)

    assert set(normalized["driver_loft"].unique()).issubset({"8", "9", "10.5", "12"})
    # fairway_wood_loft round-trips through the same float-then-string CSV
    # issue as driver_loft (16.5 -> "16.5", but 15 -> "15.0").
    assert set(normalized["fairway_wood_loft"].unique()).issubset({"15", "16.5", "18", "21"})
