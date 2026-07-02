import pandas as pd

from synthetic_data import generate_golfer_profiles
from train import LABEL_COLUMNS, normalize_label_columns, save_artifacts, train_models


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
