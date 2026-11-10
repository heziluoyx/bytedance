"""发布接口的隔离、数据完整性、预处理复用及文件保护。"""
import json
import subprocess
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from behavior_cleaning import TrainingConfig, load_predictor
from behavior_cleaning.inference import PurchasePredictor, transform_features
from behavior_cleaning.promotion_robustness import NUMERIC
from behavior_cleaning.temporal_probability_data import snapshot
from behavior_cleaning.temporal_probability_models import fit_calibrator
from behavior_cleaning.toolkit_cli import main


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "run"
    (root / "dataset").mkdir(parents=True)
    (root / "models").mkdir()
    numeric = pd.DataFrame({n: np.arange(8) for n in NUMERIC})
    encoded = np.full(8, 0.25)
    values = np.column_stack([np.log1p(numeric), encoded])
    scaler = StandardScaler().fit(values)
    x = scaler.transform(values)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = LogisticRegression().fit(x, y)
    preprocessing = {
        "features": NUMERIC + ["category_target_encoding"],
        "medians": numeric.median(),
        "scaler": scaler,
        "target_encoder": {"mapping": {"c": 0.25}, "prior": 0.25},
        "item_vocab": {"i": 2},
        "category_vocab": {"c": 2},
    }
    summary = {
        "features": preprocessing["features"],
        "item_vocabulary": 3,
        "category_vocabulary": 3,
        "windows": [
            {"window": "meta_calibration", "label_end_exclusive": "2025-12-12"}
        ],
    }
    joblib.dump(preprocessing, root / "dataset/preprocessing.joblib")
    (root / "dataset/summary.json").write_text(json.dumps(summary), encoding="utf-8")
    joblib.dump(model, root / "models/LogisticRegression.joblib")
    joblib.dump(
        {"LogisticRegression": fit_calibrator(y, model.predict_proba(x)[:, 1])},
        root / "models/probability_calibrators.joblib",
    )
    return root, preprocessing


def events():
    return pd.DataFrame(
        {
            "user_id": ["u", "u", "v"],
            "item_id": ["i", "i", "j"],
            "category_id": ["c"] * 3,
            "behavior_type": [1, 3, 1],
            "timestamp": pd.to_datetime(
                ["2025-12-09 00:00", "2025-12-11 23:00", "2025-12-10 00:00"]
            ),
        }
    )


def test_unlabeled_snapshot_ignores_future_events():
    frame = events().rename(columns={"timestamp": "event_time"})
    c = duckdb.connect()
    try:
        c.execute("CREATE TABLE events AS SELECT * FROM frame")
        before, _ = snapshot(c, "2025-12-12", 100, labeled=False)
        c.execute(
            "INSERT INTO events VALUES ('u','i','c',4,'2025-12-13'),('future','x','c',4,'2025-12-14')"
        )
        after, _ = snapshot(c, "2025-12-12", 100, labeled=False)
        assert "target" not in before and set(before.user_id) == {"u", "v"}
        pd.testing.assert_frame_equal(before, after)
    finally:
        c.close()


def test_transform_never_uses_target_and_handles_unknown_category(bundle):
    _, prep = bundle
    frame = pd.DataFrame({n: [1.0, np.nan] for n in NUMERIC})
    frame["category_id"] = ["c", "unknown"]
    before = transform_features(frame, prep)
    frame["target"] = [999, -999]
    after = transform_features(frame, prep)
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(after).all() and after.shape == (2, 26)


def test_missing_features_fail_with_clear_message(bundle):
    with pytest.raises(ValueError, match="缺少特征列"):
        transform_features(pd.DataFrame({"category_id": ["c"]}), bundle[1])


def test_predict_without_any_future_data(bundle, tmp_path):
    source = tmp_path / "history.parquet"
    events().to_parquet(source, index=False)
    p = load_predictor(bundle[0], models=["LogisticRegression"])
    result = p.predict(source, "2025-12-12")
    assert len(result) == 2 and "target" not in result
    assert result.LogisticRegression_probability.between(
        0, 1, inclusive="neither"
    ).all()


def test_predict_rejects_stale_history(bundle, tmp_path):
    source = tmp_path / "old.parquet"
    events().iloc[:1].to_parquet(source, index=False)
    with pytest.raises(ValueError, match="历史观察期不完整"):
        load_predictor(bundle[0], models=["LogisticRegression"]).predict(
            source, "2025-12-12"
        )


@pytest.mark.parametrize(
    "cutoff", ["2025-12-11", "2025-12-12 00:01", "2025-12-12T00:00:00+08:00"]
)
def test_predict_rejects_leaky_or_nonhour_cutoff(bundle, cutoff):
    with pytest.raises(ValueError):
        load_predictor(bundle[0], models=["LogisticRegression"]).predict(
            "not_read.parquet", cutoff
        )


def test_predict_rejects_bad_budget_before_file_access(bundle):
    with pytest.raises(ValueError, match="max_pairs"):
        load_predictor(bundle[0], models=["LogisticRegression"]).predict(
            "not_read.parquet", "2025-12-12", max_pairs=0
        )


def test_empty_and_nonfinite_array_contract(bundle):
    p = load_predictor(bundle[0], models=["LogisticRegression"])
    assert p.predict_arrays({"x": np.empty((0, 26))}).shape == (0, 1)
    with pytest.raises(ValueError, match="非有限"):
        p.predict_arrays({"x": np.full((2, 26), np.nan)})


def test_model_bundle_can_reload_and_protects_existing_destination(bundle, tmp_path):
    p = load_predictor(bundle[0], models=["LogisticRegression"])
    target = p.export_bundle(tmp_path / "asset")
    reloaded = load_predictor(target, models=["LogisticRegression"])
    x = np.zeros((3, 26))
    pd.testing.assert_frame_equal(
        p.predict_arrays({"x": x}), reloaded.predict_arrays({"x": x})
    )
    assert (target / "bundle_manifest.json").is_file()
    assert not (target / "dataset/samples.parquet").exists()
    with pytest.raises(FileExistsError):
        p.export_bundle(target)


@pytest.mark.parametrize("models", [["bad"], ["XGBoost", "XGBoost"], []])
def test_unknown_or_duplicate_models_fail(bundle, models):
    with pytest.raises(ValueError, match="模型名"):
        PurchasePredictor(bundle[0], models=models)


def test_training_budget_validation(tmp_path):
    with pytest.raises(ValueError):
        TrainingConfig(tmp_path / "source", tmp_path / "output", trials=0)


def test_corrupted_bundle_is_rejected_before_unpickling(bundle, tmp_path):
    target = load_predictor(bundle[0], models=["LogisticRegression"]).export_bundle(
        tmp_path / "asset"
    )
    (target / "dataset/preprocessing.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="校验失败"):
        load_predictor(target, models=["LogisticRegression"])


def test_cli_does_not_overwrite_prediction(tmp_path):
    output = tmp_path / "pred.csv"
    output.write_text("original")
    with pytest.raises(SystemExit):
        main(
            [
                "predict",
                "--run-dir",
                "missing",
                "--source",
                "missing",
                "--cutoff",
                "2025-12-12",
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == "original"


def test_package_import_is_lightweight():
    code = "import sys,behavior_cleaning;assert 'torch' not in sys.modules;assert behavior_cleaning.__version__=='1.1.0'"
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=20)
