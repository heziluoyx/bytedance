"""独立传统模型的标签隔离、接口一致性和报告输出。"""
import numpy as np
import pandas as pd

from behavior_cleaning.report_utils import save_report, table
from behavior_cleaning.traditional_dataset import encode_category, split_indices
from behavior_cleaning.traditional_models import choose_threshold, classifier, metrics


def test_pair_split_is_disjoint_and_stratified():
    y = np.tile([0] * 19 + [1], 100)
    train, valid, test = split_indices(y)
    assert [len(train), len(valid), len(test)] == [1400, 400, 200]
    assert not set(train) & set(valid)
    assert not set(train) & set(test)
    assert not set(valid) & set(test)
    assert [y[i].mean() for i in [train, valid, test]] == [0.05] * 3


def test_target_encoder_does_not_read_held_labels():
    y = np.tile([0, 1], 30)
    train = np.arange(40)
    category = np.array(["a", "b", "c"] * 20)
    before, mapping = encode_category(category, y, train)
    changed = y.copy()
    changed[40:] = 1 - changed[40:]
    after, changed_mapping = encode_category(category, changed, train)
    np.testing.assert_array_equal(before, after)
    assert mapping == changed_mapping


def test_all_three_classifiers_fit_and_expose_probabilities():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 1).astype(int)
    for name in ["LogisticRegression", "XGBoost", "LightGBM"]:
        for strategy in ["none", "class_weight"]:
            model = classifier(name, {}, strategy, (y == 0).sum() / y.sum())
            model.fit(x, y)
            probability = model.predict_proba(x)[:, 1]
            assert probability.shape == (200,)
            assert np.isfinite(probability).all()
            assert ((0 <= probability) & (probability <= 1)).all()
            assert metrics(y, probability)["auc"] > 0.8


def test_validation_threshold_finds_non_default_operating_point():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.01, 0.02, 0.03, 0.04])
    threshold = choose_threshold(y, probability)
    assert threshold == 0.03
    assert metrics(y, probability, threshold)["f1"] == 1
    assert metrics(y, probability)["recall"] == 0


def test_report_writer_escapes_html_and_keeps_metrics(tmp_path):
    path = tmp_path / "report"
    lines = ["# 模型评估", "<script>unsafe</script>"]
    lines += table(pd.DataFrame({"auc": [0.85]}), {"auc": "AUC"})
    save_report(path, lines)
    html = path.with_suffix(".html").read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "0.8500" in html
    assert path.with_suffix(".md").exists()
