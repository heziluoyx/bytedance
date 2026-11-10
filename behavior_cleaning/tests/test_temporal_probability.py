"""完整标签窗口、历史资格、概率校准和分组隔离。"""
import duckdb
import numpy as np
import pandas as pd

from behavior_cleaning.temporal_probability_data import (
    WINDOWS,
    calibration_user,
    snapshot,
    window_bounds,
)
from behavior_cleaning.temporal_probability_models import (
    apply_calibrator,
    fit_calibrator,
    probability_metrics,
)


def test_labels_are_mature_before_next_window():
    ranges = [window_bounds(cutoff) for _, cutoff in WINDOWS]
    assert ranges[0][0] == pd.Timestamp("2025-11-18")
    assert ranges[-1][-1] == pd.Timestamp("2025-12-19")
    for before, after in zip(ranges, ranges[1:]):
        assert before[-1] == after[1]
    assert all(
        (end - cutoff).days == 7 and (cutoff - start).days == 3
        for start, cutoff, end in ranges
    )


def test_meta_calibration_user_assignment_is_stable_and_exclusive():
    users = [str(i) for i in range(100)]
    cal = {u for u in users if calibration_user(u)}
    meta = set(users) - cal
    assert len(cal) > 20 and len(meta) > 20 and not cal & meta
    assert all(calibration_user(u) == calibration_user(u) for u in users)


def test_future_changes_only_labels_not_history_or_eligibility():
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE events(user_id VARCHAR,item_id VARCHAR,category_id VARCHAR,behavior_type INTEGER,event_time TIMESTAMP)"
    )
    c.execute(
        "INSERT INTO events VALUES ('a','1','c',1,'2025-11-18'),('b','2','c',3,'2025-11-20'),('c','3','c',1,'2025-11-20'),('a','1','c',4,'2025-11-23')"
    )
    before, _ = snapshot(c, "2025-11-21", 100)
    c.execute(
        "INSERT INTO events VALUES ('z','future_only','c',4,'2025-11-22'),('b','2','c',1,'2025-11-28'),('a','1','c',1,'2025-11-17'),('b','2','c',4,'2025-11-25')"
    )
    after, _ = snapshot(c, "2025-11-21", 100)
    pd.testing.assert_frame_equal(
        before.drop(columns="target").sort_values("user_id").reset_index(drop=True),
        after.drop(columns="target").sort_values("user_id").reset_index(drop=True),
    )
    assert before.set_index("user_id").loc["b", "target"] == 0
    assert after.set_index("user_id").loc["b", "target"] == 1
    c.close()


def test_calibration_produces_continuous_probabilities_not_hard_labels():
    rng = np.random.default_rng(42)
    score = rng.uniform(0.1, 0.9, 2000)
    y = (rng.random(2000) < score * 0.2).astype(int)
    inflated = np.clip(score + 0.09, 0, 1)
    model = fit_calibrator(y[:1000], inflated[:1000])
    p = apply_calibrator(model, inflated[1000:])
    assert ((p > 0) & (p < 1)).all() and len(np.unique(p)) > 100
    assert (
        probability_metrics(y[1000:], p)["brier"]
        < probability_metrics(y[1000:], inflated[1000:])["brier"]
    )


def test_probability_metrics_distinguish_ranking_from_calibration():
    y = np.array([0, 0, 1, 1])
    a = probability_metrics(y, np.array([0.01, 0.02, 0.03, 0.04]))
    b = probability_metrics(y, np.array([0.01, 0.02, 0.98, 0.99]))
    assert a["auc"] == b["auc"] == 1
    assert a["log_loss"] > b["log_loss"] and a["brier"] > b["brier"]


def test_din_shap_interface_and_calibrated_logit_additivity():
    from types import SimpleNamespace

    import torch

    from behavior_cleaning.temporal_probability_analysis import (
        calibrated_din_explanation,
    )

    class ToyModel(torch.nn.Module):
        def forward(
            self,
            numeric,
            target_item,
            target_category,
            history_item,
            history_category,
            auxiliary,
            mask,
        ):
            return numeric.sum(1) + target_item.float() + auxiliary.sum((1, 2))

    arrays = {
        "x": np.arange(16, dtype=np.float32).reshape(4, 4) / 10,
        "y": np.array([0, 1, 0, 1]),
        "row_users": np.arange(4),
        "target_items": np.arange(4),
        "target_categories": np.arange(4),
        "sequence_items": np.ones((4, 2, 1), dtype=np.int64),
        "sequence_categories": np.ones((4, 2, 1), dtype=np.int64),
        "sequence_aux": np.ones((4, 2, 5), dtype=np.float32),
        "sequence_mask": np.ones((4, 2), dtype=bool),
        "matches": np.ones((4, 2), dtype=np.float32),
    }
    calibrator = SimpleNamespace(coef_=np.array([[0.7]]), intercept_=np.array([-1.2]))
    values, baseline, full, checks = calibrated_din_explanation(
        ToyModel(),
        arrays,
        np.array([2, 3]),
        np.array([0, 1]),
        ["user_a", "item_b", "pair_c", "category_target_encoding"],
        calibrator,
    )
    assert values.shape == (2, 6)
    np.testing.assert_allclose(baseline + values.sum(1), full, atol=1e-6)
    assert checks["additivity_max_error"] < 1e-5


def test_daily_promotion_context_counts_events_users_and_buyers():
    from behavior_cleaning.temporal_probability_analysis import daily_context

    c = duckdb.connect()
    c.execute(
        "CREATE TABLE events(user_id VARCHAR,behavior_type INTEGER,event_time TIMESTAMP)"
    )
    c.execute(
        "INSERT INTO events VALUES ('a',1,'2025-12-12'),('a',4,'2025-12-12'),('a',4,'2025-12-12'),('b',1,'2025-12-13')"
    )
    daily = daily_context(c)
    assert daily.events.tolist() == [3, 1]
    assert daily.dau.tolist() == [1, 1]
    assert daily.buy_events.tolist() == [2, 0]
    assert daily.buyers.tolist() == [1, 0]
    c.close()
