"""检验折外隔离、Shapley效率性质及模拟收益和分组口径。"""

import numpy as np
import pandas as pd
import pytest

from behavior_cleaning.business_analysis import (
    revenue_scenarios,
    sample_size_two_proportions,
    stable_arm,
    subgroup_metrics,
)
from behavior_cleaning.model_explanations import din_group_values, exact_group_shap
from behavior_cleaning.sequence_models import PurchaseSequenceModel
from behavior_cleaning.stacking_analysis import (
    fold_numeric,
    logit_features,
    transform_fold_numeric,
)


def test_outer_held_labels_do_not_change_fold_preprocessing():
    frame = pd.DataFrame(
        {
            "user_events": np.arange(120, dtype=float),
            "category_id": [str(i % 7) for i in range(120)],
        }
    )
    y = np.tile([0, 0, 0, 1], 30)
    train = np.arange(80)
    expected, preprocessing = fold_numeric(
        frame, y, train, ["user_events", "category_target_encoding"], 42
    )
    changed = y.copy()
    changed[80:] = 1 - changed[80:]
    actual, _ = fold_numeric(
        frame, changed, train, ["user_events", "category_target_encoding"], 42
    )
    np.testing.assert_array_equal(expected, actual)
    replay = transform_fold_numeric(frame.iloc[80:], preprocessing)
    np.testing.assert_array_equal(replay, expected[80:])


def test_exact_shap_linear_and_interaction():
    values = np.array(
        [
            3
            + 2 * bool(mask & 1)
            - 4 * bool(mask & 2)
            + 6 * bool(mask & 1) * bool(mask & 2)
            for mask in range(4)
        ]
    )
    actual = exact_group_shap(values, 2)
    np.testing.assert_allclose(actual, [5, -1])
    assert actual.sum() == values[-1] - values[0]
    with pytest.raises(ValueError):
        exact_group_shap([1, 2, 3], 2)


def test_din_shap_covers_all_inputs_and_adds_to_model_margin():
    import torch

    torch.set_num_threads(2)
    rng = np.random.RandomState(42)
    n, length, items = 8, 3, 2
    arrays = dict(
        x=rng.normal(size=(n, 4)).astype(np.float32),
        y=np.zeros(n, np.int64),
        row_users=np.arange(n),
        target_items=rng.randint(1, 9, n),
        target_categories=rng.randint(1, 6, n),
        sequence_items=rng.randint(1, 9, (n, length, items)),
        sequence_categories=rng.randint(1, 6, (n, length, items)),
        sequence_aux=rng.normal(size=(n, length, 5)).astype(np.float32),
        sequence_mask=np.ones((n, length), bool),
        matches=rng.randint(0, 2, (n, length)).astype(np.float32),
    )
    model = PurchaseSequenceModel("DIN", 10, 7, 4, embedding=4, hidden=8)
    shap, baseline, prediction, checks = din_group_values(
        model,
        arrays,
        np.array([6, 7]),
        np.array([0, 1, 2, 3]),
        ["user_events", "item_events", "pair_days", "category_target_encoding"],
    )
    assert shap.shape == (2, 6)
    np.testing.assert_allclose(shap.sum(1) + baseline, prediction, atol=1e-5)
    assert checks["additivity_max_error"] < 1e-5


def test_assignment_is_stable_by_user_and_roughly_balanced():
    assert stable_arm("42") == stable_arm(42)
    arms = [stable_arm(i) for i in range(10000)]
    assert 0.47 < arms.count("treatment") / len(arms) < 0.53


def test_sample_size_increases_for_smaller_effect():
    small = sample_size_two_proportions(0.2, 0.01)
    large = sample_size_two_proportions(0.2, 0.02)
    assert small > 3.5 * large
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.99, 0.02)


def test_subsidy_includes_natural_buyers_and_profit_identity():
    frame = revenue_scenarios(0.6)
    row = frame[
        (frame.policy == "coupon_marketing") & (frame.assumed_absolute_lift == 0.01)
    ].iloc[0]
    assert row.expected_incremental_buyers == 20
    assert row.expected_subsidy_cost == pytest.approx(2000 * 0.61 * 0.3 * 5)
    assert row.expected_incremental_contribution == pytest.approx(
        row.expected_incremental_net_gmv * 0.25 - row.expected_total_cost
    )
    assert row.expected_incremental_contribution < 0


def test_single_class_subgroup_does_not_report_invented_auc():
    result = subgroup_metrics(np.zeros(5), np.array([0.1, 0.8, 0.2, 0.9, 0.1]), 0.5)
    assert np.isnan(result["auc"]) and np.isnan(result["average_precision"])
    assert result["false_positive"] == 2
    assert result["true_negative"] == 3


def test_logit_is_monotonic_and_finite_at_endpoints():
    value = logit_features(np.array([0, 0.1, 0.5, 0.9, 1]))
    assert np.isfinite(value).all()
    assert (np.diff(value) > 0).all()
