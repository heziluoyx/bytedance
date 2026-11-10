"""校准概率的解释、错误诊断与不含真实触达的业务模拟。"""
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import norm

from behavior_cleaning.model_explanations import GROUP_NAMES, din_group_values
from behavior_cleaning.temporal_probability_data import open_events
from behavior_cleaning.temporal_probability_models import (
    load_sequence,
    probability_metrics,
)


def diagnostic_metrics(y, p, threshold):
    if len(np.unique(y)) < 2:
        predicted = p >= threshold
        return {
            "auc": None,
            "rows": len(y),
            "positives": int(y.sum()),
            "mean_probability": float(np.mean(p)),
            "brier": float(np.mean((p - y) ** 2)),
            "false_positives": int(((y == 0) & predicted).sum()),
            "false_negatives": int(((y == 1) & ~predicted).sum()),
        }
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        **probability_metrics(y, p, threshold),
    }


def cluster_auc_difference(y, a, b, users, repeats=200):
    """同一用户的多对样本整体重采样，避免把每行当作独立用户。"""
    from sklearn.metrics import roc_auc_score

    unique, inverse = np.unique(users, return_inverse=True)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(repeats):
        weights = np.bincount(
            rng.integers(len(unique), size=len(unique)), minlength=len(unique)
        )[inverse]
        if np.sum(weights * y) == 0 or np.sum(weights * (1 - y)) == 0:
            continue
        differences.append(
            roc_auc_score(y, a, sample_weight=weights)
            - roc_auc_score(y, b, sample_weight=weights)
        )
    return {
        "delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "user_cluster_bootstrap_95ci": [
            float(v) for v in np.quantile(differences, [0.025, 0.975])
        ],
        "repeats": len(differences),
        "status": "conditional on this fixed time window; not causal or fresh repeated holdout",
    }


def required_per_arm(p, absolute_lift):
    q = min(p + absolute_lift, 0.999999)
    middle = (p + q) / 2
    return int(
        np.ceil(
            (
                (
                    norm.ppf(0.975) * np.sqrt(2 * middle * (1 - middle))
                    + norm.ppf(0.8) * np.sqrt(p * (1 - p) + q * (1 - q))
                )
                / (q - p)
            )
            ** 2
        )
    )


def calibrated_din_explanation(
    model, arrays, indices, background, features, calibrator
):
    """旧解释器已返回六组Shapley，直接仿射变换到校准logit。"""
    values, raw_baseline, raw_full, original_checks = din_group_values(
        model, arrays, indices, background, features
    )
    slope = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    return (
        values * slope,
        raw_baseline * slope + intercept,
        raw_full * slope + intercept,
        original_checks,
    )


def daily_context(connection):
    return connection.execute(
        '''SELECT CAST(event_time AS DATE) AS "day",COUNT(*) events,
        COUNT(DISTINCT user_id) dau,SUM(behavior_type=4) buy_events,
        COUNT(DISTINCT CASE WHEN behavior_type=4 THEN user_id END) buyers
        FROM events GROUP BY "day" ORDER BY "day"'''
    ).df()


def explain_and_simulate(source, output, arrays, summary, sequence):
    output = Path(output)
    directory = output / "interpretability"
    directory.mkdir(exist_ok=True)
    prediction = pd.read_parquet(output / "all_window_predictions.parquet")
    frame = pd.read_parquet(output / "dataset/samples.parquet")
    features = summary["features"]
    test = arrays["test"]
    train = arrays["train"]
    test_frame = frame.iloc[test].reset_index(drop=True)
    test_pred = prediction.iloc[test].reset_index(drop=True)
    calibrators = joblib.load(output / "models/probability_calibrators.joblib")
    thresholds = json.loads(
        (output / "models/frozen_thresholds.json").read_text(encoding="utf-8")
    )
    lr = joblib.load(output / "models/LogisticRegression.joblib")
    a = float(calibrators["LogisticRegression"].coef_[0, 0])
    pd.DataFrame(
        {
            "feature": features,
            "raw_coefficient": lr.coef_[0],
            "calibrated_logit_coefficient": lr.coef_[0] * a,
        }
    ).sort_values("calibrated_logit_coefficient", key=abs, ascending=False).to_csv(
        directory / "logistic_coefficients.csv", index=False
    )
    rng = np.random.default_rng(42)
    indices = rng.choice(test, min(2000, len(test)), replace=False)
    model = joblib.load(output / "models/XGBoost.joblib")
    matrix = xgb.DMatrix(arrays["x"][indices])
    contributions = model.get_booster().predict(
        matrix, pred_contribs=True, approx_contribs=False
    )
    margins = model.get_booster().predict(matrix, output_margin=True)
    xgb_error = float(np.max(np.abs(contributions.sum(1) - margins)))
    a = float(calibrators["XGBoost"].coef_[0, 0])
    b = float(calibrators["XGBoost"].intercept_[0])
    contributions = contributions * a
    contributions[:, -1] += b
    pd.DataFrame(
        {
            "feature": features,
            "mean_absolute_calibrated_logit_shap": np.abs(contributions[:, :-1]).mean(
                0
            ),
        }
    ).sort_values("mean_absolute_calibrated_logit_shap", ascending=False).to_csv(
        directory / "xgboost_shap_importance.csv", index=False
    )
    np.savez_compressed(
        directory / "xgboost_shap_values.npz",
        indices=indices,
        contributions=contributions,
    )
    model = load_sequence("DIN", sequence["DIN"], arrays, summary, output)
    explain = rng.choice(test, min(96, len(test)), replace=False)
    background = rng.choice(train, min(8, len(train)), replace=False)
    logging.info("计算DIN六组输入的完整Shapley，样本%d 背景%d", len(explain), len(background))
    calibrated_values, baseline, full, background_checks = calibrated_din_explanation(
        model, arrays, explain, background, features, calibrators["DIN"]
    )
    din_error = float(np.max(np.abs(baseline + calibrated_values.sum(1) - full)))
    pd.DataFrame(
        {
            "group": GROUP_NAMES,
            "mean_absolute_calibrated_logit_shap": np.abs(calibrated_values).mean(0),
        }
    ).sort_values("mean_absolute_calibrated_logit_shap", ascending=False).to_csv(
        directory / "din_group_shap_importance.csv", index=False
    )
    np.savez_compressed(
        directory / "din_group_shap_values.npz",
        indices=explain,
        background=background,
        values=calibrated_values,
        baseline=baseline,
        calibrated_logit=full,
    )
    model = None
    torch.set_num_threads(2)
    training_users = set(frame.iloc[train].user_id)
    training_items = set(frame.iloc[train].item_id)
    tail_threshold = float(
        frame.iloc[train].drop_duplicates("item_id").item_events.quantile(0.25)
    )
    groups = {
        "all": np.ones(len(test), bool),
        "unseen_user_in_training": ~test_frame.user_id.isin(training_users).to_numpy(),
        "unseen_item_in_training": ~test_frame.item_id.isin(training_items).to_numpy(),
        "long_tail_item": test_frame.item_events.le(tail_threshold).to_numpy(),
        "known_user_and_item": (
            test_frame.user_id.isin(training_users)
            & test_frame.item_id.isin(training_items)
        ).to_numpy(),
    }
    diagnostics = []
    budget = []
    for name, threshold in thresholds.items():
        p = test_pred[name + "_probability"].to_numpy()
        y = test_frame.target.to_numpy()
        for group, mask in groups.items():
            if mask.any():
                diagnostics.append(
                    {
                        "model": name,
                        "group": group,
                        **diagnostic_metrics(y[mask], p[mask], threshold),
                    }
                )
        ranked = np.argsort(-p, kind="stable")
        for fraction in [0.01, 0.05, 0.1, 0.2]:
            ix = ranked[: max(1, int(len(ranked) * fraction))]
            budget.append(
                {
                    "model": name,
                    "candidate_fraction": fraction,
                    "rows": len(ix),
                    "precision": float(y[ix].mean()),
                    "recall": float(y[ix].sum() / max(y.sum(), 1)),
                    "expected_positive_pairs": float(p[ix].sum()),
                    "observed_positive_pairs": int(y[ix].sum()),
                }
            )
    pd.DataFrame(diagnostics).to_csv(output / "error_group_metrics.csv", index=False)
    pd.DataFrame(budget).to_csv(output / "ranking_budget_comparison.csv", index=False)
    p = test_pred["Stacking_probability"].to_numpy()
    predicted = p >= thresholds["Stacking"]
    test_frame["probability"] = p
    test_frame["error_type"] = np.select(
        [(test_frame.target == 1) & ~predicted, (test_frame.target == 0) & predicted],
        ["false_negative", "false_positive"],
        default="correct",
    )
    test_frame.groupby("error_type")[
        [
            "pair_cart",
            "pair_recent_events",
            "last_time_days_ago",
            "item_events",
            "user_buys",
            "probability",
        ]
    ].median().to_csv(output / "stacking_error_profile.csv")
    cluster = cluster_auc_difference(
        test_frame.target.to_numpy(),
        test_pred.Stacking_probability.to_numpy(),
        test_pred.XGBoost_probability.to_numpy(),
        test_frame.user_id.to_numpy(),
    )
    validation_note = {
        "xgboost_raw_margin_additivity_max_error": xgb_error,
        "din_calibrated_logit_additivity_max_error": din_error,
        "din_samples": len(explain),
        "background_samples": len(background),
        "din_explanation": "exact six-group interventional Shapley; calibrated logit units; not individual embedding coordinates or causal effects",
        "long_tail_history_event_threshold": tail_threshold,
        "true_zero_history_candidate_rows": 0,
        "cold_start_limit": "recent-interaction candidates exclude true zero-history users/items; unseen in training is only a proxy",
        "stacking_vs_xgboost": cluster,
    }
    (output / "explanation_checks.json").write_text(
        json.dumps(validation_note, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 营销资格只由更早校准窗口的用户最高得分阈值决定，不将其当作用户购买概率。
    cal = frame.iloc[arrays["calibration"]].copy()
    cal["score"] = prediction.iloc[
        arrays["calibration"]
    ].Stacking_probability.to_numpy()
    eligibility_threshold = float(cal.groupby("user_id").score.max().quantile(0.8))
    users = (
        test_frame.groupby("user_id")
        .probability.max()
        .rename("ranking_score")
        .reset_index()
    )
    users["eligible"] = users.ranking_score >= eligibility_threshold
    users["arm"] = users.user_id.map(
        lambda u: "treatment"
        if int(hashlib.sha256(("ab20261109:" + u).encode()).hexdigest()[:8], 16) % 2
        else "control"
    )
    c = open_events(source, output / "duckdb_temp")
    daily = daily_context(c)
    daily.to_csv(output / "daily_promotion_context.csv", index=False)
    c.register("test_candidates", test_frame[["user_id", "item_id"]])
    first_dates = c.execute(
        """SELECT c.user_id,c.item_id,MIN(e.event_time) first_purchase
      FROM test_candidates c JOIN events e USING(user_id,item_id)
      WHERE e.event_time>='2025-12-12' AND e.event_time<'2025-12-19' AND e.behavior_type=4
      GROUP BY c.user_id,c.item_id"""
    ).df()
    first_dates["day"] = pd.to_datetime(first_dates.first_purchase).dt.strftime(
        "%Y-%m-%d"
    )
    date_counts = (
        first_dates.groupby("day").size().rename("first_purchase_pairs").reset_index()
    )
    date_counts["share_of_positive_pairs"] = date_counts.first_purchase_pairs / len(
        first_dates
    )
    date_counts.to_csv(output / "test_first_purchase_dates.csv", index=False)
    buyers = set(
        c.execute(
            "SELECT DISTINCT user_id FROM events WHERE event_time>='2025-12-12' AND event_time<'2025-12-19' AND behavior_type=4"
        ).fetchnumpy()["user_id"]
    )
    c.close()
    users["historical_any_item_purchase"] = users.user_id.isin(buyers)
    users.to_parquet(output / "simulated_ab_assignment.parquet", index=False)
    qualified = users[users.eligible]
    baseline_rate = float(qualified.historical_any_item_purchase.mean())
    power = [
        {
            "absolute_lift": delta,
            "required_users_per_arm": required_per_arm(baseline_rate, delta),
        }
        for delta in [0.005, 0.01, 0.02]
    ]
    # 金额完全是可调整假设；补贴成本覆盖自然购买者，非仅增量购买者。
    scenarios = []
    for lift in [0.005, 0.01, 0.02]:
        reached = 2000
        incremental_buyers = reached * lift
        gmv = incremental_buyers * 150 * 0.9
        for coupon in [False, True]:
            cost = reached * (0.2 if coupon else 0.05)
            subsidy = reached * min(baseline_rate + lift, 1) * 5 * 0.3 if coupon else 0
            scenarios.append(
                {
                    "assumed_absolute_lift": lift,
                    "coupon": coupon,
                    "eligible_users": 10000,
                    "reached_users": reached,
                    "incremental_buyers": incremental_buyers,
                    "assumed_net_gmv": gmv,
                    "contact_cost": cost,
                    "coupon_cost": subsidy,
                    "assumed_incremental_contribution": gmv * 0.25 - cost - subsidy,
                }
            )
    pd.DataFrame(scenarios).to_csv(
        output / "assumed_business_scenarios.csv", index=False
    )
    business = {
        "execution": "simulation only; no real outreach or causal effect measured",
        "cohort_users": len(users),
        "eligible_users": len(qualified),
        "eligibility_score_threshold": eligibility_threshold,
        "score_is": "maximum sampled pair score for ranking only, not user-level purchase probability",
        "assignment": "stable user hash 50/50; random assignment must occur before intervention in real experiment",
        "primary_outcome": "paid user conversion ITT in a future real experiment; current baseline is any-item purchase-behavior proxy",
        "proxy_baseline_rate": baseline_rate,
        "power_alpha": 0.05,
        "power": 0.8,
        "power_scenarios": power,
        "enrollment_days": 14,
        "followup_days": 7,
        "primary_readout_days": 21,
        "refund_maturity_days_after_purchase": 30,
        "assumptions": {
            "qualified_population": 10000,
            "reach_fraction": 0.2,
            "aov": 150,
            "margin_before_coupon": 0.25,
            "refund_fraction": 0.1,
            "orders_per_incremental_buyer": 1,
            "coupon_value": 5,
            "redemption_fraction": 0.3,
        },
        "interpretation": "uplift/AOV/margin/refund and costs are assumptions; predicted propensity is not incremental treatment effect",
    }
    (output / "business_simulation.json").write_text(
        json.dumps(business, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("解释、错误分析和业务模拟完成")
