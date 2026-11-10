"""冻结融合结果的分群诊断、用户随机分组和可审计收益情景测算。"""

import hashlib
import json
import logging
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import average_precision_score, roc_auc_score

from behavior_cleaning.model_experiments import metrics
from behavior_cleaning.model_explanations import run_explanations


def stable_arm(user_id, seed="douyin_ab_20261102"):
    """同一用户固定分组；仅输出模拟分组，不调用触达或投放服务。"""
    value = int.from_bytes(
        hashlib.sha256((seed + ":" + str(user_id)).encode()).digest()[:8], "big"
    )
    return "treatment" if value < 2**63 else "control"


def sample_size_two_proportions(p0, absolute_lift, alpha=0.05, power=0.8):
    """等组、双侧两独立比例正态近似，每组人数；非订单次数。"""
    p1 = p0 + absolute_lift
    if not (0 < p0 < p1 < 1):
        raise ValueError("需要0<p0<p0+lift<1")
    pooled = (p0 + p1) / 2
    numerator = (
        norm.ppf(1 - alpha / 2) * math.sqrt(2 * pooled * (1 - pooled))
        + norm.ppf(power) * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    ) ** 2
    return int(math.ceil(numerator / absolute_lift**2))


def revenue_scenarios(
    p0, eligible=10000, reach_fraction=0.2, aov=150, margin=0.25, refund_rate=0.1
):
    """收益来自假设的因果提升；优惠成本覆盖全部领券成交，而非仅增量成交。"""
    if not 0 <= p0 < 1 or not 0 <= refund_rate < 1 or not 0 <= reach_fraction <= 1:
        raise ValueError("概率参数超出范围")
    rows = []
    reached = eligible * reach_fraction
    for uplift in [0.005, 0.01, 0.02]:
        if p0 + uplift >= 1:
            continue
        for policy, contact_cost, coupon, redemption in [
            ("recommendation_without_coupon", 0.05, 0, 0),
            ("coupon_marketing", 0.2, 5, 0.3),
        ]:
            incremental_buyers = reached * uplift
            net_gmv = incremental_buyers * aov * (1 - refund_rate)
            subsidy = reached * (p0 + uplift) * redemption * coupon
            spend = subsidy + reached * contact_cost
            contribution = net_gmv * margin - spend
            # 设增量毛利=补贴+触达成本，求盈亏平衡绝对转化提升。
            denominator = aov * (1 - refund_rate) * margin - redemption * coupon
            break_even = (
                (p0 * redemption * coupon + contact_cost) / denominator
                if denominator > 0
                else None
            )
            rows.append(
                dict(
                    policy=policy,
                    eligible_users=eligible,
                    reach_fraction=reach_fraction,
                    reached_users=reached,
                    baseline_purchase_proxy=p0,
                    assumed_absolute_lift=uplift,
                    assumed_aov=aov,
                    assumed_margin_rate=margin,
                    assumed_refund_rate=refund_rate,
                    assumed_contact_cost=contact_cost,
                    assumed_coupon=coupon,
                    assumed_coupon_redemption=redemption,
                    expected_incremental_buyers=incremental_buyers,
                    expected_incremental_net_gmv=net_gmv,
                    expected_subsidy_cost=subsidy,
                    expected_total_cost=spend,
                    expected_incremental_contribution=contribution,
                    contribution_to_cost_ratio=contribution / spend if spend else None,
                    break_even_absolute_lift=break_even,
                )
            )
    return pd.DataFrame(rows)


def subgroup_metrics(y, p, threshold):
    label = p >= threshold
    basic = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "predicted_positives": int(label.sum()),
        "true_positive": int(((y == 1) & label).sum()),
        "false_positive": int(((y == 0) & label).sum()),
        "false_negative": int(((y == 1) & ~label).sum()),
        "true_negative": int(((y == 0) & ~label).sum()),
    }
    if len(y) == 0:
        return {
            **basic,
            "auc": np.nan,
            "average_precision": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }
    basic.update(
        {
            "precision": basic["true_positive"] / max(int(label.sum()), 1),
            "recall": basic["true_positive"] / max(int(y.sum()), 1),
        }
    )
    basic["f1"] = (
        2
        * basic["precision"]
        * basic["recall"]
        / max(basic["precision"] + basic["recall"], 1e-15)
    )
    basic["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    basic["average_precision"] = (
        float(average_precision_score(y, p)) if y.sum() else np.nan
    )
    return basic


def clustered_auc_difference(y, candidate, reference, users, repeats=500):
    """按用户成簇的配对bootstrap；区间仅用于探索，未做多重比较校正。"""
    unique, inverse = np.unique(np.asarray(users), return_inverse=True)
    rng = np.random.RandomState(20261102)
    differences = []
    for _ in range(repeats):
        weight = np.bincount(
            rng.randint(0, len(unique), len(unique)), minlength=len(unique)
        )[inverse]
        if (weight[y == 1].sum() == 0) or (weight[y == 0].sum() == 0):
            continue
        differences.append(
            roc_auc_score(y, candidate, sample_weight=weight)
            - roc_auc_score(y, reference, sample_weight=weight)
        )
    return {
        "point_difference": float(
            roc_auc_score(y, candidate) - roc_auc_score(y, reference)
        ),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "valid_bootstraps": len(differences),
        "method": "paired user cluster percentile bootstrap; exploratory",
    }


def error_analysis(run, output, frame, arrays, predictions):
    output.mkdir(exist_ok=True)
    rows = predictions.sample_row.to_numpy()
    test = frame.iloc[rows].reset_index(drop=True).copy()
    train = frame.iloc[arrays["train"]]
    cutoff = float(train.drop_duplicates("item_id").item_events.quantile(0.25))
    cold = ~test.user_id.isin(set(train.user_id)).to_numpy()
    unseen_item = ~test.item_id.isin(set(train.item_id)).to_numpy()
    tail = (test.item_events <= cutoff).to_numpy()
    segments = {
        "all": np.ones(len(rows), dtype=bool),
        "user_unseen_in_train_sample": cold,
        "user_seen_in_train_sample": ~cold,
        "long_tail_item": tail,
        "non_long_tail_item": ~tail,
        "item_unseen_in_train_sample": unseen_item,
        "item_seen_in_train_sample": ~unseen_item,
        "candidate_item_embedding_OOV": arrays["target_items"][rows] == 1,
    }
    old = pd.read_csv(run / "model_test_comparison.csv").set_index("model")
    frozen = json.loads(
        (output.parent / "stacking/frozen_configuration.json").read_text(
            encoding="utf-8"
        )
    )
    thresholds = {name: float(old.loc[name, "threshold"]) for name in old.index}
    thresholds["Stacking"] = frozen["threshold"]
    results, profiles, cases, ranking = [], [], [], []
    y = predictions.target.to_numpy()
    numeric = [
        "user_events",
        "user_days",
        "user_buys",
        "item_events",
        "item_buys",
        "pair_pv",
        "pair_cart",
        "pair_fav",
        "pair_buys",
        "last_time_days_ago",
    ]
    for name in ["LogisticRegression", "XGBoost", "LightGBM", "DIN", "Stacking"]:
        p = predictions[name].to_numpy()
        label = p >= thresholds[name]
        for segment, mask in segments.items():
            results.append(
                {
                    "model": name,
                    "segment": segment,
                    **subgroup_metrics(y[mask], p[mask], thresholds[name]),
                }
            )
        for error_type, mask in {
            "TP": (y == 1) & label,
            "FP": (y == 0) & label,
            "FN": (y == 1) & ~label,
            "TN": (y == 0) & ~label,
        }.items():
            for feature in numeric:
                profiles.append(
                    {
                        "model": name,
                        "error_type": error_type,
                        "rows": int(mask.sum()),
                        "feature": feature,
                        "median": test.loc[mask, feature].median(),
                        "mean": test.loc[mask, feature].mean(),
                    }
                )
            if error_type in ["FP", "FN"]:
                chosen = np.flatnonzero(mask)
                chosen = chosen[
                    np.argsort(p[chosen])[-5:]
                    if error_type == "FP"
                    else np.argsort(p[chosen])[:5]
                ]
                for i in chosen:
                    cases.append(
                        {
                            "model": name,
                            "error_type": error_type,
                            "sample_row": int(rows[i]),
                            "user_id": test.loc[i, "user_id"],
                            "item_id": test.loc[i, "item_id"],
                            "target": int(y[i]),
                            "probability": float(p[i]),
                            "user_unseen": bool(cold[i]),
                            "long_tail": bool(tail[i]),
                            **{
                                feature: float(test.loc[i, feature])
                                for feature in numeric
                            },
                        }
                    )
        for fraction in [0.01, 0.05, 0.1, 0.2]:
            k = max(1, int(math.ceil(len(y) * fraction)))
            selected = np.argsort(-p, kind="stable")[:k]
            positives = int(y[selected].sum())
            ranking.append(
                {
                    "model": name,
                    "fraction": fraction,
                    "selected_pairs": k,
                    "purchased_pairs": positives,
                    "precision_at_k": positives / k,
                    "recall_at_k": positives / max(y.sum(), 1),
                    "lift_over_random": positives / k / y.mean(),
                }
            )
    pd.DataFrame(results).to_csv(output / "subgroup_metrics.csv", index=False)
    pd.DataFrame(profiles).to_csv(output / "error_feature_profiles.csv", index=False)
    pd.DataFrame(cases).to_parquet(output / "local_error_cases.parquet", index=False)
    pd.DataFrame(ranking).to_csv(output / "ranking_budget_comparison.csv", index=False)
    uncertainty = {
        name: clustered_auc_difference(
            y,
            predictions.Stacking.to_numpy(),
            predictions[name].to_numpy(),
            test.user_id,
        )
        for name in ["XGBoost", "DIN", "LightGBM"]
    }
    details = {
        "long_tail_definition": "item historical events <= 25th percentile over unique training candidate items",
        "long_tail_event_threshold": cutoff,
        "cold_user_definition": "user absent from supervised training sample, not necessarily zero historical events",
        "zero_history_user_rows": int((test.user_events == 0).sum()),
        "cold_user_unique_count": int(test.loc[cold, "user_id"].nunique()),
        "paired_auc_difference_stack_minus_reference": uncertainty,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("分群与误判诊断完成")
    return details


def business_simulation(run, output, frame, arrays, predictions, database):
    output.mkdir(exist_ok=True)
    configuration = json.loads(
        (output.parent / "stacking/frozen_configuration.json").read_text(
            encoding="utf-8"
        )
    )
    validation = pd.read_parquet(
        output.parent / "stacking/validation_predictions.parquet"
    )
    valid_users = (
        pd.DataFrame(
            {
                "user_id": frame.iloc[validation.sample_row].user_id.to_numpy(),
                "score": validation.Stacking.to_numpy(),
            }
        )
        .groupby("user_id")
        .score.max()
    )
    contact_threshold = float(valid_users.quantile(0.8))
    sample = frame.iloc[predictions.sample_row].reset_index(drop=True).copy()
    sample["score"] = predictions.Stacking.to_numpy()
    users = sample.groupby("user_id").agg(
        score=("score", "max"), candidate_pairs=("item_id", "size")
    )
    summary = json.loads(
        (run / "dataset/dataset_summary.json").read_text(encoding="utf-8")
    )
    connection = sqlite3.connect(f"file:{Path(database).as_posix()}?mode=ro", uri=True)
    # 用户级结果用完整标签窗口的购买行为；不把样本对购买率误当用户转化率。
    purchased_users = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT user_id FROM user_behavior WHERE behavior_type=4 AND event_time>=? AND event_time<?",
            (summary["cutoff"], summary["label_end_exclusive"]),
        )
    }
    connection.close()
    users["historical_purchase_proxy"] = users.index.isin(purchased_users).astype(int)
    users["arm"] = [stable_arm(user) for user in users.index]
    users["eligible_for_contact"] = users.score >= contact_threshold
    users["simulated_action"] = (users.arm == "treatment") & users.eligible_for_contact
    users.reset_index().to_parquet(
        output / "local_simulated_user_assignments.parquet", index=False
    )
    suggested = (
        sample.sort_values("score", ascending=False)
        .groupby("user_id", sort=False)
        .head(3)
    )
    suggested[["user_id", "item_id", "score"]].to_parquet(
        output / "local_top3_recommendations.parquet", index=False
    )
    balance = users.groupby("arm").agg(
        users=("score", "size"),
        observed_buyers=("historical_purchase_proxy", "sum"),
        observed_buyer_rate=("historical_purchase_proxy", "mean"),
        mean_score=("score", "mean"),
        contact_candidates=("eligible_for_contact", "sum"),
    )
    balance.to_csv(output / "historical_randomization_balance.csv")
    qualified_balance = (
        users.loc[users.eligible_for_contact]
        .groupby("arm")
        .agg(
            users=("score", "size"),
            observed_buyers=("historical_purchase_proxy", "sum"),
            observed_buyer_rate=("historical_purchase_proxy", "mean"),
        )
    )
    qualified_balance.to_csv(output / "historical_qualified_user_balance.csv")
    p0 = float(users.historical_purchase_proxy.mean())
    target_p0 = float(
        users.loc[users.eligible_for_contact, "historical_purchase_proxy"].mean()
    )
    if not np.isfinite(target_p0):
        target_p0 = p0
    scenarios = revenue_scenarios(target_p0)
    scenarios.to_csv(output / "assumed_business_benefits.csv", index=False)
    power_rows = []
    for lift in [0.005, 0.01, 0.02]:
        if 0 < target_p0 < target_p0 + lift < 1:
            n = sample_size_two_proportions(target_p0, lift)
            power_rows.append(
                {
                    "baseline_qualified_user_purchase_proxy": target_p0,
                    "absolute_lift": lift,
                    "alpha_two_sided": 0.05,
                    "power": 0.8,
                    "required_users_per_arm": n,
                    "required_total_users": 2 * n,
                    "minimum_daily_new_eligible_users_for_14day_enrollment": math.ceil(
                        2 * n / 14
                    ),
                    "current_qualified_cohort_sufficient": int(
                        users.eligible_for_contact.sum()
                    )
                    >= 2 * n,
                }
            )
    pd.DataFrame(power_rows).to_csv(output / "ab_sample_size_planning.csv", index=False)
    target_count = int(users.eligible_for_contact.sum())
    details = {
        "cohort_users": len(users),
        "control_users": int((users.arm == "control").sum()),
        "treatment_users": int((users.arm == "treatment").sum()),
        "historical_user_purchase_proxy": p0,
        "targeted_historical_user_purchase_proxy": target_p0,
        "validation_user_top20_score_cutoff": contact_threshold,
        "test_contact_eligible_users": target_count,
        "test_contact_eligible_rate": target_count / len(users),
        "sample_ratio_mismatch_pvalue": float(
            binomtest(int((users.arm == "treatment").sum()), len(users), 0.5).pvalue
        ),
        "status": "historical simulation only; no intervention performed; no causal uplift estimated",
        "enrollment_days": 14,
        "purchase_followup_days": 7,
        "minimum_primary_readout_day": 21,
        "primary_analysis_population": "all pre-treatment score-qualified users by assigned arm (ITT), regardless of actual opening/clicking",
        "refund_guardrail_followup_days": 30,
        "final_net_revenue_readout_day": 51,
        "scenario_assumptions": {
            "eligible_users": 10000,
            "reach_fraction": 0.2,
            "aov_yuan": 150,
            "gross_margin_rate": 0.25,
            "refund_rate": 0.1,
            "lifts_are_assumed_absolute_probability_changes": [0.005, 0.01, 0.02],
            "one_order_per_converted_user": True,
            "coupon_redemption_rate": 0.3,
        },
    }
    (output / "simulation_summary.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("用户级模拟分组 %d人；历史用户购买代理率 %.4f", len(users), p0)
    return details


def run_analysis(run: Path, output: Path, database: Path):
    if not (output / "stacking/SUCCESS").exists():
        raise ValueError("请先完成Stacking训练")
    from behavior_cleaning.stacking_analysis import source_signature

    expected = json.loads(
        (output / "stacking/source_signature.json").read_text(encoding="utf-8")
    )
    if source_signature(run) != expected:
        raise ValueError("分析源数据或基模型与已冻结的融合模型不一致")
    run_explanations(run, output / "interpretability")
    arrays = dict(np.load(run / "dataset/dataset.npz"))
    frame = pd.read_parquet(run / "dataset/pair_samples.parquet")
    predictions = pd.read_parquet(output / "stacking/test_predictions.parquet")
    diagnostic = error_analysis(run, output / "diagnostics", frame, arrays, predictions)
    business = business_simulation(
        run, output / "business", frame, arrays, predictions, database
    )
    from behavior_cleaning.followup_report import write_followup_reports

    write_followup_reports(run, output, diagnostic, business)
    (output / "SUCCESS").write_text("ok\n", encoding="utf-8")
