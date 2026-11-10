"""逻辑回归系数、XGBoost TreeSHAP及DIN六组输入的穷举Shapley解释。"""

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from behavior_cleaning.model_experiments import sequence_batch
from behavior_cleaning.stacking_analysis import load_din

GROUP_NAMES = [
    "User statistics",
    "Item statistics",
    "Pair interactions",
    "Category encoding",
    "Candidate embedding",
    "History sequence",
]


def exact_group_shap(coalitions, groups):
    """给定全部2^M个联盟的期望输出，按Shapley公式穷举求值。"""
    values = np.asarray(coalitions, dtype=float)
    if values.shape[-1] != 2**groups:
        raise ValueError("联盟数必须为2的特征组数次方")
    result = np.zeros(values.shape[:-1] + (groups,), dtype=float)
    for group in range(groups):
        for subset in range(2**groups):
            if subset & (1 << group):
                continue
            size = subset.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(groups - size - 1)
                / math.factorial(groups)
            )
            result[..., group] += weight * (
                values[..., subset | (1 << group)] - values[..., subset]
            )
    return result


def din_group_values(model, arrays, indices, background, feature_names):
    """背景来自训练集，枚举六组共64个联盟；返回logit空间的贡献。"""
    numeric_groups = [
        [i for i, name in enumerate(feature_names) if name.startswith("user_")],
        [i for i, name in enumerate(feature_names) if name.startswith("item_")],
        [
            i
            for i, name in enumerate(feature_names)
            if not name.startswith(("user_", "item_"))
            and name != "category_target_encoding"
        ],
        [feature_names.index("category_target_encoding")],
    ]
    assert sorted(sum(numeric_groups, [])) == list(range(len(feature_names)))
    reference = sequence_batch(arrays, background)[:-1]
    coalition_values, bg_first, bg_second = [], [], []
    model.eval()
    with torch.no_grad():
        for position, row in enumerate(indices):
            sample = sequence_batch(arrays, np.array([row]))[:-1]
            batch_inputs = [[] for _ in reference]
            for mask in range(64):
                hybrid = [value.clone() for value in reference]
                for group, columns in enumerate(numeric_groups):
                    if mask & (1 << group):
                        hybrid[0][:, columns] = sample[0][:, columns]
                if mask & 16:
                    hybrid[1] = sample[1].expand_as(hybrid[1])
                    hybrid[2] = sample[2].expand_as(hybrid[2])
                if mask & 32:
                    for field in range(3, 7):
                        hybrid[field] = sample[field].expand_as(hybrid[field])
                for field, value in enumerate(hybrid):
                    batch_inputs[field].append(value)
            combined = [torch.cat(values) for values in batch_inputs]
            logits = model(*combined).numpy().reshape(64, len(background))
            coalition_values.append(logits.mean(axis=1))
            half = len(background) // 2
            bg_first.append(logits[:, :half].mean(axis=1))
            bg_second.append(logits[:, half:].mean(axis=1))
            if (position + 1) % 32 == 0:
                logging.info("DIN分组SHAP %d/%d", position + 1, len(indices))
    values = np.array(coalition_values)
    shap_values = exact_group_shap(values, 6)
    error = float(np.max(np.abs(shap_values.sum(1) + values[:, 0] - values[:, -1])))
    assert error < 2e-5
    return (
        shap_values,
        values[:, 0],
        values[:, -1],
        {
            "additivity_max_error": error,
            "first_background_half_mean_abs": np.abs(exact_group_shap(bg_first, 6))
            .mean(0)
            .tolist(),
            "second_background_half_mean_abs": np.abs(exact_group_shap(bg_second, 6))
            .mean(0)
            .tolist(),
        },
    )


def run_explanations(run: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").exists():
        logging.info("复用已完成的解释产物")
        return
    arrays = dict(np.load(run / "dataset/dataset.npz"))
    summary = json.loads(
        (run / "dataset/dataset_summary.json").read_text(encoding="utf-8")
    )
    names = summary["features"]
    lr = joblib.load(run / "models/LogisticRegression.joblib")
    weights = lr.coef_[0]
    coefficients = pd.DataFrame(
        {
            "feature": names,
            "coefficient_per_standard_deviation": weights,
            "odds_multiplier": np.exp(weights),
            "absolute_coefficient": np.abs(weights),
        }
    )
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(
        output / "logistic_coefficients.csv", index=False
    )
    test = arrays["test"]
    model = joblib.load(run / "models/XGBoost.joblib")
    data = xgb.DMatrix(arrays["x"][test])
    shap_values = model.get_booster().predict(
        data, pred_contribs=True, approx_contribs=False
    )
    margins = model.get_booster().predict(data, output_margin=True)
    additivity_error = float(np.max(np.abs(shap_values.sum(1) - margins)))
    assert additivity_error < 1e-4
    pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(shap_values[:, :-1]).mean(0),
            "mean_signed_shap": shap_values[:, :-1].mean(0),
        }
    ).sort_values("mean_abs_shap", ascending=False).to_csv(
        output / "xgboost_shap_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "sample_row": test,
            "expected_margin": shap_values[:, -1],
            **dict(zip(names, shap_values[:, :-1].T)),
        }
    ).to_parquet(output / "xgboost_shap_values.parquet", index=False)
    logging.info("XGBoost精确TreeSHAP完成，加和误差 %.8f", additivity_error)
    rng = np.random.RandomState(20261102)
    explained = rng.choice(test, min(128, len(test)), replace=False)
    background = rng.choice(arrays["train"], 8, replace=False)
    group_values, baseline, margin, checks = din_group_values(
        load_din(run), arrays, explained, background, names
    )
    pd.DataFrame(
        {
            "sample_row": explained,
            "expected_margin": baseline,
            "predicted_margin": margin,
            **dict(zip(GROUP_NAMES, group_values.T)),
        }
    ).to_parquet(output / "din_group_shap_values.parquet", index=False)
    pd.DataFrame(
        {
            "group": GROUP_NAMES,
            "mean_abs_shap": np.abs(group_values).mean(0),
            "mean_signed_shap": group_values.mean(0),
            "background_first_half": checks["first_background_half_mean_abs"],
            "background_second_half": checks["second_background_half_mean_abs"],
        }
    ).sort_values("mean_abs_shap", ascending=False).to_csv(
        output / "din_group_shap_importance.csv", index=False
    )
    details = {
        "xgboost_additivity_max_error": additivity_error,
        "DIN": checks,
        "DIN_group_names": GROUP_NAMES,
        "DIN_background_rows": background.tolist(),
        "DIN_explained_rows": explained.tolist(),
        "DIN_global_sample_policy": "uniform test sample, independent of labels",
        "DIN_method": "exact six-group Shapley enumeration, 64 coalitions; empirical training background of eight rows; model logit output",
        "limitations": "background-dependent interventional input attributions; hybrid blocks may violate observed joint distributions; not causal; group attributions not individual embedding coordinates",
    }
    (output / "explanation_checks.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "SUCCESS").write_text("ok\n", encoding="utf-8")
