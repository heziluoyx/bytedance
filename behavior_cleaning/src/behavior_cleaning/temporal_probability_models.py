"""时间隔离的多模型训练、连续购买概率校准及留出式Stacking。"""
import gc
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from behavior_cleaning.model_experiments import deep_predict, fit_deep
from behavior_cleaning.promotion_robustness import BASELINES, resample_training
from behavior_cleaning.traditional_models import (
    choose_threshold,
    classifier,
    metrics,
    suggest,
)

BASE_NAMES = [
    "LogisticRegression",
    "XGBoost",
    "LightGBM",
    "LSTM",
    "GRU",
    "DIN",
    "DIN_MHA",
]


def logit(probability):
    p = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, bins=10):
    y = np.asarray(y)
    p = np.asarray(p)
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(bucket == i) * abs(y[bucket == i].mean() - p[bucket == i].mean())
            for i in range(bins)
            if np.any(bucket == i)
        )
    )


def probability_metrics(y, p, threshold=0.5):
    return {
        **metrics(y, np.asarray(p), threshold),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "ece_10_equal_width": expected_calibration_error(y, p),
        "mean_probability": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def fit_calibrator(y, probability):
    """校准集保持自然正例率；不使用测试标签。"""
    model = LogisticRegression(C=1.0, max_iter=600, random_state=42)
    model.fit(logit(probability).reshape(-1, 1), y)
    return model


def apply_calibrator(model, probability):
    return model.predict_proba(logit(probability).reshape(-1, 1))[:, 1]


def train_traditional(arrays, output, trials):
    output = Path(output)
    directory = output / "models"
    directory.mkdir(exist_ok=True)
    selection_path = directory / "traditional_selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    x, y = arrays["x"], arrays["y"]
    train, valid = arrays["train"], arrays["valid"]
    ratio = float((y[train] == 0).sum() / y[train].sum())
    selection = {}
    rows = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name, base in BASELINES.items():
        best = -np.inf
        selected = None
        for strategy in ["none", "smote", "undersample", "class_weight"]:
            xx, yy = resample_training(x[train], y[train], strategy)
            fitted = classifier(name, base, strategy, ratio).fit(xx, yy)
            p = fitted.predict_proba(x[valid])[:, 1]
            row = {
                "model": name,
                "strategy": strategy,
                "training_rows": len(yy),
                **probability_metrics(y[valid], p),
            }
            rows.append(row)
            logging.info("%s %s 验证AUC %.5f", name, strategy, row["auc"])
            if row["auc"] > best:
                best = row["auc"]
                selected = {
                    "strategy": strategy,
                    "params": base.copy(),
                    "validation_auc": best,
                }
                winner = fitted
            del xx, yy
            gc.collect()
        pd.DataFrame(rows).to_csv(
            output / "imbalance_validation_comparison.csv", index=False
        )
        xx, yy = resample_training(x[train], y[train], selected["strategy"])

        def objective(trial):
            nonlocal winner, selected
            parameters = suggest(trial, name)
            fitted = classifier(name, parameters, selected["strategy"], ratio).fit(
                xx, yy
            )
            auc = float(roc_auc_score(y[valid], fitted.predict_proba(x[valid])[:, 1]))
            if auc > selected["validation_auc"]:
                winner = fitted
                selected.update(params=parameters, validation_auc=auc)
            return auc

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.enqueue_trial(base)
        study.optimize(objective, n_trials=trials)
        study.trials_dataframe().to_csv(
            output / (name + "_optuna_trials.csv"), index=False
        )
        joblib.dump(winner, directory / (name + ".joblib"))
        selection[name] = selected
        logging.info("完成传统模型 %s %s", name, selected)
        del xx, yy, winner
        gc.collect()
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def train_sequence_models(arrays, summary, output, epochs=8, configurations=2):
    output = Path(output)
    directory = output / "models"
    directory.mkdir(exist_ok=True)
    selection_path = directory / "sequence_selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    configs = [
        {
            "embedding": 8,
            "hidden": 16,
            "dropout": 0.15,
            "heads": 2,
            "learning_rate": 0.002,
        },
        {
            "embedding": 16,
            "hidden": 32,
            "dropout": 0.3,
            "heads": 2,
            "learning_rate": 0.001,
        },
    ]
    selected = {}
    for kind in BASE_NAMES[3:]:
        best = -np.inf
        for i, parameters in enumerate(configs[:configurations]):
            model, auc = fit_deep(kind, parameters, arrays, summary, epochs, output, i)
            if auc > best:
                best = auc
                torch.save(model.state_dict(), directory / (kind + ".pt"))
                selected[kind] = {
                    "params": parameters,
                    "validation_auc": auc,
                    "maximum_epochs": epochs,
                    "configuration": i,
                    "strategy": "class_weight",
                }
            del model
            gc.collect()
        logging.info("完成序列模型 %s %s", kind, selected[kind])
    selection_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selected


def load_sequence(kind, info, arrays, summary, output):
    from behavior_cleaning.sequence_models import PurchaseSequenceModel

    params = {k: v for k, v in info["params"].items() if k != "learning_rate"}
    model = PurchaseSequenceModel(
        kind,
        summary["item_vocabulary"],
        summary["category_vocabulary"],
        arrays["x"].shape[1],
        **params,
    )
    model.load_state_dict(
        torch.load(Path(output) / "models" / (kind + ".pt"), map_location="cpu")
    )
    model.eval()
    return model


def frozen_prediction_and_calibration(arrays, summary, output, traditional, sequence):
    output = Path(output)
    models = output / "models"
    raw = {}
    all_indices = np.arange(len(arrays["y"]))
    for name in BASE_NAMES:
        if name in traditional:
            raw[name] = joblib.load(models / (name + ".joblib")).predict_proba(
                arrays["x"]
            )[:, 1]
        else:
            model = load_sequence(name, sequence[name], arrays, summary, output)
            raw[name] = deep_predict(model, arrays, all_indices)
            del model
            gc.collect()
    meta, cal, test = arrays["meta"], arrays["calibration"], arrays["test"]
    stack_columns = ["LogisticRegression", "XGBoost", "DIN"]
    logits = np.column_stack([logit(raw[n]) for n in stack_columns])
    scaler = StandardScaler().fit(logits[meta])
    stacking = LogisticRegression(C=0.3, max_iter=600, random_state=42).fit(
        scaler.transform(logits[meta]), arrays["y"][meta]
    )
    joblib.dump(
        {"scaler": scaler, "model": stacking, "columns": stack_columns},
        models / "Stacking.joblib",
    )
    raw["Stacking"] = stacking.predict_proba(scaler.transform(logits))[:, 1]
    calibrated = {}
    calibrators = {}
    thresholds = {}
    for name, p in raw.items():
        calibrators[name] = fit_calibrator(arrays["y"][cal], p[cal])
        calibrated[name] = apply_calibrator(calibrators[name], p)
        thresholds[name] = choose_threshold(arrays["y"][cal], calibrated[name][cal])
    joblib.dump(calibrators, models / "probability_calibrators.joblib")
    (models / "frozen_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    # 只有至此模型、融合、概率校准及辅助分类阈值均冻结后才评估测试标签。
    rows = []
    labels = arrays["y"]
    prediction = {"sample_row": all_indices, "target": labels}
    for name in raw:
        prediction[name + "_raw"] = raw[name]
        prediction[name + "_probability"] = calibrated[name]
        for partition, ix in [
            ("validation", arrays["valid"]),
            ("calibration_in_sample", cal),
            ("test", test),
        ]:
            for state, p in [("raw", raw[name]), ("calibrated", calibrated[name])]:
                if partition == "validation" and (
                    state == "calibrated" or name == "Stacking"
                ):
                    continue  # 后续阶段的模型不能冒充更早时点可用的验证预测。
                rows.append(
                    {
                        "partition": partition,
                        "model": name,
                        "probability_state": state,
                        "rows": len(ix),
                        "positives": int(labels[ix].sum()),
                        **probability_metrics(
                            labels[ix],
                            p[ix],
                            thresholds[name] if state == "calibrated" else 0.5,
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "probability_model_comparison.csv", index=False)
    predictions = pd.DataFrame(prediction)
    identifiers = pd.read_parquet(
        output / "dataset/samples.parquet",
        columns=["user_id", "item_id", "cutoff", "split"],
    )
    predictions = pd.concat([identifiers.reset_index(drop=True), predictions], axis=1)
    predictions.to_parquet(output / "all_window_predictions.parquet", index=False)
    predictions.iloc[test].to_parquet(output / "test_predictions.parquet", index=False)
    predictions.iloc[test][
        ["user_id", "item_id", "cutoff"] + [n + "_probability" for n in raw]
    ].to_csv(output / "test_purchase_probabilities.csv", index=False)
    return raw, calibrated, thresholds
