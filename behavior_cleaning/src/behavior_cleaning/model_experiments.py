"""训练内不平衡处理、Optuna调优及冻结测试集的模型比较。"""

import copy
import gc
import json
import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from xgboost import XGBClassifier

from behavior_cleaning.sequence_models import PurchaseSequenceModel


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    """统一指标，阈值由验证集确定后固定用于测试集。"""
    predicted = probability >= threshold
    return {
        "auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """在验证集上最大化F1，不查看测试标签。"""
    precision, recall, threshold = precision_recall_curve(y, probability)
    scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(threshold[np.argmax(scores)])


def classifier(name, params, strategy, ratio):
    """为三类模型统一配置训练权重和受限CPU线程。"""
    if name == "LogisticRegression":
        return LogisticRegression(
            max_iter=600,
            random_state=42,
            class_weight="balanced" if strategy == "class_weight" else None,
            **params,
        )
    if name == "XGBoost":
        return XGBClassifier(
            tree_method="hist",
            eval_metric="auc",
            objective="binary:logistic",
            random_state=42,
            n_jobs=2,
            scale_pos_weight=ratio if strategy == "class_weight" else 1,
            **params,
        )
    return LGBMClassifier(
        random_state=42,
        n_jobs=2,
        verbosity=-1,
        scale_pos_weight=ratio if strategy == "class_weight" else 1,
        **params,
    )


def suggest(trial: optuna.Trial, name: str) -> dict:
    """在有限预算下探索正则、树深和学习率等关键参数。"""
    if name == "LogisticRegression":
        return {"C": trial.suggest_float("C", 0.001, 30, log=True)}
    result = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 240, step=40),
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.15, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
    }
    if name == "XGBoost":
        result.update(
            max_depth=trial.suggest_int("max_depth", 2, 6),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 12),
            subsample=trial.suggest_float("subsample", 0.7, 1),
        )
    else:
        result.update(
            num_leaves=trial.suggest_categorical("num_leaves", [15, 31, 63]),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 100),
        )
    return result


def seed_everything(seed: int = 42) -> None:
    """固定CPU训练随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sequence_batch(arrays: dict, index: np.ndarray) -> list[torch.Tensor]:
    """向量化取批次，避免CPU逐样本拼接历史张量。"""
    users = arrays["row_users"][index]
    return [
        torch.from_numpy(value)
        for value in [
            arrays["x"][index],
            arrays["target_items"][index],
            arrays["target_categories"][index],
            arrays["sequence_items"][users],
            arrays["sequence_categories"][users],
            np.concatenate(
                [arrays["sequence_aux"][users], arrays["matches"][index, :, None]],
                axis=2,
            ),
            arrays["sequence_mask"][users],
            arrays["y"][index].astype(np.float32),
        ]
    ]


def deep_predict(model: nn.Module, arrays: dict, indices: np.ndarray) -> np.ndarray:
    """无梯度批量推断，返回与传入索引同序的概率。"""
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(indices), 2048):
            batch = sequence_batch(arrays, indices[start : start + 2048])
            result.append(torch.sigmoid(model(*batch[:-1])).cpu().numpy())
    return np.concatenate(result)


def fit_deep(kind, params, arrays, summary, epochs, output, trial_number):
    """验证AUC早停和学习率衰减；保留最佳epoch而非最后epoch。"""
    seed_everything(42 + trial_number)
    model = PurchaseSequenceModel(
        kind,
        summary["item_vocabulary"],
        summary["category_vocabulary"],
        arrays["x"].shape[1],
        **{k: v for k, v in params.items() if k != "learning_rate"},
    )
    y_train = arrays["y"][arrays["train"]]
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5
    )
    rng = np.random.RandomState(42 + trial_number)
    best, best_state, stale, history = -1, None, 0, []
    for epoch in range(epochs):
        model.train()
        losses, seen = 0, 0
        shuffled = rng.permutation(arrays["train"])
        for start in range(0, len(shuffled), 1024):
            batch = sequence_batch(arrays, shuffled[start : start + 1024])
            optimizer.zero_grad()
            loss = loss_function(model(*batch[:-1]), batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses += float(loss.detach()) * len(batch[-1])
            seen += len(batch[-1])
        probability = deep_predict(model, arrays, arrays["valid"])
        auc = roc_auc_score(arrays["y"][arrays["valid"]], probability)
        scheduler.step(auc)
        history.append(
            {
                "epoch": epoch + 1,
                "train_weighted_loss": losses / seen,
                "validation_auc": float(auc),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        logging.info(
            "%s trial%d epoch%d loss%.4f valAUC%.5f",
            kind,
            trial_number,
            epoch + 1,
            losses / seen,
            auc,
        )
        if auc > best + 1e-5:
            best, best_state, stale = auc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(
        output / f"{kind}_trial{trial_number}_history.csv", index=False
    )
    return model, float(best)


def run_experiments(
    run: Path,
    trials: int = 8,
    deep_trials: int = 2,
    epochs: int = 10,
    lstm_epochs: int | None = None,
) -> None:
    """完成三种传统模型及四种深度架构比较，测试集最后统一评估。"""
    run = Path(run)
    if (run / "SUCCESS").exists():
        raise FileExistsError("本次实验已完成，请使用新运行目录进行新实验")
    models = run / "models"
    models.mkdir(exist_ok=True)
    summary = json.loads(
        (run / "dataset/dataset_summary.json").read_text(encoding="utf-8")
    )
    with np.load(run / "dataset/dataset.npz") as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    x, y, train, valid, test = [arrays[k] for k in ["x", "y", "train", "valid", "test"]]
    torch.set_num_threads(2)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ratio = float((y[train] == 0).sum() / (y[train] == 1).sum())
    baseline_params = {
        "LogisticRegression": {"C": 1.0},
        "XGBoost": {
            "n_estimators": 160,
            "max_depth": 4,
            "learning_rate": 0.08,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "subsample": 0.9,
        },
        "LightGBM": {
            "n_estimators": 160,
            "num_leaves": 31,
            "learning_rate": 0.08,
            "reg_lambda": 1.0,
            "min_child_samples": 40,
        },
    }
    baseline_file = run / "imbalance_validation_comparison.csv"
    comparison = (
        pd.read_csv(baseline_file).to_dict("records") if baseline_file.exists() else []
    )
    selection_file = run / "traditional_selection.json"
    selection_saved = (
        json.loads(selection_file.read_text(encoding="utf-8"))
        if selection_file.exists()
        else {}
    )
    selected = {}
    storage = "sqlite:///" + str((run / "optuna.sqlite").absolute()).replace("\\", "/")
    logging.info("比较原始分布、SMOTE、欠采样及类别权重；验证集不重采样")
    for name in baseline_params:
        if name in selection_saved and (models / (name + ".joblib")).exists():
            selected[name] = {
                **selection_saved[name],
                "model": joblib.load(models / (name + ".joblib")),
            }
            logging.info("复用已完成的%s调优结果", name)
            continue
        comparison = [row for row in comparison if row["model"] != name]
        best = -1
        for strategy in ["none", "smote", "undersample", "class_weight"]:
            started = time.monotonic()
            fit_x, fit_y = x[train], y[train]
            if strategy == "smote":
                fit_x, fit_y = SMOTE(
                    random_state=42, k_neighbors=min(5, int(fit_y.sum()) - 1)
                ).fit_resample(fit_x, fit_y)
            elif strategy == "undersample":
                fit_x, fit_y = RandomUnderSampler(random_state=42).fit_resample(
                    fit_x, fit_y
                )
            model = classifier(name, baseline_params[name], strategy, ratio).fit(
                fit_x, fit_y
            )
            p = model.predict_proba(x[valid])[:, 1]
            row = {
                "model": name,
                "strategy": strategy,
                "fit_rows": len(fit_y),
                "fit_positives": int(fit_y.sum()),
                "seconds": time.monotonic() - started,
                **metrics(y[valid], p),
            }
            comparison.append(row)
            logging.info(
                "%s %s valAUC%.5f AP%.5f",
                name,
                strategy,
                row["auc"],
                row["average_precision"],
            )
            pd.DataFrame(comparison).to_csv(
                run / "imbalance_validation_comparison.csv", index=False
            )
            if row["auc"] > best:
                best = row["auc"]
                selected[name] = {
                    "model": model,
                    "strategy": strategy,
                    "params": baseline_params[name],
                    "validation_auc": best,
                }
            del fit_x, fit_y
            gc.collect()
        strategy = selected[name]["strategy"]
        fit_x, fit_y = x[train], y[train]
        if strategy == "smote":
            fit_x, fit_y = SMOTE(random_state=42).fit_resample(fit_x, fit_y)
        elif strategy == "undersample":
            fit_x, fit_y = RandomUnderSampler(random_state=42).fit_resample(
                fit_x, fit_y
            )
        study = optuna.create_study(
            study_name=name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.enqueue_trial(baseline_params[name])

        def objective(trial):
            params = suggest(trial, name)
            fitted = classifier(name, params, strategy, ratio).fit(fit_x, fit_y)
            auc = float(roc_auc_score(y[valid], fitted.predict_proba(x[valid])[:, 1]))
            if auc >= selected[name]["validation_auc"]:
                selected[name].update(model=fitted, params=params, validation_auc=auc)
            logging.info("Optuna %s trial%d valAUC%.5f", name, trial.number, auc)
            return auc

        study.optimize(objective, n_trials=trials)
        study.trials_dataframe().to_csv(run / f"{name}_optuna_trials.csv", index=False)
        joblib.dump(selected[name]["model"], models / (name + ".joblib"))
        selection_saved[name] = {
            k: v for k, v in selected[name].items() if k != "model"
        }
        selection_file.write_text(
            json.dumps(selection_saved, indent=2), encoding="utf-8"
        )
        del fit_x, fit_y
        gc.collect()
    logging.info("开始CPU序列模型训练；深度模型只使用真实序列与类别权重")
    deep_file = run / "deep_selection.json"
    deep_selected = (
        json.loads(deep_file.read_text(encoding="utf-8")) if deep_file.exists() else {}
    )
    for kind in ["LSTM", "GRU", "DIN", "DIN_MHA"]:
        if kind in deep_selected and (models / (kind + ".pt")).exists():
            logging.info("复用已完成的%s调优结果", kind)
            continue
        epoch_limit = (
            lstm_epochs if kind == "LSTM" and lstm_epochs is not None else epochs
        )
        study = optuna.create_study(
            study_name=kind,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        for unfinished in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(unfinished.number, state=optuna.trial.TrialState.FAIL)
        if kind != "LSTM":
            # 小预算下确保实际比较不同嵌入和隐藏层，而非只随机改变学习率。
            study.enqueue_trial(
                {
                    "embedding": 8,
                    "hidden": 64,
                    "dropout": 0.3,
                    "learning_rate": 0.0015,
                    "heads": 4,
                }
            )
        study.enqueue_trial(
            {
                "embedding": 16,
                "hidden": 32,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "heads": 2,
            }
        )
        deep_selected[kind] = {"validation_auc": -1}

        def deep_objective(trial):
            params = {
                "embedding": trial.suggest_categorical("embedding", [8, 16]),
                "hidden": trial.suggest_categorical("hidden", [32, 64]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.4),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.0005, 0.003, log=True
                ),
                "heads": trial.suggest_categorical("heads", [2, 4])
                if kind == "DIN_MHA"
                else 2,
            }
            fitted, auc = fit_deep(
                kind, params, arrays, summary, epoch_limit, run, trial.number
            )
            if auc > deep_selected[kind]["validation_auc"]:
                deep_selected[kind].update(
                    params=params,
                    validation_auc=auc,
                    best_trial=trial.number,
                    epoch_limit=epoch_limit,
                )
                torch.save(
                    {
                        "state_dict": fitted.state_dict(),
                        "params": params,
                        "kind": kind,
                        "item_count": summary["item_vocabulary"],
                        "category_count": summary["category_vocabulary"],
                        "numeric_count": x.shape[1],
                    },
                    models / (kind + ".pt"),
                )
            del fitted
            gc.collect()
            return auc

        study.optimize(deep_objective, n_trials=deep_trials)
        study.trials_dataframe().to_csv(run / f"{kind}_optuna_trials.csv", index=False)
        deep_file.write_text(json.dumps(deep_selected, indent=2), encoding="utf-8")
    # 所有策略和超参冻结后才读取测试预测，禁止据测试表现重调。
    results, predictions = [], {}
    for name, info in selected.items():
        p_valid = info["model"].predict_proba(x[valid])[:, 1]
        threshold = choose_threshold(y[valid], p_valid)
        p_test = info["model"].predict_proba(x[test])[:, 1]
        results.append(
            {
                "model": name,
                "strategy": info["strategy"],
                "validation_auc": info["validation_auc"],
                **metrics(y[test], p_test, threshold),
                "accuracy_at_0_5": float(accuracy_score(y[test], p_test >= 0.5)),
            }
        )
        predictions[name] = p_test
        info.pop("model")
    for kind, info in deep_selected.items():
        checkpoint = torch.load(models / (kind + ".pt"), map_location="cpu")
        model = PurchaseSequenceModel(
            kind,
            checkpoint["item_count"],
            checkpoint["category_count"],
            checkpoint["numeric_count"],
            **{k: v for k, v in checkpoint["params"].items() if k != "learning_rate"},
        )
        # 学习率属于训练配置，构造网络时移除。
        model.load_state_dict(checkpoint["state_dict"])
        p_valid = deep_predict(model, arrays, valid)
        p_test = deep_predict(model, arrays, test)
        threshold = choose_threshold(y[valid], p_valid)
        results.append(
            {
                "model": kind,
                "strategy": "class_weight",
                "validation_auc": info["validation_auc"],
                **metrics(y[test], p_test, threshold),
                "accuracy_at_0_5": float(accuracy_score(y[test], p_test >= 0.5)),
            }
        )
        predictions[kind] = p_test
    pd.DataFrame(results).to_csv(run / "model_test_comparison.csv", index=False)
    prediction_frame = pd.DataFrame(
        {"sample_row": test, "target": y[test], **predictions}
    )
    prediction_frame.to_parquet(run / "test_predictions.parquet", index=False)
    metadata = {
        "traditional": selected,
        "deep": deep_selected,
        "trials_per_traditional_model": trials,
        "trials_per_deep_model": deep_trials,
        "maximum_epochs": max(
            info.get("epoch_limit", epochs) for info in deep_selected.values()
        ),
        "deep_epoch_limits": {
            name: info.get("epoch_limit", epochs)
            for name, info in deep_selected.items()
        },
        "device": "cpu",
        "test_threshold_rule": "maximize validation F1, frozen before test",
        "validation_selection_winner": max(results, key=lambda r: r["validation_auc"])[
            "model"
        ],
    }
    (run / "experiment_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from behavior_cleaning.model_report import write_report

    write_report(run, summary, pd.DataFrame(results), predictions, y[test], metadata)
    (run / "SUCCESS").write_text("ok\n", encoding="utf-8")
    logging.info("完成七个模型训练及冻结测试集评估：%s", results)
