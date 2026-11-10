"""传统分类模型、验证阈值和Optuna搜索空间。"""
import numpy as np
import optuna
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
from xgboost import XGBClassifier


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
