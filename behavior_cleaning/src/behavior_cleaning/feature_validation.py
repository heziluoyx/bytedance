"""按时间切分的特征预处理、筛选和购买标签预评估。"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from behavior_cleaning.analysis_runner import connect_readonly
from behavior_cleaning.feature_snapshots import implicit_features, labels, snapshot


def encode_target(training: pd.DataFrame, validation: pd.DataFrame) -> tuple:
    """同一历史时点按用户折外编码，验证期仅使用训练标签统计。"""
    category = training["dominant_category"].fillna("__missing__").astype(str)
    target = training["target"].astype(int)
    folds = min(5, int(target.value_counts().min()))
    if folds < 2:
        raise ValueError("训练标签每类至少需要两个用户才能生成折外 Target Encoding")
    encoded = pd.Series(np.nan, index=training.index)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    def mapping(indices: np.ndarray) -> tuple[pd.Series, float]:
        prior = float(target.iloc[indices].mean())
        stats = (
            pd.DataFrame(
                {"category": category.iloc[indices], "target": target.iloc[indices]}
            )
            .groupby("category")["target"]
            .agg(["sum", "count"])
        )
        return (stats["sum"] + 20 * prior) / (stats["count"] + 20), prior

    for fit_indices, held_indices in splitter.split(category, target):
        means, prior = mapping(fit_indices)
        encoded.iloc[held_indices] = (
            category.iloc[held_indices].map(means).fillna(prior)
        )
    means, prior = mapping(np.arange(len(training)))
    held_encoded = (
        validation["dominant_category"]
        .fillna("__missing__")
        .astype(str)
        .map(means)
        .fillna(prior)
    )
    return (
        encoded,
        held_encoded,
        {"mapping": means, "prior": prior, "smoothing": 20, "folds": folds},
    )


def preprocess(training: pd.DataFrame, validation: pd.DataFrame, output: Path) -> dict:
    """仅训练集拟合缺失值、缩放、方差、相关性、互信息及基线模型。"""
    if training["target"].nunique() != 2 or validation["target"].nunique() != 2:
        raise ValueError("训练和验证时点均须同时包含购买与未购买用户")
    train_encoded, valid_encoded, encoding = encode_target(training, validation)
    excluded = {"target", "user_id"}
    columns = [
        name
        for name in training.select_dtypes(include="number").columns
        if name not in excluded
    ]
    train = training[columns].copy()
    held = validation[columns].copy()
    train["dominant_category_target_encoded"] = train_encoded
    held["dominant_category_target_encoded"] = valid_encoded
    train["activity_level_code"] = (
        training["activity_level"].map({"low": 0, "medium": 1, "high": 2}).astype(float)
    )
    held["activity_level_code"] = (
        validation["activity_level"]
        .map({"low": 0, "medium": 1, "high": 2})
        .astype(float)
    )
    train = train.replace([np.inf, -np.inf], np.nan)
    held = held.replace([np.inf, -np.inf], np.nan)
    medians = train.median().fillna(0)
    train, held = train.fillna(medians), held.fillna(medians)
    standard = StandardScaler().fit(train)
    minmax = MinMaxScaler().fit(train)
    train_standard = pd.DataFrame(standard.transform(train), columns=train.columns)
    held_standard = pd.DataFrame(standard.transform(held), columns=held.columns)
    train_minmax = pd.DataFrame(minmax.transform(train), columns=train.columns)
    held_minmax = pd.DataFrame(minmax.transform(held), columns=held.columns)
    variance = train_minmax.var(ddof=0)
    variance_kept = variance[variance > 1e-5].index.tolist()
    correlations = train_standard[variance_kept].corr().abs()
    kept: list[str] = []
    redundant: list[str] = []
    for name in variance_kept:
        if kept and (correlations.loc[name, kept] > 0.95).any():
            redundant.append(name)
        else:
            kept.append(name)
    if not kept:
        raise ValueError("特征筛选后没有可用特征")
    y_train = training["target"].astype(int).to_numpy()
    y_held = validation["target"].astype(int).to_numpy()
    mutual_information = pd.Series(
        mutual_info_classif(train_standard[kept], y_train, random_state=42),
        index=kept,
    )
    # 不用验证标签挑特征；最多30个训练互信息为正的非冗余特征。
    selected = mutual_information[mutual_information > 0].nlargest(30).index.tolist()
    if not selected:
        selected = mutual_information.nlargest(1).index.tolist()
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(train_standard[selected], y_train)
    probability = model.predict_proba(held_standard[selected])[:, 1]
    importance = pd.DataFrame(index=train.columns)
    importance["train_variance_after_minmax"] = variance
    importance["train_target_pearson"] = train.corrwith(
        pd.Series(y_train, index=train.index)
    )
    importance["train_mutual_information"] = mutual_information
    importance["drop_reason"] = [
        "low_variance"
        if name not in variance_kept
        else "correlation_above_0.95"
        if name in redundant
        else "not_top30_positive_mi"
        if name not in selected
        else "selected"
        for name in importance.index
    ]
    importance["standardized_logistic_coefficient"] = pd.Series(
        model.coef_[0], index=selected
    )
    importance.index.name = "feature"
    importance.sort_values("train_mutual_information", ascending=False).to_csv(
        output / "feature_importance_preassessment.csv", encoding="utf-8-sig"
    )
    for split, source, scaled, normalized in [
        ("train", training, train_standard, train_minmax),
        ("validation", validation, held_standard, held_minmax),
    ]:
        for kind, features in [("standardized", scaled), ("minmax", normalized)]:
            frame = features.copy()
            frame.insert(0, "user_id", source["user_id"].to_numpy())
            frame["target"] = source["target"].to_numpy()
            frame.to_parquet(output / (split + "_" + kind + ".parquet"), index=False)
    pd.DataFrame(
        {
            "user_id": validation["user_id"],
            "target": y_held,
            "predicted_purchase_probability": probability,
        }
    ).to_csv(output / "validation_predictions.csv", index=False)
    joblib.dump(
        {
            "model": model,
            "selected": selected,
            "columns": train.columns.tolist(),
            "medians": medians,
            "standard_scaler": standard,
            "minmax_scaler": minmax,
            "target_encoding": encoding,
        },
        output / "preassessment_artifacts.joblib",
    )
    return {
        "input_features": len(train.columns),
        "selected_features": selected,
        "train_users": len(training),
        "validation_users": len(validation),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_held.mean()),
        "validation_roc_auc": float(roc_auc_score(y_held, probability)),
        "validation_average_precision": float(
            average_precision_score(y_held, probability)
        ),
        "validation_brier": float(brier_score_loss(y_held, probability)),
        "scope": "single_temporal_holdout_seen_users_exploratory_not_final_model",
    }


def run_validation(database: Path, output: Path, max_items: int = 1000) -> dict:
    """最后两段完整7天作为训练和验证标签窗，历史窗口固定为截止前全部记录。"""
    with connect_readonly(database) as connection:
        finish = pd.Timestamp(
            connection.execute("SELECT MAX(event_time) FROM dim_time").fetchone()[0]
        ) + pd.Timedelta(hours=1)
        start = pd.Timestamp(
            connection.execute("SELECT MIN(event_time) FROM dim_time").fetchone()[0]
        )
    validation_cutoff = finish - pd.Timedelta(days=7)
    training_cutoff = validation_cutoff - pd.Timedelta(days=7)
    if training_cutoff - start < pd.Timedelta(days=7):
        raise ValueError("至少需要21天连续观察范围才能进行本例时间切分")
    snapshots = []
    for cutoff in [training_cutoff, validation_cutoff]:
        cutoff_text = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        logging.info("构建预测快照：%s", cutoff_text)
        frame = snapshot(database, cutoff_text).merge(
            labels(database, cutoff_text),
            on="user_id",
            how="left",
            validate="one_to_one",
        )
        frame["target"] = frame["target"].fillna(0).astype(int)
        snapshots.append(frame)
    # 只评估训练历史中已见用户；新增用户不默认映射为零向量。
    common = set(snapshots[0]["user_id"]) & set(snapshots[1]["user_id"])
    omitted = [len(frame) - len(common) for frame in snapshots]
    snapshots = [
        frame[frame["user_id"].isin(common)]
        .sort_values("user_id")
        .reset_index(drop=True)
        for frame in snapshots
    ]
    snapshots, implicit = implicit_features(database, snapshots, output, max_items)
    for split, frame in zip(["train", "validation"], snapshots):
        frame.to_parquet(output / (split + "_raw_features.parquet"), index=False)
    metrics = preprocess(snapshots[0], snapshots[1], output)
    return {
        "training_cutoff": str(training_cutoff),
        "validation_cutoff": str(validation_cutoff),
        "data_finish_exclusive": str(finish),
        "excluded_users_without_both_histories": omitted,
        "implicit_features": implicit,
        "preassessment": metrics,
        "limitations": [
            "RFM金额维度缺失，仅RF代理；全量商品隐式特征未完成。",
            "隐式特征和关联规则仅训练热门商品子集，阈值5名共现用户。",
            "共现基于用户是否交互，规则为集合关联而非时间因果。",
            "SVD、编码、缩放及筛选只拟合训练历史；验证标签只用于评估。",
            "同小时行为序列不可识别，仅统计相邻单一类型小时桶转移。",
            "历史长度随快照增加，不同截点的累计特征受观察窗长度影响。",
            "仅一次时间验证且用户重叠，不能用于宣称新用户泛化能力。",
            "清洗异常阈值曾由全期数据拟合，预测评估仍有预处理回看局限。",
        ],
    }
