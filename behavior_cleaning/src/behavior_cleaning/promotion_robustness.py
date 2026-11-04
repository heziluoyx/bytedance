"""在较早截点调参，冻结后评估覆盖双十二的跨期样本。"""

import gc
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from behavior_cleaning.analysis_runner import connect_readonly
from behavior_cleaning.traditional_dataset import encode_category, split_indices
from behavior_cleaning.traditional_models import (
    choose_threshold,
    classifier,
    metrics,
    suggest,
)

EARLY = "2025-11-25 00:00:00"
LATE = "2025-12-12 00:00:00"
NUMERIC = [
    "pair_pv",
    "pair_fav",
    "pair_cart",
    "pair_buys",
    "pair_recent_events",
    "pair_days",
    "item_events",
    "item_buys",
    "item_users",
    "user_events",
    "user_items",
    "user_days",
    "user_pv",
    "user_fav",
    "user_cart",
    "user_buys",
    "user_last_days_ago",
    "user_last_buy_days_ago",
    "pair_buy_share",
    "item_buy_share",
    "user_buy_share",
    "pair_user_event_share",
    "first_time_days_ago",
    "last_time_days_ago",
    "last_buy_days_ago",
]
BASELINES = {
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


def history_bounds(cutoff):
    center = pd.Timestamp(cutoff)
    return tuple(
        t.strftime("%Y-%m-%d %H:%M:%S")
        for t in [center - pd.Timedelta(days=7), center, center + pd.Timedelta(days=7)]
    )


def build_snapshot(database, cutoff, destination, max_pairs=120000, seed=42):
    """仅历史候选决定抽样；所有累计特征统一为预测前7天。"""
    destination = Path(destination)
    if destination.exists():
        info = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
        if (
            info["cutoff"] != cutoff
            or info["max_pairs"] != max_pairs
            or info["seed"] != seed
        ):
            raise ValueError("快照配置不符，请使用新目录")
        return pd.read_parquet(destination), info
    start, cutoff, finish = history_bounds(cutoff)
    connection = connect_readonly(Path(database))
    latest = connection.execute("SELECT MAX(event_time) FROM dim_time").fetchone()[0]
    earliest = connection.execute("SELECT MIN(event_time) FROM dim_time").fetchone()[0]
    if pd.Timestamp(earliest) > pd.Timestamp(start) or pd.Timestamp(
        latest
    ) + pd.Timedelta(hours=1) < pd.Timestamp(finish):
        raise ValueError("完整7天历史或7天标签窗口不足")
    query = """SELECT user_id,item_id,
      SUM(event_time<:cutoff AND behavior_type=1),
      SUM(event_time<:cutoff AND behavior_type=2),
      SUM(event_time<:cutoff AND behavior_type=3),
      SUM(event_time<:cutoff AND behavior_type=4),
      SUM(event_time<:cutoff),
      MIN(CASE WHEN event_time<:cutoff THEN event_time END),
      MAX(CASE WHEN event_time<:cutoff THEN event_time END),
      MAX(CASE WHEN event_time<:cutoff AND behavior_type=4 THEN event_time END),
      COUNT(DISTINCT CASE WHEN event_time<:cutoff THEN date(event_time) END),
      MAX(event_time>=:cutoff AND behavior_type=4)
      FROM user_behavior INDEXED BY idx_behavior_quadruple
      WHERE event_time>=:start AND event_time<:finish GROUP BY user_id,item_id"""
    rng = np.random.RandomState(seed)
    records, item_stats = [], {}
    candidates = positives = all_future = 0
    logging.info("构建固定7日历史快照 %s", cutoff)
    for row in connection.execute(
        query, {"cutoff": cutoff, "start": start, "finish": finish}
    ):
        user, item, pv, fav, cart, buy, count, first, last, last_buy, days, target = row
        all_future += target
        if count:
            values = item_stats.setdefault(item, [0, 0, 0])
            values[0] += count
            values[1] += buy
            values[2] += 1
        else:
            continue
        candidates += 1
        positives += target
        if len(records) < max_pairs:
            records.append(row)
        else:
            position = rng.randint(candidates)
            if position < max_pairs:
                records[position] = row
    frame = pd.DataFrame.from_records(
        records,
        columns=[
            "user_id",
            "item_id",
            "pair_pv",
            "pair_fav",
            "pair_cart",
            "pair_buys",
            "pair_recent_events",
            "first_time",
            "last_time",
            "last_buy",
            "pair_days",
            "target",
        ],
    )
    for position, name in enumerate(["item_events", "item_buys", "item_users"]):
        frame[name] = [item_stats[item][position] for item in frame.item_id]
    del records, item_stats
    gc.collect()
    users = pd.read_sql_query(
        """SELECT user_id,COUNT(*) user_events,COUNT(DISTINCT item_id) user_items,
      COUNT(DISTINCT date(event_time)) user_days,SUM(behavior_type=1) user_pv,
      SUM(behavior_type=2) user_fav,SUM(behavior_type=3) user_cart,SUM(behavior_type=4) user_buys,
      MAX(event_time) user_last,MAX(CASE WHEN behavior_type=4 THEN event_time END) user_last_buy
      FROM user_behavior INDEXED BY idx_behavior_quadruple WHERE event_time>=? AND event_time<? GROUP BY user_id""",
        connection,
        params=(start, cutoff),
    )
    frame = frame.merge(users, on="user_id", validate="many_to_one")
    for field in ["first_time", "last_time", "last_buy", "user_last", "user_last_buy"]:
        frame[field + "_days_ago"] = (
            pd.Timestamp(cutoff) - pd.to_datetime(frame[field])
        ).dt.total_seconds() / 86400
    total = frame.pair_pv + frame.pair_fav + frame.pair_cart + frame.pair_buys
    frame["pair_buy_share"] = frame.pair_buys / total
    frame["item_buy_share"] = frame.item_buys / frame.item_events
    frame["user_buy_share"] = frame.user_buys / frame.user_events
    frame["pair_user_event_share"] = total / frame.user_events
    category_map = {}
    items = frame.item_id.unique().tolist()
    for position in range(0, len(items), 800):
        batch = items[position : position + 800]
        category_map.update(
            connection.execute(
                "SELECT item_id,category_id FROM dim_item WHERE item_id IN ("
                + ",".join("?" for _ in batch)
                + ")",
                batch,
            )
        )
    connection.close()
    frame["category_id"] = frame.item_id.map(category_map).fillna("unknown")
    frame["cutoff"] = cutoff
    if frame.target.nunique() != 2 or frame.duplicated(["user_id", "item_id"]).any():
        raise ValueError("标签不足两类或样本组合不唯一")
    frame.to_parquet(destination, index=False)
    info = {
        "history_start": start,
        "cutoff": cutoff,
        "label_end_exclusive": finish,
        "history_days": 7,
        "seed": seed,
        "max_pairs": max_pairs,
        "sample_pairs": len(frame),
        "sample_positives": int(frame.target.sum()),
        "candidate_pairs": candidates,
        "candidate_positives": positives,
        "all_future_purchase_pairs": all_future,
        "candidate_purchase_coverage": positives / all_future if all_future else None,
        "sampling": "uniform reservoir independent of future labels",
    }
    destination.with_suffix(".json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("完成快照 %s：%d对，%d正例", cutoff, len(frame), frame.target.sum())
    return frame, info


def prepare_matrices(early, late, train):
    """预处理仅较早训练子集拟合；晚期标签不参与编码或缩放。"""
    joined = pd.concat([early, late], ignore_index=True)
    y = joined.target.to_numpy(dtype=np.int64)
    raw = joined[NUMERIC].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = raw.iloc[train].median().fillna(0)
    values = np.log1p(raw.fillna(medians).clip(lower=0)).to_numpy(np.float32)
    encoded, encoder = encode_category(joined.category_id, y, train, seed=42)
    values = np.column_stack([values, encoded])
    scaler = StandardScaler().fit(values[train])
    x = scaler.transform(values).astype(np.float32)
    return x, {
        "medians": medians,
        "scaler": scaler,
        "target_encoder": encoder,
        "features": NUMERIC + ["category_target_encoding"],
        "history_days": 7,
    }


def resample_training(x, y, strategy):
    if strategy == "smote":
        return SMOTE(
            random_state=42, k_neighbors=min(5, int(y.sum()) - 1)
        ).fit_resample(x, y)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=42).fit_resample(x, y)
    return x, y


def train_earlier_models(x, y, train, valid, output, trials):
    """仅较早验证集选择策略和超参，不评价测试或晚期标签。"""
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    selection = {}
    rows = []
    ratio = float((y[train] == 0).sum() / y[train].sum())
    storage = "sqlite:///" + str((output / "optuna.sqlite").absolute()).replace(
        "\\", "/"
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name, base in BASELINES.items():
        logging.info("较早窗口训练 %s", name)
        best, info = -1.0, None
        for strategy in ["none", "smote", "undersample", "class_weight"]:
            fit_x, fit_y = resample_training(x[train], y[train], strategy)
            started = time.monotonic()
            model = classifier(name, base, strategy, ratio).fit(fit_x, fit_y)
            probability = model.predict_proba(x[valid])[:, 1]
            row = {
                "model": name,
                "strategy": strategy,
                "fit_rows": len(fit_y),
                "fit_positives": int(fit_y.sum()),
                "seconds": time.monotonic() - started,
                **metrics(y[valid], probability),
            }
            rows.append(row)
            logging.info("%s %s validationAUC %.5f", name, strategy, row["auc"])
            if row["auc"] > best:
                best = row["auc"]
                info = {
                    "model": model,
                    "strategy": strategy,
                    "params": base.copy(),
                    "validation_auc": best,
                }
            pd.DataFrame(rows).to_csv(
                output / "imbalance_validation_comparison.csv", index=False
            )
            del fit_x, fit_y
            gc.collect()
        strategy = info["strategy"]
        fit_x, fit_y = resample_training(x[train], y[train], strategy)
        study = optuna.create_study(
            study_name=name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.enqueue_trial(base)

        def objective(trial):
            nonlocal info
            params = suggest(trial, name)
            fitted = classifier(name, params, strategy, ratio).fit(fit_x, fit_y)
            auc = float(roc_auc_score(y[valid], fitted.predict_proba(x[valid])[:, 1]))
            if auc > info["validation_auc"]:
                info.update(model=fitted, params=params, validation_auc=auc)
            logging.info("%s Optuna %d validationAUC %.5f", name, trial.number, auc)
            return auc

        study.optimize(objective, n_trials=trials)
        study.trials_dataframe().to_csv(
            output / (name + "_optuna_trials.csv"), index=False
        )
        p_valid = info["model"].predict_proba(x[valid])[:, 1]
        info["threshold"] = choose_threshold(y[valid], p_valid)
        joblib.dump(info["model"], model_dir / (name + ".joblib"))
        selection[name] = {key: value for key, value in info.items() if key != "model"}
        del fit_x, fit_y, model, info
        gc.collect()
    (output / "frozen_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def run_promotion_review(database, output, max_pairs=120000, trials=12):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").exists():
        raise FileExistsError("本次复核已完成，请查阅报告或使用新目录")
    if (output / "optuna.sqlite").exists() and not (
        output / "frozen_selection.json"
    ).exists():
        raise FileExistsError("发现未完成调优，请使用新训练目录；可以复制已完成快照")
    early, early_info = build_snapshot(
        database, EARLY, output / "earlier_snapshot.parquet", max_pairs
    )
    late, late_info = build_snapshot(
        database, LATE, output / "dec12_snapshot.parquet", max_pairs
    )
    y_early = early.target.to_numpy(np.int64)
    train, valid, test = split_indices(y_early, 42)
    x, preprocessing = prepare_matrices(early, late, train)
    joblib.dump(preprocessing, output / "preprocessing.joblib")
    early["split"] = "train"
    early.loc[valid, "split"] = "validation"
    early.loc[test, "split"] = "test"
    early.to_parquet(output / "earlier_samples_with_split.parquet", index=False)
    np.savez_compressed(
        output / "matrices.npz",
        x=x,
        y=np.concatenate([y_early, late.target.to_numpy(np.int64)]),
        train=train,
        valid=valid,
        test=test,
        stress=np.arange(len(early), len(early) + len(late)),
    )
    config = {
        "earlier": early_info,
        "dec12_stress": late_info,
        "trials_per_model": trials,
        "features": preprocessing["features"],
        "feature_history_days": 7,
        "early_splits": {
            name: {
                "rows": len(index),
                "positives": int(y_early[index].sum()),
                "positive_rate": float(y_early[index].mean()),
            }
            for name, index in [("train", train), ("validation", valid), ("test", test)]
        },
        "development_split": "stratified random 7:2:1 within earlier anchor, seed42",
        "stress_status": "later snapshot frozen before reading evaluation; dates previously inspected in project, not new unseen raw data",
        "limitations": [
            "activity participation and exact campaign windows unavailable; earlier is not certified non-promotion",
            "legacy cleaned data used full-period anomaly thresholds; not fully history-refit end-to-end evaluation",
            "same users and items can cross early subsets and later snapshot",
            "negative means no observed purchase, not verified unpaid order",
        ],
    }
    (output / "run_configuration.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_file = output / "frozen_selection.json"
    selection = (
        json.loads(selection_file.read_text(encoding="utf-8"))
        if selection_file.exists()
        else train_earlier_models(x, y_early, train, valid, output, trials)
    )
    # 至此超参和阈值已冻结；下面不再依据任何测试结果调整策略。
    comparison, prediction_frames = [], []
    for scenario, index, target in [
        ("earlier_holdout", test, y_early[test]),
        (
            "dec12_cross_period_stress",
            np.arange(len(early), len(early) + len(late)),
            late.target.to_numpy(np.int64),
        ),
    ]:
        predictions = {"sample_row": index, "target": target}
        for name, info in selection.items():
            fitted = joblib.load(output / "models" / (name + ".joblib"))
            probability = fitted.predict_proba(x[index])[:, 1]
            comparison.append(
                {
                    "scenario": scenario,
                    "model": name,
                    "rows": len(index),
                    "positives": int(target.sum()),
                    "positive_rate": float(target.mean()),
                    "strategy": info["strategy"],
                    "validation_auc": info["validation_auc"],
                    **metrics(target, probability, info["threshold"]),
                }
            )
            predictions[name] = probability
        pd.DataFrame(predictions).to_parquet(
            output / (scenario + "_predictions.parquet"), index=False
        )
        logging.info("完成评估 %s", scenario)
    pd.DataFrame(comparison).to_csv(output / "model_comparison.csv", index=False)
    from behavior_cleaning.promotion_report import write_promotion_report

    write_promotion_report(database, output, config)
    (output / "SUCCESS").write_text("ok\n", encoding="utf-8")
