"""三折折外Stacking；折内重新拟合填补、缩放与目标编码。"""

import gc
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from behavior_cleaning.model_experiments import (
    choose_threshold,
    classifier,
    deep_predict,
    metrics,
    seed_everything,
    sequence_batch,
)
from behavior_cleaning.pair_dataset import encode_category
from behavior_cleaning.sequence_models import PurchaseSequenceModel

BASE_NAMES = ["LogisticRegression", "XGBoost", "DIN"]


def logit_features(probabilities):
    """避免加权模型的概率尺度不一致；仅在OOF上拟合二层缩放。"""
    p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fold_numeric(frame, y, train, features, seed):
    """外层折的验证标签不能影响该折填补、缩放或类别目标映射。"""
    numeric = features[:-1]
    assert features[-1] == "category_target_encoding"
    raw = frame[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = raw.iloc[train].median().fillna(0)
    values = np.log1p(raw.fillna(medians).clip(lower=0)).to_numpy(np.float32)
    encoded, encoder = encode_category(frame.category_id, y, train, seed)
    values = np.column_stack([values, encoded])
    scaler = StandardScaler().fit(values[train])
    return scaler.transform(values).astype(np.float32), {
        "medians": medians,
        "scaler": scaler,
        "target_encoder": encoder,
        "features": features,
    }


def load_din(run):
    checkpoint = torch.load(Path(run) / "models/DIN.pt", map_location="cpu")
    model = PurchaseSequenceModel(
        "DIN",
        checkpoint["item_count"],
        checkpoint["category_count"],
        checkpoint["numeric_count"],
        **{k: v for k, v in checkpoint["params"].items() if k != "learning_rate"},
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def transform_fold_numeric(frame, preprocessing):
    """重放折内预处理；保持目标编码float32，避免树阈值处精度漂移。"""
    features = preprocessing["features"]
    raw = frame[features[:-1]].astype(float).replace([np.inf, -np.inf], np.nan)
    values = np.log1p(raw.fillna(preprocessing["medians"]).clip(lower=0)).to_numpy(
        np.float32
    )
    encoder = preprocessing["target_encoder"]
    encoded = (
        frame.category_id.map(encoder["mapping"])
        .fillna(encoder["prior"])
        .to_numpy(dtype=np.float32)
    )
    return (
        preprocessing["scaler"]
        .transform(np.column_stack([values, encoded]))
        .astype(np.float32)
    )


def fixed_epoch_din(arrays, summary, train, params, epochs, seed, output):
    """固定原验证集所选轮数，不使用外层留出标签早停，保证OOF隔离。"""
    seed_everything(seed)
    model = PurchaseSequenceModel(
        "DIN",
        summary["item_vocabulary"],
        summary["category_vocabulary"],
        arrays["x"].shape[1],
        **{k: v for k, v in params.items() if k != "learning_rate"},
    )
    y = arrays["y"][train]
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float((y == 0).sum() / y.sum()))
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=1e-5
    )
    rng = np.random.RandomState(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        shuffled = rng.permutation(train)
        for start in range(0, len(shuffled), 1024):
            batch = sequence_batch(arrays, shuffled[start : start + 1024])
            optimizer.zero_grad()
            loss = loss_fn(model(*batch[:-1]), batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch[-1])
        history.append(
            {"epoch": epoch + 1, "weighted_train_loss": loss_sum / len(train)}
        )
        logging.info(
            "OOF DIN %s epoch %d/%d loss %.4f",
            output.name,
            epoch + 1,
            epochs,
            history[-1]["weighted_train_loss"],
        )
        pd.DataFrame(history).to_csv(output, index=False)
    return model.eval()


def source_signature(run):
    paths = [
        run / "dataset/dataset.npz",
        run / "dataset/pair_samples.parquet",
        run / "experiment_summary.json",
    ]
    paths += [
        run / "models" / (name + (".pt" if name == "DIN" else ".joblib"))
        for name in BASE_NAMES
    ]
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(run))] = digest.hexdigest()
    return result


def run_stacking(run: Path, output: Path):
    """重训三折基模型，再拟合OOF二层；原验证集选阈值，测试仅报告。"""
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = source_signature(run)
    signature_path = output / "source_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("源模型或数据已改变，请使用新的输出目录")
    if (output / "SUCCESS").exists():
        logging.info("复用已完成的Stacking产物")
        return
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    arrays = dict(np.load(run / "dataset/dataset.npz"))
    frame = pd.read_parquet(run / "dataset/pair_samples.parquet")
    summary = json.loads(
        (run / "dataset/dataset_summary.json").read_text(encoding="utf-8")
    )
    original = json.loads((run / "experiment_summary.json").read_text(encoding="utf-8"))
    din_info = original["deep"]["DIN"]
    curve = pd.read_csv(run / f"DIN_trial{din_info['best_trial']}_history.csv")
    epochs = int(curve.loc[curve.validation_auc.idxmax(), "epoch"])
    train = arrays["train"]
    y = arrays["y"]
    oof = np.full((len(train), 3), np.nan, dtype=np.float64)
    fold_ids = np.full(len(train), -1, dtype=int)
    splitter = StratifiedKFold(3, shuffle=True, random_state=20261102)
    for fold, (fit_local, held_local) in enumerate(splitter.split(train, y[train])):
        held = train[held_local]
        fit = train[fit_local]
        saved = output / f"fold{fold}_predictions.npz"
        if saved.exists():
            values = np.load(saved)
            assert np.array_equal(values["held_rows"], held)
            oof[held_local] = values["probabilities"]
            fold_ids[held_local] = fold
            logging.info("复用折 %d", fold)
            continue
        logging.info("训练Stacking折 %d：fit %d，held %d", fold, len(fit), len(held))
        numeric, preprocessing = fold_numeric(
            frame, y, fit, summary["features"], 500 + fold
        )
        joblib.dump(preprocessing, output / f"fold{fold}_preprocessing.joblib")
        fold_arrays = {**arrays, "x": numeric}
        predictions = []
        ratio = float((y[fit] == 0).sum() / (y[fit] == 1).sum())
        for name in BASE_NAMES[:2]:
            config = original["traditional"][name]
            model = classifier(name, config["params"], config["strategy"], ratio)
            model.fit(numeric[fit], y[fit])
            predictions.append(model.predict_proba(numeric[held])[:, 1])
            joblib.dump(model, output / f"fold{fold}_{name}.joblib")
        model = fixed_epoch_din(
            fold_arrays,
            summary,
            fit,
            din_info["params"],
            epochs,
            900 + fold,
            output / f"fold{fold}_DIN_history.csv",
        )
        predictions.append(deep_predict(model, fold_arrays, held))
        torch.save(
            {
                "state_dict": model.state_dict(),
                "params": din_info["params"],
                "item_count": summary["item_vocabulary"],
                "category_count": summary["category_vocabulary"],
                "numeric_count": numeric.shape[1],
                "kind": "DIN",
            },
            output / f"fold{fold}_DIN.pt",
        )
        values = np.column_stack(predictions)
        np.savez_compressed(saved, held_rows=held, fit_rows=fit, probabilities=values)
        oof[held_local], fold_ids[held_local] = values, fold
        logging.info(
            "完成折 %d，OOF AUC %s",
            fold,
            [round(metrics(y[held], p)["auc"], 5) for p in predictions],
        )
        del fold_arrays, model, numeric
        gc.collect()
    assert np.isfinite(oof).all() and (fold_ids >= 0).all()
    meta = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=600, random_state=42)
    )
    meta.fit(logit_features(oof), y[train])
    joblib.dump(meta, output / "stacking_meta.joblib")
    pd.DataFrame(
        {
            "sample_row": train,
            "fold": fold_ids,
            "target": y[train],
            **dict(zip(BASE_NAMES, oof.T)),
        }
    ).to_parquet(output / "oof_predictions.parquet", index=False)
    valid = arrays["valid"]
    valid_base = [
        joblib.load(run / "models" / (name + ".joblib")).predict_proba(
            arrays["x"][valid]
        )[:, 1]
        for name in BASE_NAMES[:2]
    ]
    valid_base.append(deep_predict(load_din(run), arrays, valid))
    p_valid = meta.predict_proba(logit_features(np.column_stack(valid_base)))[:, 1]
    threshold = choose_threshold(y[valid], p_valid)
    top_fraction = 0.1
    ranking_cutoff = float(np.quantile(p_valid, 1 - top_fraction))
    frozen = {
        "base_models": BASE_NAMES,
        "folds": 3,
        "seed": 20261102,
        "DIN_fixed_epochs": epochs,
        "meta_C": 1.0,
        "meta_class_weight": None,
        "threshold": threshold,
        "validation_top10_score_cutoff": ranking_cutoff,
        "validation_metrics": metrics(y[valid], p_valid, threshold),
        "meta_coefficients": meta[-1].coef_[0].tolist(),
        "meta_intercept": float(meta[-1].intercept_[0]),
        "vocabulary_policy": "fixed original train-only unsupervised vocabulary; supervised encoding and numeric preprocessing refit within folds",
        "evaluation_status": "previously inspected holdout; extension evaluation, not a new untouched test",
    }
    (output / "frozen_configuration.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {"sample_row": valid, "target": y[valid], "Stacking": p_valid}
    ).to_parquet(output / "validation_predictions.parquet", index=False)
    # 所有超参与阈值已经写盘；不依据下列测试结果重新训练或选择融合方案。
    test_frame = pd.read_parquet(run / "test_predictions.parquet")
    p_test = meta.predict_proba(logit_features(test_frame[BASE_NAMES].to_numpy()))[:, 1]
    test_frame["Stacking"] = p_test
    test_frame.to_parquet(output / "test_predictions.parquet", index=False)
    comparison = pd.read_csv(run / "model_test_comparison.csv")
    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "model": "Stacking",
                        "strategy": "OOF_meta",
                        "validation_auc": frozen["validation_metrics"]["auc"],
                        **metrics(test_frame.target.to_numpy(), p_test, threshold),
                        "accuracy_at_0_5": float(
                            np.mean((p_test >= 0.5) == test_frame.target.to_numpy())
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    comparison.to_csv(output / "model_comparison.csv", index=False)
    (output / "SUCCESS").write_text("ok\n", encoding="utf-8")
    logging.info(
        "Stacking测试指标：%s", metrics(test_frame.target.to_numpy(), p_test, threshold)
    )
