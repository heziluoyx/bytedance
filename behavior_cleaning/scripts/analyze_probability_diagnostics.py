"""冻结基础模型的报告补充诊断；新融合仅在meta拟合、calibration校准。"""
import argparse
import os
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="补充概率模型排序、漂移、融合消融与SHAP诊断")
parser.add_argument(
    "--run-dir", type=Path, default=project / "outputs/temporal_probability_20261109"
)
args = parser.parse_args()
for folder in [project / "src", project / ".runtime", project / ".model_runtime"]:
    sys.path.insert(0, str(folder))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

from behavior_cleaning.model_experiments import deep_predict
from behavior_cleaning.temporal_probability_models import (
    apply_calibrator,
    fit_calibrator,
    load_sequence,
    probability_metrics,
)
from behavior_cleaning.traditional_models import choose_threshold

R = args.run_dir.resolve()
W = R / "report_diagnostics"
W.mkdir(parents=True, exist_ok=True)
O = W / "figures"
O.mkdir(exist_ok=True)
E = W / "supplement"
E.mkdir(exist_ok=True)
P = pd.read_parquet(R / "all_window_predictions.parquet")
D = pd.read_parquet(R / "dataset/samples.parquet")
A = dict(np.load(R / "dataset/arrays.npz"))
S = json.loads((R / "dataset/summary.json").read_text("utf-8"))
names = ["LogisticRegression", "XGBoost", "LightGBM", "LSTM", "GRU", "DIN", "DIN_MHA"]
base = ["LogisticRegression", "XGBoost", "DIN"]
ix = {
    k: np.flatnonzero(P.split.eq(k))
    for k in ["train", "validation", "meta", "calibration", "test"]
}
tr, va, me, ca, te = [ix[k] for k in ix]
y = P.target.to_numpy()
t = D.iloc[te].reset_index(drop=True)
yt = y[te]
thresholds = json.loads((R / "models/frozen_thresholds.json").read_text("utf-8"))
F = {}
scores = {n: P[n + "_probability"].to_numpy() for n in names + ["Stacking"]}
plt.rcParams.update(
    {
        "font.size": 10,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(O / name, bbox_inches="tight")
    plt.close(fig)


def psi(a, b):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    edges = np.r_[-np.inf, edges[1:-1], np.inf]
    aa = np.histogram(a, edges)[0] / len(a)
    bb = np.histogram(b, edges)[0] / len(b)
    aa = np.maximum(aa, 1e-6)
    bb = np.maximum(bb, 1e-6)
    return float(np.sum((bb - aa) * np.log(bb / aa)))


def deciles(p):
    groups = np.array_split(np.argsort(-p, kind="stable"), 10)
    return [
        {
            "bucket": i + 1,
            "rows": len(g),
            "positives": int(yt[g].sum()),
            "rate": float(yt[g].mean()),
            "lift": float(yt[g].mean() / yt.mean()),
            "prediction": float(p[g].mean()),
        }
        for i, g in enumerate(groups)
    ]


# 用户内排序只针对每名用户已抽样的候选；所有分母明确保留。
user_groups = list(t.groupby("user_id", sort=False).indices.values())
positive_groups = [g for g in user_groups if yt[g].sum() > 0]
F["ranking_denominators"] = {
    "all_users": len(user_groups),
    "positive_users": len(positive_groups),
    "candidate_count_median": float(t.groupby("user_id").size().median()),
    "candidate_count_below_5": int((t.groupby("user_id").size() < 5).sum()),
}


def ranking(p):
    result = {}
    for k in [5, 10, 20]:
        nd, hits, ap = [], [], []
        for g in positive_groups:
            z = yt[g[np.argsort(-p[g], kind="stable")[:k]]]
            r = int(yt[g].sum())
            dcg = (z / np.log2(np.arange(len(z)) + 2)).sum()
            ideal = (1 / np.log2(np.arange(min(r, k)) + 2)).sum()
            nd.append(dcg / ideal)
            hits.append(int(z.sum() > 0))
            ap.append(
                float((np.cumsum(z) / (np.arange(len(z)) + 1) * z).sum() / min(r, k))
            )
        result[str(k)] = {
            "ndcg": float(np.mean(nd)),
            "hr": float(np.mean(hits)),
            "map": float(np.mean(ap)),
            "hr_all_users": float(np.sum(hits) / len(user_groups)),
        }
    return result


def result(p, th):
    return {
        **probability_metrics(yt, p, th),
        "confusion": confusion_matrix(yt, p >= th).tolist(),
        "ranking": ranking(p),
        "deciles": deciles(p),
    }


F["single"] = {}
for n in names:
    p = scores[n]
    row = result(p[te], thresholds[n])
    raw = P[n + "_raw"].to_numpy()
    row.update(
        train_auc=float(roc_auc_score(y[tr], raw[tr])),
        validation_auc=float(roc_auc_score(y[va], raw[va])),
        score_psi=psi(raw[va], raw[te]),
    )
    F["single"][n] = row
F["label_psi"] = float(
    sum(
        (b - a) * np.log(b / a)
        for a, b in zip([1 - y[va].mean(), y[va].mean()], [1 - yt.mean(), yt.mean()])
    )
)
F["feature_psi"] = sorted(
    [
        {"feature": n, "psi": psi(A["x"][tr, j], A["x"][te, j])}
        for j, n in enumerate(S["features"])
    ],
    key=lambda a: -a["psi"],
)
train_users = set(D.iloc[tr].user_id)
train_items = set(D.iloc[tr].item_id)
meduser = float(D.iloc[tr].drop_duplicates("user_id").user_events.median())
groups = {
    "unseen_user": ~t.user_id.isin(train_users).to_numpy(),
    "known_user": t.user_id.isin(train_users).to_numpy(),
    "unseen_item": ~t.item_id.isin(train_items).to_numpy(),
    "long_tail": t.item_events.le(2).to_numpy(),
    "high_activity": t.user_events.ge(meduser).to_numpy(),
    "low_activity": t.user_events.lt(meduser).to_numpy(),
}
cats = t.category_id.value_counts().head(3).index
for c in cats:
    groups["category_" + str(c)] = t.category_id.eq(c).to_numpy()
F["activity_threshold"] = meduser
F["slices"] = {}
for g, m in groups.items():
    F["slices"][g] = {"rows": int(m.sum()), "positives": int(yt[m].sum()), "models": {}}
    for n in names + ["Stacking"]:
        pp = scores[n][te][m]
        yy = yt[m]
        z = probability_metrics(yy, pp, thresholds[n])
        F["slices"][g]["models"][n] = z
F[
    "unsupported_slices"
] = "price/channel/first-ever-user/weekday-versus-weekend separate snapshots unavailable"
# 补充融合对照，基础模型完全冻结。加权与方案固定只用meta。
X = np.column_stack([scores[n] for n in base])
raw_fusions = {}
recipes = {}
raw_fusions["EqualSoftVote"] = X.mean(1)
recipes["EqualSoftVote"] = {"weights": [1 / 3] * 3}
grid = []
for a in range(11):
    for b in range(11 - a):
        weight = np.array([a, b, 10 - a - b]) / 10
        pp = X[me] @ weight
        grid.append((average_precision_score(y[me], pp), weight))
best = max(grid, key=lambda z: z[0])
weight = best[1]
raw_fusions["WeightedSoftVote"] = X @ weight
recipes["WeightedSoftVote"] = {
    "weights": weight.tolist(),
    "meta_ap": float(best[0]),
    "grid": len(grid),
}
raw_fusions["HardVote"] = np.column_stack(
    [scores[n] >= thresholds[n] for n in base]
).mean(1)
recipes["HardVote"] = {
    "thresholds": {n: thresholds[n] for n in base},
    "meaning": "vote fraction, not probability before calibration",
}
# probability-score inputs, minmax fit meta only. clipping avoids extrapolation beyond training normalization.
def fit_score_stack(cols):
    z = np.column_stack([scores[n] for n in cols])
    sc = MinMaxScaler(clip=True).fit(z[me])
    zz = sc.transform(z)
    model = LogisticRegression(C=0.3, max_iter=600, random_state=42).fit(zz[me], y[me])
    return model.predict_proba(zz)[:, 1], {
        "columns": cols,
        "coefficients": model.coef_[0].tolist(),
        "normalizer": "meta minmax clipped",
        "C": 0.3,
    }


raw_fusions["ScoreStacking"], recipes["ScoreStacking"] = fit_score_stack(base)
for drop in base:
    name = "Drop_" + drop
    raw_fusions[name], recipes[name] = fit_score_stack([n for n in base if n != drop])
# 路由只凭历史交互条数；规则事先指定，不以测试标签优化。
route = D.user_events.ge(5).to_numpy()
raw_fusions["HistoryRoute"] = np.where(
    route, (scores["XGBoost"] + scores["DIN"]) / 2, scores["LogisticRegression"]
)
recipes["HistoryRoute"] = {
    "history_threshold": 5,
    "test_deep_route_share": float(route[te].mean()),
    "high_history": "XGB/DIN equal",
    "low_history": "LR",
}
F["fusion_recipes"] = recipes
F["fusions"] = {}
for n, raw in raw_fusions.items():
    cal = fit_calibrator(y[ca], raw[ca])
    p = apply_calibrator(cal, raw)
    th = choose_threshold(y[ca], p[ca])
    scores[n] = p
    F["fusions"][n] = result(p[te], th)
    F["fusions"][n]["threshold"] = float(th)
F["fusions"]["Stacking"] = result(scores["Stacking"][te], thresholds["Stacking"])
F["fusions"]["XGBoost"] = F["single"]["XGBoost"]
F["fusions"]["LogisticRegression"] = F["single"]["LogisticRegression"]
F["fusion_score_correlation"] = np.corrcoef(X[me].T).tolist()
pd.DataFrame({n: p[te] for n, p in scores.items()}).to_parquet(
    E / "supplementary_test_scores.parquet", index=False
)
# 选型基准固定为验证期AP最高基础模型，不按测试追选。
F["validation_ap"] = {
    n: float(average_precision_score(y[va], P[n + "_raw"].to_numpy()[va]))
    for n in names
}
# 同期切片增益（事后诊断，不以切片结果选路由）。
F["fusion_slices"] = {}
for g, m in groups.items():
    order = lambda n: np.argsort(-scores[n][te][m], kind="stable")[
        : max(1, int(np.ceil(m.sum() * 0.1)))
    ]
    F["fusion_slices"][g] = {
        "rows": int(m.sum()),
        "positives": int(yt[m].sum()),
        "xgb_auc": float(roc_auc_score(yt[m], scores["XGBoost"][te][m])),
        "stack_auc": float(roc_auc_score(yt[m], scores["Stacking"][te][m])),
        "xgb_top10_hits": int(yt[m][order("XGBoost")].sum()),
        "stack_top10_hits": int(yt[m][order("Stacking")].sum()),
    }
# 模型内核批量耗时，复用已生成特征，不含IO、预处理、召回与网络。
torch.set_num_threads(2)
seq = json.loads((R / "models/sequence_selection.json").read_text("utf-8"))
F["timing"] = {}
bench = te[:2048]
for n in names:
    if n in names[:3]:
        model = joblib.load(R / "models" / (n + ".joblib"))
        fn = lambda: model.predict_proba(A["x"][bench])[:, 1]
    else:
        model = load_sequence(n, seq[n], A, S, R)
        fn = lambda: deep_predict(model, A, bench)
    fn()
    elapsed = []
    for _ in range(7):
        start = time.perf_counter()
        fn()
        elapsed.append((time.perf_counter() - start) * 1000)
    F["timing"][n] = {
        "batch_2048_median_ms": float(np.median(elapsed)),
        "batch_p95_ms": float(np.quantile(elapsed, 0.95)),
    }
F["timing_context"] = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "threads": 2,
    "rows": 2048,
    "warmup": 1,
    "repeats": 7,
}
# 单模型专属图，移除任何融合结果。
from sklearn.metrics import precision_recall_curve, roc_curve

for metric, filename in [("roc", "16_single_roc.png"), ("pr", "17_single_pr.png")]:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for n in names:
        pp = scores[n][te]
        if metric == "roc":
            xx, zz, _ = roc_curve(yt, pp)
        else:
            zz, xx, _ = precision_recall_curve(yt, pp)
        ax.plot(xx, zz, label=n, lw=1)
    ax.set(
        xlabel="False positive rate" if metric == "roc" else "Recall",
        ylabel="True positive rate" if metric == "roc" else "Precision",
        title="Single models only - final future window",
    )
    ax.legend(fontsize=7)
    save(fig, filename)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for n in names:
    d = F["single"][n]["deciles"]
    axes[0].plot(
        [v["prediction"] for v in d], [v["rate"] for v in d], marker=".", label=n
    )
axes[0].plot([0, 0.08], [0, 0.08], "--", c="gray")
axes[0].set(
    xlabel="Predicted mean", ylabel="Observed purchase rate", title="Decile calibration"
)
axes[0].legend(fontsize=6)
z = np.arange(7)
axes[1].bar(z - 0.18, [scores[n][te].mean() for n in names], 0.36, label="Predicted")
axes[1].bar(z + 0.18, [yt.mean()] * 7, 0.36, label="Observed")
axes[1].set_xticks(z)
axes[1].set_xticklabels(names, rotation=55, ha="right", fontsize=8)
axes[1].legend(fontsize=8)
save(fig, "18_single_calibration.png")
fig, ax = plt.subplots(figsize=(9, 4.4))
for n in names:
    ax.plot(
        range(1, 11),
        [v["lift"] for v in F["single"][n]["deciles"]],
        marker=".",
        label=n,
    )
ax.set(
    xlabel="Score decile (1 = highest)",
    ylabel="Observed rate / overall rate",
    title="Single-model decile lift",
)
ax.legend(fontsize=7)
save(fig, "19_single_lift.png")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
show = [
    "XGBoost",
    "EqualSoftVote",
    "WeightedSoftVote",
    "ScoreStacking",
    "HistoryRoute",
    "Stacking",
]
axes[0].barh(show, [F["fusions"][n]["auc"] for n in show], color="#3B7C9C")
axes[0].set(xlim=(0.80, 0.90), xlabel="AUC")
axes[1].barh(
    show, [F["fusions"][n]["deciles"][0]["lift"] for n in show], color="#B45C47"
)
axes[1].set(xlabel="Top 10% lift")
save(fig, "20_fusion_comparison.png")
fig, ax = plt.subplots(figsize=(9, 4.2))
for n in ["XGBoost", "Stacking", "ScoreStacking", "WeightedSoftVote"]:
    ax.plot(
        range(1, 11),
        [v["rate"] for v in F["fusions"][n]["deciles"]],
        marker="o",
        label=n,
    )
ax.set(
    xlabel="Score decile (1 = highest)",
    ylabel="Observed purchase rate",
    title="Fusion decile comparison",
)
ax.legend()
save(fig, "21_fusion_deciles.png")
# SHAP依赖、精确交互、分群、四类case、验证期漂移。
sh = np.load(R / "interpretability/xgboost_shap_values.npz")
ind = sh["indices"]
con = sh["contributions"]
feat = S["features"]
sf = D.iloc[ind].reset_index(drop=True)
calibrators = joblib.load(R / "models/probability_calibrators.joblib")
slope = float(calibrators["XGBoost"].coef_[0, 0])
intercept = float(calibrators["XGBoost"].intercept_[0])
model = joblib.load(R / "models/XGBoost.joblib")
j = feat.index("pair_recent_events")
q = feat.index("pair_cart")
fig, ax = plt.subplots(figsize=(9, 4.5))
sc = ax.scatter(
    sf.pair_recent_events, con[:, j], c=sf.pair_cart, cmap="coolwarm", s=10, alpha=0.6
)
ax.set(
    xlabel="Pair interactions in history (raw count)",
    ylabel="SHAP contribution to calibrated logit",
    title="Dependence: recent interactions, colored by cart count",
    xscale="symlog",
)
fig.colorbar(sc, ax=ax, label="Pair cart count")
save(fig, "22_shap_dependence.png")
iv = (
    model.get_booster().predict(
        xgb.DMatrix(A["x"][ind]), pred_interactions=True, approx_contribs=False
    )
    * slope
)
add_err = float(np.max(np.abs(iv.sum(2)[:, :-1] - con[:, :-1])))
fig, ax = plt.subplots(figsize=(9, 4.5))
sc = ax.scatter(
    sf.pair_recent_events, iv[:, j, q], c=sf.pair_cart, cmap="coolwarm", s=10, alpha=0.6
)
ax.set(
    xlabel="Pair interactions in history (raw count)",
    ylabel="TreeSHAP interaction value",
    title="Exact TreeSHAP interaction: recent interactions x cart",
    xscale="symlog",
)
fig.colorbar(sc, ax=ax, label="Pair cart count")
save(fig, "23_shap_interaction.png")
F["interaction"] = {
    "max_row_sum_error": add_err,
    "mean_abs": float(np.abs(iv[:, j, q]).mean()),
    "positive_cart_median": float(np.median(con[sf.pair_cart.gt(0), q])),
    "zero_cart_median": float(np.median(con[sf.pair_cart.eq(0), q])),
}
new = ~sf.user_id.isin(train_users).to_numpy()
positive = sf.target.eq(1).to_numpy()
rank = np.argsort(np.abs(con[:, :-1]).mean(0))[::-1][:8]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
F["shap_groups"] = {}
for ax, mask, label in [
    (axes[0], new, "Training-unseen vs known users"),
    (axes[1], positive, "Future buyers vs nonbuyers"),
]:
    vals = []
    for m, name, offset, color in [
        (mask, "Unseen" if ax == axes[0] else "Buyer", -0.18, "#B45C47"),
        (~mask, "Known" if ax == axes[0] else "Nonbuyer", 0.18, "#3B7C9C"),
    ]:
        v = np.abs(con[m, :-1]).mean(0)
        ax.barh(np.arange(8) + offset, v[rank[::-1]], 0.36, label=name, color=color)
        F["shap_groups"][name] = {
            "rows": int(m.sum()),
            "top": [
                {"feature": feat[z], "mean_abs": float(v[z])}
                for z in np.argsort(-v)[:5]
            ],
        }
    ax.set_yticks(range(8))
    ax.set_yticklabels([feat[z] for z in rank[::-1]], fontsize=8)
    ax.set(title=label, xlabel="Mean absolute SHAP")
    ax.legend(fontsize=8)
save(fig, "24_shap_groups.png")
pp = P.iloc[ind].XGBoost_probability.to_numpy()
yy = sf.target.to_numpy()
pred = pp >= thresholds["XGBoost"]
cases = {}
for name, mask, rule in [
    ("High_TP", (yy == 1) & pred, "max"),
    ("Low_TN", (yy == 0) & ~pred, "min"),
    ("False_positive", (yy == 0) & pred, "max"),
    ("False_negative", (yy == 1) & ~pred, "median"),
]:
    choices = np.flatnonzero(mask)
    order = choices[np.argsort(pp[choices])]
    z = int(
        order[-1]
        if rule == "max"
        else order[0]
        if rule == "min"
        else order[len(order) // 2]
    )
    cases[name] = z
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
F["cases"] = {}
for ax, (name, z) in zip(axes.flat, cases.items()):
    v = con[z]
    order = np.argsort(-np.abs(v[:-1]))[:5]
    values = list(v[order]) + [float(v[:-1].sum() - v[order].sum())]
    labels = [feat[j] for j in order] + ["Other features"]
    running = float(v[-1])
    for a, value in enumerate(values):
        ax.barh(
            a,
            abs(value),
            left=min(running, running + value),
            color="#B45C47" if value > 0 else "#3B7C9C",
        )
        running += value
    ax.axvline(v[-1], ls="--", c="gray")
    ax.axvline(running, ls="--", c="black")
    ax.set_yticks(range(6))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set(
        title=f"{name}: p={pp[z]:.2%}, y={yy[z]}", xlabel="Cumulative calibrated logit"
    )
    F["cases"][name] = {
        "p": float(pp[z]),
        "y": int(yy[z]),
        "cart": int(sf.iloc[z].pair_cart),
        "events": int(sf.iloc[z].pair_recent_events),
        "last_hours": float(sf.iloc[z].last_time_days_ago * 24),
        "top_features": labels[:5],
    }
save(fig, "25_four_shap_cases.png")
rng = np.random.default_rng(42)
vi = rng.choice(va, 2000, replace=False)
vc = (
    model.get_booster().predict(xgb.DMatrix(A["x"][vi]), pred_contribs=True)[:, :-1]
    * slope
)
vimp = np.abs(vc).mean(0)
timp = np.abs(con[:, :-1]).mean(0)
F["shap_stability"] = {
    "spearman": float(spearmanr(vimp, timp).statistic),
    "top10_overlap": len(set(np.argsort(-vimp)[:10]) & set(np.argsort(-timp)[:10])),
    "validation_positives": int(y[vi].sum()),
    "test_positives": int(yy.sum()),
}
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.barh(np.arange(8) - 0.18, vimp[rank[::-1]], 0.36, label="Earlier validation")
ax.barh(np.arange(8) + 0.18, timp[rank[::-1]], 0.36, label="Final test")
ax.set_yticks(range(8))
ax.set_yticklabels([feat[z] for z in rank[::-1]])
ax.set(xlabel="Mean absolute SHAP", title="Same frozen XGBoost, two time windows")
ax.legend()
save(fig, "26_shap_stability.png")
# 既有Stacking第二层的模型级精确线性归因，解释单位不是原始业务字段。
st = joblib.load(R / "models/Stacking.joblib")
Z = np.column_stack([logit(np.clip(P[n + "_raw"], 1e-6, 1 - 1e-6)) for n in base])
std = st["scaler"].transform(Z)
contrib = (
    (std[te] - std[me].mean(0))
    * st["model"].coef_[0]
    * float(calibrators["Stacking"].coef_[0, 0])
)
F["stack_model_attribution"] = {
    n: float(v) for n, v in zip(base, np.abs(contrib).mean(0))
}
fig, ax = plt.subplots(figsize=(8, 3.7))
ax.barh(base, np.abs(contrib).mean(0), color="#3B7C9C")
ax.set(
    xlabel="Mean absolute contribution to final calibrated logit",
    title="Frozen Stacking: second-layer model-input attribution",
)
save(fig, "27_stack_attribution.png")
# 强度不同切片的概率偏差，不声称敏感属性公平性。
F[
    "note"
] = "Supplementary retrospective diagnostics; base models frozen; new fusion recipes fitted meta, calibrated cal; test already inspected, not fresh confirmation."
(E / "evidence.json").write_text(
    json.dumps(F, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
)
print(
    json.dumps(
        {
            "ranking": F["ranking_denominators"],
            "validation_ap": F["validation_ap"],
            "weighted": F["fusion_recipes"]["WeightedSoftVote"],
            "route": F["fusion_recipes"]["HistoryRoute"],
            "fusion_metrics": {
                n: {
                    "auc": v["auc"],
                    "ap": v["average_precision"],
                    "ndcg10": v["ranking"]["10"]["ndcg"],
                    "lift10": v["deciles"][0]["lift"],
                }
                for n, v in F["fusions"].items()
            },
            "stability": F["shap_stability"],
            "cases": F["cases"],
            "psi": F["feature_psi"][:5],
        },
        ensure_ascii=False,
        indent=2,
    )
)
