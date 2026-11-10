"""促销窗口敏感性、跨期模型表现与历史报告修订说明。"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from behavior_cleaning.analysis_runner import connect_readonly
from behavior_cleaning.promotion_robustness import NUMERIC
from behavior_cleaning.report_utils import save_report, table

ACTIVITY_SOURCE = "https://news.pedaily.cn/20251201/119266.shtml"
ACTIVITY_CONTEXT = "https://sw.wuhan.gov.cn/xwdt/mtbd/202512/t20251229_2703604.shtml"


def population_stability(reference, comparison):
    """训练分位箱加平滑的PSI描述量，不以固定阈值证明因果。"""
    a, b = np.asarray(reference, float), np.asarray(comparison, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    cuts = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    if len(cuts) < 2:
        # 常量列另外记录分布变化，不用一个无区分度箱掩盖变化。
        epsilon = max(abs(a[0]) * 1e-6, 1e-6)
        cuts = np.array([-np.inf, a[0] - epsilon, a[0] + epsilon, np.inf])
    else:
        cuts[0], cuts[-1] = -np.inf, np.inf
    left = np.histogram(a, cuts)[0].astype(float) + 0.5
    right = np.histogram(b, cuts)[0].astype(float) + 0.5
    left /= left.sum()
    right /= right.sum()
    return float(np.sum((right - left) * np.log(right / left)))


def write_promotion_report(database, output, config):
    output = Path(output)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    with connect_readonly(Path(database)) as connection:
        daily = pd.read_sql_query(
            """SELECT date(event_time) day,COUNT(*) events,
          COUNT(DISTINCT user_id) dau,SUM(behavior_type=4) buy_events,
          COUNT(DISTINCT CASE WHEN behavior_type=4 THEN user_id END) buyers
          FROM user_behavior INDEXED BY idx_behavior_quadruple GROUP BY date(event_time) ORDER BY day""",
            connection,
        )
    daily["daily_buyer_share"] = daily.buyers / daily.dau
    daily.to_csv(output / "daily_distribution.csv", index=False)
    windows = []
    for name, start, finish in [
        ("较早标签周", "2025-11-25", "2025-12-02"),
        ("旧用户训练标签周", "2025-12-05", "2025-12-12"),
        ("覆盖双十二标签周", "2025-12-12", "2025-12-19"),
    ]:
        subset = daily[(daily.day >= start) & (daily.day < finish)]
        windows.append(
            {
                "window": name,
                "start": start,
                "end_exclusive": finish,
                "days": len(subset),
                "mean_dau": float(subset.dau.mean()),
                "mean_daily_buy_events": float(subset.buy_events.mean()),
                "mean_daily_buyer_share": float(subset.daily_buyer_share.mean()),
            }
        )
    windows = pd.DataFrame(windows)
    windows.to_csv(output / "window_distribution_comparison.csv", index=False)
    early = pd.read_parquet(output / "earlier_samples_with_split.parquet")
    late = pd.read_parquet(output / "dec12_snapshot.parquet")
    train = early[early.split == "train"]
    drift = []
    for name in NUMERIC:
        drift.append(
            {
                "feature": name,
                "earlier_train_median": float(train[name].median()),
                "dec12_median": float(late[name].median()),
                "earlier_train_missing": float(train[name].isna().mean()),
                "dec12_missing": float(late[name].isna().mean()),
                "psi": population_stability(train[name], late[name]),
            }
        )
    drift = pd.DataFrame(drift).sort_values("psi", ascending=False)
    drift.to_csv(output / "feature_distribution_drift.csv", index=False)
    comparison = pd.read_csv(output / "model_comparison.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 130})
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, column, title in zip(
        axes,
        ["dau", "buy_events", "daily_buyer_share"],
        [
            "Daily active users",
            "Daily purchase events",
            "Daily buyer / active-user share",
        ],
    ):
        ax.plot(pd.to_datetime(daily.day), daily[column], marker=".", color="#326b8c")
        ax.axvline(
            pd.Timestamp("2025-12-03"),
            color="#d49c40",
            linestyle="--",
            label="Reported campaign start Dec 3",
        )
        ax.axvline(
            pd.Timestamp("2025-12-12"), color="#bb533d", linestyle=":", label="Dec 12"
        )
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots / "01_daily_distribution.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, column in zip(axes, ["auc", "average_precision", "f1"]):
        comparison.pivot(index="model", columns="scenario", values=column).plot.bar(
            ax=ax
        )
        ax.set_title(column)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "02_frozen_model_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, scenario in enumerate(["earlier_holdout", "dec12_cross_period_stress"]):
        p = pd.read_parquet(output / (scenario + "_predictions.parquet"))
        for name in ["LogisticRegression", "XGBoost", "LightGBM"]:
            fpr, tpr, _ = roc_curve(p.target, p[name])
            precision, recall, _ = precision_recall_curve(p.target, p[name])
            axes[row, 0].plot(fpr, tpr, label=name)
            axes[row, 1].plot(recall, precision, label=name)
        axes[row, 0].plot([0, 1], [0, 1], "--", color="grey")
        axes[row, 1].axhline(p.target.mean(), linestyle="--", color="grey")
        axes[row, 0].set(
            title=scenario + " ROC", xlabel="False positive rate", ylabel="Recall"
        )
        axes[row, 1].set(title=scenario + " PR", xlabel="Recall", ylabel="Precision")
        for ax in axes[row]:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "03_roc_pr_by_window.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    shown = drift.head(12).iloc[::-1]
    ax.barh(shown.feature, shown.psi, color="#326b8c")
    ax.set(
        title="Feature distribution drift: fixed 7-day histories",
        xlabel="PSI descriptive statistic",
    )
    fig.tight_layout()
    fig.savefig(plots / "04_feature_drift.png")
    plt.close(fig)
    early_best = (
        comparison[comparison.scenario == "earlier_holdout"]
        .sort_values("auc", ascending=False)
        .iloc[0]
    )
    late_best = (
        comparison[comparison.scenario == "dec12_cross_period_stress"]
        .sort_values("auc", ascending=False)
        .iloc[0]
    )
    lines = [
        "# 双十二窗口敏感性与传统模型重训报告",
        "",
        "## 结论与适用范围",
        "",
        "原评估存在促销期代表性风险。2025年12月12日前后的流量、人群、优惠和购买节奏可能改变，某个窗口的AUC与购买率不能直接代表常态经营。它不必然导致计算错误，也不能仅因模型分数较高就认定促销造成高估；本次实际重训并分别报告较早窗口和跨期压力测试。",
        f"较早窗口留出集AUC最高的是{early_best.model}，为{early_best.auc:.4f}；覆盖双十二的压力样本中，最高的是{late_best.model}，为{late_best.auc:.4f}。这些排序是在冻结后描述测试结果，未据此继续调参或重新挑选阈值。",
        "本次评估用户—商品对任务中的逻辑回归、XGBoost、LightGBM。早期用户级61项特征预评估的0.7618与本次目标和划分不同，保留为历史结果，不与本次传统模型混排。",
        "",
        "## 活动日期依据与识别边界",
        "",
        "公开活动介绍称2025抖音商城年终狂欢季于12月3日启动；武汉市商务局转载报道也记载了这一日期。由此看，旧用户级训练标签12月5日至11日与验证标签12月12日至18日都处于该活动覆盖阶段附近，不能把12月12日作为唯一活动开始日，也不能简单把前一周叫作无促销对照。",
        f"[活动介绍]({ACTIVITY_SOURCE})；[武汉市商务局转载报道]({ACTIVITY_CONTEXT})。这些信息仅作为平台活动背景；源行为表没有活动ID、报名状态、优惠券和渠道曝光，无法确认每条记录是否实际参与活动，也不能确认样本来源覆盖全平台。",
        "因此将11月25日至12月1日称为较早标签窗口，不称为已证明的无促销期。预热、其他活动、季节性、用户构成与数据生成过程仍可能影响它；本次是稳健性和分布诊断，不是促销因果效应估计。",
        "",
        "## 新训练与评估设计",
        "",
        "较早预测时点为11月25日00时，特征只用11月18日至24日，标签为11月25日至12月1日。候选仅取此前7天有交互的用户商品对，再独立于未来标签做均匀蓄水池抽样。该快照内部按标签分层7:2:1、随机种子42划分训练、验证和留出测试，保留原任务要求。",
        "跨期压力预测时点为12月12日00时，特征只用12月5日至11日，标签为12月12日至18日。压力样本不参与重采样、编码拟合、超参选择、早停或阈值选择。两个时点的所有累计特征都限定为固定7天，避免旧方案17天与24天累计历史长度不同造成额外混淆。",
        "这不是把较早训练和验证也改为完全时间隔离：较早快照内仍为随机留出，同一用户或商品可以跨子集。真正跨时间的是较早开发完成后冻结评估12月12日快照。现有31天数据不足以在统一7日历史和7日标签下，构造三个完全不重叠、且都早于12月3日活动启动的连续训练验证测试窗口；需要更早或更长数据才能做到。",
        "",
        "| 集合 | 预测截点 | 样本数 | 正例 | 正例比例 |",
        "|---|---|---:|---:|---:|",
    ]
    for name, info in config["early_splits"].items():
        lines.append(
            f"| 较早{name} | 2025-11-25 | {info['rows']:,} | {info['positives']:,} | {info['positive_rate']:.3%} |"
        )
    info = config["dec12_stress"]
    lines += [
        f"| 跨期压力测试 | 2025-12-12 | {info['sample_pairs']:,} | {info['sample_positives']:,} | {info['sample_positives']/info['sample_pairs']:.3%} |",
        "",
        "## 重训与调优实际完成情况",
        "",
        f"三种模型各比较不处理、SMOTE、欠采样、类别权重，共12组基础训练；每种模型再完成{config['trials_per_model']}次Optuna试验，以较早验证AUC选择策略和参数。26项数值及类别编码全部保留，未额外做方差和相关性筛选。固定7天历史下pair_recent_events与历史总交互相等是预期定义重合，不作新的独立信息解释。",
        "填补、log1p、标准化及类别目标编码仅在较早训练子集拟合；训练编码5折折外，验证、较早测试和跨期样本只用训练映射。类别权重和重采样只作用于训练。分类阈值在较早验证集最大F1后固定，后续不随双十二购买率提高而重新调整。",
        "相较旧实验，这次同时改变了开发截点和累计历史长度，所以不能把新旧分数差全部归因为促销。针对实际业务时期选择模型还需要新的时间外样本，不能以本次压力集再次反复筛选。",
        "",
        "## 冻结后的模型结果",
        "",
    ]
    lines += table(
        comparison,
        {
            "scenario": "评估窗口",
            "model": "模型",
            "auc": "AUC",
            "average_precision": "AP",
            "accuracy": "准确率",
            "recall": "召回率",
            "f1": "F1",
            "brier": "Brier",
        },
    )
    lines += [
        "![两个窗口的模型结果](plots/02_frozen_model_comparison.png)",
        "",
        "AUC用于排序质量，AP还受到窗口正例比例影响；阈值指标同时反映固定阈值在购买率变化后的效果。类别极不平衡时，准确率不能单独作为改进证据。跨期指标升高并不代表常态能力提升，降低也不能全部归因于促销。",
        "",
        "![分窗口ROC与PR](plots/03_roc_pr_by_window.png)",
        "",
        "## 分布变化的实际证据",
        "",
    ]
    lines += table(
        windows,
        {
            "window": "观察窗口",
            "mean_dau": "日均活跃用户",
            "mean_daily_buy_events": "日均购买行为",
            "mean_daily_buyer_share": "平均日买家比例",
        },
    )
    lines += [
        "![每日分布与活动日期](plots/01_daily_distribution.png)",
        "",
        "每日买家比例为当日有购买行为的用户占当日活跃用户比例，不等同于7天用户购买率，更不等同于用户商品对购买率。不同指标不可替代；日均统计避免直接用天数不同的总量比较。活动日期虚线仅为背景标注。",
        "",
        "![特征分布变化](plots/04_feature_drift.png)",
        "",
        "PSI使用较早训练分位箱、0.5计数平滑并独立报告缺失率，描述输入分布变化。固定7日历史后仍有变化的字段，可能反映人群与行为变化；PSI不是促销因果检验，也不使用单一阈值自动宣告失效。新类目回退率、购买最近性缺失和分数校准还需持续观察。",
        "",
        "## 尚未解决的限制与业务使用",
        "",
        "沿用的修订清洗文件，其异常阈值曾参考完整观察期。本次模型特征和编码严格遵守时间边界，但没有把历史内清洗阈值重估一并纳入，因此不宣称端到端完全没有回看信息。后续若做正式上线验收，应在历史期拟合清洗参数并冻结应用。",
        "新压力集为12万候选对，旧报告测试只有1.2万对；不能把二者当作同一测试集进行直接显著性比较。旧12月12日时期数据已在项目中查看，本次只保证它不参与此次调参，不能称为从未接触过的全新数据。",
        "现有观察期在活动启动后仅覆盖至12月18日，没有12月31日后回落阶段，无法完整评估促销结束后的稳定性。建议补齐活动前至少数周和活动后数周记录，按常态、预热、活动、回落分层报告，再进行滚动时间回测。",
        "本次预测目标仍为购买行为，缺订单支付和退款。原业务收益只是情景假设，不因这次AUC或购买率变化而自动上调。",
        "",
        "## 产物与复现",
        "",
        "入口为scripts/review_promotion_models.py。输出包含两个历史快照、样本划分、训练预处理、三种模型权重、36次Optuna记录、基础不平衡对比、冻结配置、分窗口预测、每日分布和特征漂移。用户商品明细仅保存在本地。",
        "",
    ]
    save_report(output / "双十二敏感性与传统模型重训报告", lines)
    summary = {
        "earlier_test_best_by_auc": early_best.model,
        "earlier_test_best_auc": float(early_best.auc),
        "stress_best_by_auc": late_best.model,
        "stress_best_auc": float(late_best.auc),
        "window_distribution": windows.to_dict(orient="records"),
        "activity_sources": [ACTIVITY_SOURCE, ACTIVITY_CONTEXT],
        "interpretation": "distribution sensitivity, not identified promotion causality",
    }
    (output / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("重训报告完成：%s", output)
