"""严格时间外购买概率、模型解释与业务模拟的中文报告。"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from behavior_cleaning.report_utils import save_report, table

ACTIVITY_SOURCE = "https://sw.wuhan.gov.cn/xwdt/mtbd/202512/t20251229_2703604.shtml"


def make_plots(output, comparison, prediction):
    output = Path(output)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )
    y = prediction.target.to_numpy()
    names = comparison[comparison.probability_state.eq("calibrated")].model.tolist()

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        f, t, _ = roc_curve(y, prediction[name + "_probability"])
        ax.plot(f, t, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Strict future-window ROC",
    )
    ax.legend(fontsize=8)
    finish(fig, "01_roc.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        precision, recall, _ = precision_recall_curve(
            y, prediction[name + "_probability"]
        )
        ax.plot(recall, precision, label=name)
    ax.axhline(y.mean(), ls="--", color="gray", label="Prevalence baseline")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Strict future-window precision-recall",
    )
    ax.legend(fontsize=8)
    finish(fig, "02_precision_recall.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, state in zip(axes, ["raw", "calibrated"]):
        for name in names:
            fraction, average = calibration_curve(
                y,
                prediction[name + ("_raw" if state == "raw" else "_probability")],
                n_bins=8,
                strategy="quantile",
            )
            ax.plot(average, fraction, marker="o", markersize=3, label=name)
        ax.plot([0, 0.12], [0, 0.12], "k--", alpha=0.4)
        ax.set(
            xlabel="Mean predicted probability",
            ylabel="Observed purchase fraction",
            title=state.capitalize() + " probabilities",
        )
        ax.set_xlim(0, 0.12)
        ax.set_ylim(0, 0.12)
    axes[1].legend(fontsize=7)
    finish(fig, "03_calibration.png")
    for metric, filename in [
        ("brier", "04_brier.png"),
        ("log_loss", "05_log_loss.png"),
    ]:
        matrix = comparison.pivot(
            index="model", columns="probability_state", values=metric
        ).reindex(names)
        fig, ax = plt.subplots(figsize=(9, 5))
        matrix.plot.bar(ax=ax, color=["#4B8AA8", "#B5C6CE"])
        ax.set(
            ylabel=metric, title=metric + " on final future window - lower is better"
        )
        ax.tick_params(axis="x", rotation=25)
        finish(fig, filename)
    fig, ax = plt.subplots(figsize=(9, 5))
    final = comparison[comparison.probability_state.eq("calibrated")].set_index("model")
    xpos = np.arange(len(names))
    ax.bar(
        xpos - 0.18, final.loc[names, "mean_probability"], 0.36, label="Predicted mean"
    )
    ax.bar(xpos + 0.18, final.loc[names, "observed_rate"], 0.36, label="Observed rate")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=25)
    ax.set(ylabel="Probability", title="Probability level versus actual purchase rate")
    ax.legend()
    finish(fig, "06_probability_level.png")
    budget = pd.read_csv(output / "ranking_budget_comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        rows = budget[budget.model.eq(name)]
        ax.plot(rows.candidate_fraction, rows.recall, marker="o", label=name)
    ax.set(
        xlabel="Fraction of candidate pairs selected",
        ylabel="Purchase-pair recall",
        title="Recall under equal candidate budgets",
    )
    ax.legend(fontsize=8)
    finish(fig, "07_budget_recall.png")
    for file, value, label, image in [
        (
            "xgboost_shap_importance.csv",
            "mean_absolute_calibrated_logit_shap",
            "feature",
            "08_xgboost_shap.png",
        ),
        (
            "din_group_shap_importance.csv",
            "mean_absolute_calibrated_logit_shap",
            "group",
            "09_din_shap.png",
        ),
    ]:
        data = pd.read_csv(output / "interpretability" / file).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(data[label], data[value], color="#4B8AA8")
        ax.set(
            xlabel="Mean absolute contribution in calibrated logit units",
            title="XGBoost TreeSHAP"
            if label == "feature"
            else "DIN exact six-group Shapley",
        )
        finish(fig, image)
    values = (
        pd.read_csv(output / "interpretability/logistic_coefficients.csv")
        .head(12)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        values.feature,
        values.calibrated_logit_coefficient,
        color=[
            "#4B8AA8" if v > 0 else "#B46C52"
            for v in values.calibrated_logit_coefficient
        ],
    )
    ax.set(
        xlabel="Calibrated logit coefficient per standardized input unit",
        title="Logistic regression feature coefficients",
    )
    finish(fig, "10_logistic_coefficients.png")
    daily = pd.read_csv(output / "daily_promotion_context.csv")
    dates = pd.to_datetime(daily.day)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, column in zip(axes, ["dau", "buy_events", "events"]):
        ax.plot(dates, daily[column], marker="o", markersize=3, color="#4B8AA8")
        ax.axvline(
            pd.Timestamp("2025-12-03"),
            ls="--",
            color="#B46C52",
            label="Reported campaign start",
        )
        ax.axvline(
            pd.Timestamp("2025-12-12"),
            ls="--",
            color="gray",
            label="Final prediction cutoff",
        )
        ax.set_ylabel(column)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Daily descriptive context - not model input")
    fig.autofmt_xdate()
    finish(fig, "11_promotion_context.png")


def write_reports(output):
    output = Path(output)
    summary = json.loads((output / "dataset/summary.json").read_text(encoding="utf-8"))
    checks = json.loads(
        (output / "explanation_checks.json").read_text(encoding="utf-8")
    )
    business = json.loads(
        (output / "business_simulation.json").read_text(encoding="utf-8")
    )
    all_comparison = pd.read_csv(output / "probability_model_comparison.csv")
    comparison = all_comparison[all_comparison.partition.eq("test")].copy()
    final = comparison[comparison.probability_state.eq("calibrated")].sort_values(
        "auc", ascending=False
    )
    raw = comparison[comparison.probability_state.eq("raw")].set_index("model")
    calibrated = final.set_index("model")
    leaders = {
        "auc": final.loc[final.auc.idxmax(), "model"],
        "ap": final.loc[final.average_precision.idxmax(), "model"],
        "brier": final.loc[final.brier.idxmin(), "model"],
    }
    prediction = pd.read_parquet(output / "test_predictions.parquet")
    make_plots(output, comparison, prediction)
    window = pd.DataFrame(summary["windows"])
    window["positive_rate"] = window.positives / window.sample_pairs
    train_rate = summary["splits"]["train"]["positive_rate"]
    cal_rate = summary["splits"]["calibration"]["positive_rate"]
    test_rate = summary["splits"]["test"]["positive_rate"]
    purchase_dates = pd.read_csv(output / "test_first_purchase_dates.csv")
    peak_date = purchase_dates.loc[purchase_dates.first_purchase_pairs.idxmax()]
    auc_interval = checks["stacking_vs_xgboost"]["user_cluster_bootstrap_95ci"]
    fusion_uncertainty = (
        "本轮区间包含0，尚不能确认Stacking相对XGBoost有稳定AUC提升。"
        if auc_interval[0] <= 0 <= auc_interval[1]
        else "本轮区间未包含0，但结论仍限于当前固定测试周，需更多时间窗口确认。"
    )
    baseline = []
    from behavior_cleaning.temporal_probability_models import probability_metrics

    for name, rate in [("training_prior", train_rate), ("calibration_prior", cal_rate)]:
        baseline.append(
            {
                "baseline": name,
                **probability_metrics(
                    prediction.target.to_numpy(), np.full(len(prediction), rate)
                ),
            }
        )
    baseline = pd.DataFrame(baseline)
    baseline.to_csv(output / "constant_probability_baselines.csv", index=False)
    lines = [
        "# 严格时间窗口购买概率预测报告",
        "",
        "本轮重新训练逻辑回归、XGBoost、LightGBM、LSTM、GRU、DIN、多头注意力DIN，并构建逻辑回归、XGBoost和DIN的时间留出Stacking。主要输出为用户商品对未来7天购买的0到1连续概率，保留真实0/1标签，以概率损失学习，不把标签强行改成连续回归值。",
        "这是给定最近交互候选的条件购买概率，不是用户购买任意商品的概率，也不是全商城候选召回或促销因果增量。行为4尚未核实为支付成功订单。",
        f"最终测试中，{leaders['auc']}的AUC最高，为{calibrated.loc[leaders['auc'],'auc']:.4f}；校准后模型中{leaders['ap']}的AP最高，为{calibrated.loc[leaders['ap'],'average_precision']:.4f}，{leaders['brier']}的Brier最低，为{calibrated.loc[leaders['brier'],'brier']:.6f}。不同指标反映不同用途，不能认定同一模型在所有方面领先。",
        f"最重要的限制是概率低估：校准后平均预测约{final.mean_probability.min():.2%}至{final.mean_probability.max():.2%}，而实际购买率为{summary['splits']['test']['positive_rate']:.2%}。当前结果可供排序和实验设计参考，绝对概率不宜直接用于承诺销量或收益。",
        "",
        "## 时间窗口与标签成熟",
        "源数据仅2025年11月18日至12月18日。为使训练、调参、融合校准和最终测试的7天标签完全不重叠，四阶段统一使用预测前3天历史。原任务的7天是未来标签长度；本轮放弃7:2:1随机划分，不宣称时间外各窗口购买率相同。",
        *table(
            window,
            {
                "window": "阶段",
                "history_start": "历史起点",
                "cutoff": "预测时点",
                "label_end_exclusive": "标签终点不含",
                "sample_pairs": "抽样候选对",
                "positives": "正例",
                "positive_rate": "购买率",
            },
        ),
        "训练为11月21日预测，调参为11月28日预测，融合与校准为12月5日预测，测试为12月12日预测。12月5日窗口按用户稳定哈希分成互斥的融合组与校准组，用户分组不使用购买标签。所有12月5日标签在12月12日预测前已成熟。",
        "同一用户或商品可以跨不同日期出现；同一快照内用户商品对唯一。未来购买只用于标签，历史候选及稳定哈希抽样不依赖未来标签。填补、缩放、目标编码和词表只在训练窗口拟合。",
        "每小时保留最多4件商品，对最后16个活跃小时进行池化与序列建模；不虚构小时内先后。DIN为小时桶适配版本，多头注意力为其扩展。26项输入未再次进行统计去冗余筛选。",
        "",
        "## 最终测试概率性能",
        "校准器在自然分布的独立校准组拟合一维logit的逻辑回归。校准器和辅助阈值均先冻结，才评价12月12日至18日最终窗口。AUC和AP评价排序；Brier、Log Loss越低越好，评价概率质量。ECE使用10个等宽概率箱，在低购买率时应结合分位校准图解读。",
        *table(
            final,
            {
                "model": "模型",
                "auc": "AUC",
                "average_precision": "AP",
                "brier": "Brier",
                "log_loss": "Log Loss",
                "ece_10_equal_width": "ECE",
                "mean_probability": "平均概率",
                "observed_rate": "实测购买率",
            },
        ),
        "![最终窗口ROC](plots/01_roc.png)",
        "![最终窗口PR](plots/02_precision_recall.png)",
        "ROC反映不同阈值下的排序区分度；在购买样本较少时，PR图和AP更能展示高得分候选中的购买集中度。AUC较高不保证概率数值准确。",
        "![校准前后可靠性](plots/03_calibration.png)",
        "图中每点为概率分位组的预测均值与实际购买比例。点在对角线上方表示低估购买概率，下方表示高估。为阅读主要低概率区域，坐标限制在0至0.12，超出点不展示；完整概率与指标均保留。",
        "![Brier对比](plots/04_brier.png)",
        "![LogLoss对比](plots/05_log_loss.png)",
        "![概率均值与实测比例](plots/06_probability_level.png)",
        "校准不保证跨期改善：最终窗口若出现校准期未出现的促销强度变化，提前冻结的校准器仍可能系统性低估或高估。不能用最终测试标签再调整校准器并将调整后成绩称为独立测试。",
        *table(
            baseline,
            {
                "baseline": "常数概率基线",
                "brier": "Brier",
                "log_loss": "Log Loss",
                "mean_probability": "固定概率",
            },
        ),
        "除排序外，应比较模型是否优于只输出较早购买率的常数概率基线。预测概率求和仅对应当前候选样本的预期购买对数，不是独立用户数、销售额或营销增量。",
        "",
        "## 辅助分类指标",
        "以下分类阈值在校准组按F1选择，不是本次主要交付。连续概率预测无需先截断为0或1。校准组拟合内指标不能作为泛化结果。",
        *table(
            final,
            {
                "model": "模型",
                "accuracy": "准确率",
                "precision": "精确率",
                "recall": "召回率",
                "f1": "F1",
                "threshold": "冻结阈值",
            },
        ),
        "## 不平衡方案与参数优化",
        "三种传统模型各比较不处理、SMOTE、随机欠采样和类别权重，仅作用训练窗口。调参窗口保持自然分布，以AUC选择方案并分别执行Optuna搜索；序列模型对比两组嵌入、隐藏层和Dropout配置，验证AUC早停、学习率衰减并恢复最佳权重。最终自然分布校准用于纠正权重或采样引起的概率尺度偏移。",
        *table(
            pd.read_csv(output / "imbalance_validation_comparison.csv"),
            {
                "model": "模型",
                "strategy": "方案",
                "training_rows": "拟合行数",
                "auc": "调参AUC",
                "brier": "未经校准Brier",
                "recall": "0.5阈值召回",
            },
        ),
        "逻辑回归适合低成本、可解释的排序基线；树模型能处理非线性交互，本轮XGBoost的AP高于各序列单模型。LSTM和GRU适合有较充分序列的场景，但当前商品未知率高、历史仅3天，复杂网络没有稳定超过表格模型。DIN及注意力扩展需要更多可泛化的商品属性与跨期行为支撑，不能仅通过增加网络规模期待提升。",
        f"概率校准明显纠正了类别权重造成的尺度偏移，例如逻辑回归Brier由{raw.loc['LogisticRegression','brier']:.6f}降至{calibrated.loc['LogisticRegression','brier']:.6f}；但XGBoost的Log Loss由{raw.loc['XGBoost','log_loss']:.6f}变为{calibrated.loc['XGBoost','log_loss']:.6f}，未同步改善。应分别判断排序与概率质量，并通过新的时间窗口验证。",
        "## 促销影响与适用边界",
        f"训练、校准、最终测试购买率分别为{train_rate:.4%}、{cal_rate:.4%}、{test_rate:.4%}。应把促销变化作为跨期分布漂移解释，不能全部归因于模型能力变化。",
        f"武汉市商务局转载报道记载2025年12月3日启动抖音商城年终狂欢季。训练标签在此前，调参标签跨越启动，融合校准和最终测试覆盖活动期。[活动背景]({ACTIVITY_SOURCE})。日期标记是事后分析背景，不代表样本用户确实曝光或参加活动；没有把事后新闻当作当时可用的预测输入。",
        "![每日活跃和购买行为](plots/11_promotion_context.png)",
        *table(
            pd.read_csv(output / "test_first_purchase_dates.csv"),
            {
                "day": "测试正例首次购买日",
                "first_purchase_pairs": "首次购买对数",
                "share_of_positive_pairs": "占全部正例比例",
            },
        ),
        "首次购买日分布仅帮助观察最终标签是否集中于活动节点，不代表删除活动当日后仍能沿用同一7天目标进行公平比较，也不能识别促销带来的因果增量。",
        f"本次{summary['splits']['test']['positives']}个测试购买正例中，{int(peak_date.first_purchase_pairs)}个首次购买发生在{peak_date.day}，占{peak_date.share_of_positive_pairs:.2%}。标签的日集中度提示需要补充多场活动、活动前后更长时期的数据，不能把当前测试成绩直接推广到普通一周。该分布不能证明促销是唯一原因。",
        "当前仅一个最终测试周，且没有足够的活动前和活动后多周数据，不能单独验证常态泛化或计算促销因果效应。需要补充更长时期与活动曝光、优惠和订单字段。已清洗数据的异常阈值曾参考全期，本轮未重做历史内清洗拟合，仍有回看限制。",
        "本轮窗口与此前项目研究期重叠，原始日期已被查看；只保证未使用此次测试标签选择本轮模型、融合参数、概率校准或阈值。深度模型商品词表仅用训练历史拟合，最终商品未知率高，应结合长尾和冷启动诊断解读。",
        "",
        "## 复用与输出",
        "入口 scripts/train_temporal_probability.py；数据、权重、校准器、连续概率预测、解释和业务模拟均存放本次独立目录，原实验结果保留。all_window_predictions.parquet中的后期校准概率不可冒充早期时点可获得的预测；正式外部评估只使用test_predictions.parquet及报告最终测试行。",
    ]
    save_report(output / "时间窗口购买概率模型评估报告", lines)
    detail = [
        "# 模型融合与概率可解释性报告",
        "",
        "Stacking使用已在更早训练窗口拟合、调参窗口选定的逻辑回归、XGBoost与DIN。三个原始概率转logit后，在12月5日融合组拟合缩放与二层逻辑回归；同日互斥用户校准组只拟合概率校准和辅助阈值。最终12月12日标签不参与拟合。",
        "这是时间留出式Stacking，不声称完成训练窗口内部多折时间OOF，也不将三者原始概率简单平均。二层权重只表示条件线性组合，不能解释为业务贡献百分比。",
        *table(
            final,
            {
                "model": "模型",
                "auc": "AUC",
                "average_precision": "AP",
                "brier": "Brier",
                "log_loss": "Log Loss",
            },
        ),
        f"Stacking相对XGBoost的最终AUC差为{checks['stacking_vs_xgboost']['delta_auc']:.6f}，按用户整组重采样得到95%区间{checks['stacking_vs_xgboost']['user_cluster_bootstrap_95ci']}。该区间只描述当前固定周内的不确定性，不替代多个独立时间窗口，也不证明融合必然提高效果。",
        fusion_uncertainty + "融合应分别比较排序和概率误差，不能以单一小幅分差宣称全面胜出。",
        "![相同候选预算下召回](plots/07_budget_recall.png)",
        "预算图比较相同候选对数量下的召回，不把不同模型各自阈值下的触达量当作相同资源投入。",
        "## 逻辑回归解释",
        "![逻辑回归系数](plots/10_logistic_coefficients.png)",
        "系数对应已标准化输入的一单位变化，对校准后logit的关联影响。计数输入先经过log1p，不能解释为原始行为增加一次的因果效果；共线特征会影响系数稳定性。",
        "## XGBoost解释",
        "![TreeSHAP排序](plots/08_xgboost_shap.png)",
        f"使用原生精确TreeSHAP并检验原始margin加和，最大误差{checks['xgboost_raw_margin_additivity_max_error']:.8f}。Platt校准在logit空间是仿射变换，贡献按校准斜率缩放、基线加入截距，可解释最终概率的logit。贡献不是购买率百分点。",
        "本轮XGBoost最依赖指定商品的加购次数、近期交互次数，以及该商品占用户交互的比例，说明商品级意向比仅看用户整体活跃更有区分力。运营上可先验证这类意向商品的推荐承接与信息完整性，不把高SHAP值直接解释为增加某行为就能带来购买。",
        "## DIN解释",
        "![DIN六组Shapley](plots/09_din_shap.png)",
        f"随机抽取{checks['din_samples']}个测试样本，使用{checks['background_samples']}个训练背景，完整枚举六组输入的64种联盟。六组为用户、商品、用户商品交互、类目编码、候选嵌入和历史序列。校准后logit加和最大误差{checks['din_calibrated_logit_additivity_max_error']:.8f}。这是分组Shapley而不是注意力权重，也不逐一解释每个嵌入坐标；背景替换可能产生不自然的混合组合。",
        "DIN分组贡献仍以用户商品交互为主，其后为用户统计，候选嵌入和历史序列也有贡献。当前证据支持数值交互与序列共同使用，不支持仅凭加入注意力机制就认定更准确。",
        "## 冷启动长尾与错误样本",
        *table(
            pd.read_csv(output / "error_group_metrics.csv"),
            {
                "model": "模型",
                "group": "样本分组",
                "rows": "数量",
                "positives": "正例",
                "auc": "AUC",
                "brier": "Brier",
                "mean_probability": "平均概率",
            },
        ),
        f"长尾定义为训练候选商品历史行为量25%分位数阈值{checks['long_tail_history_event_threshold']:.1f}以下。训练未见用户和商品是分布外代理，不能等同真实零历史冷启动。本轮候选要求最近有交互，真正零历史样本为0，故不能声称已经验证真正冷启动效果。",
        *table(
            pd.read_csv(output / "stacking_error_profile.csv"),
            {
                "error_type": "Stacking样本类型",
                "pair_cart": "加购中位数",
                "pair_recent_events": "交互中位数",
                "last_time_days_ago": "交互间隔中位天数",
                "item_events": "商品热度中位数",
                "probability": "概率中位数",
            },
        ),
        "低热度、训练未见商品的嵌入不足，建议优先补充可泛化的商品属性和类目语义，并增加历史词表覆盖。对历史高活跃但预测未购买或漏判的样本，应分开检查商品意向与整体活跃度，避免把访问频繁直接等同于指定商品购买。",
    ]
    save_report(output / "时间融合与概率解释分析报告", detail)
    b = [
        "# 概率模型业务落地模拟方案",
        "",
        "本方案仅生成模拟分组和预算情景，没有向用户发送消息、发券、修改推荐或实施真实实验。概率高只代表观察性购买倾向，不代表被营销后更容易产生增量。",
        "## 分组与实验指标",
        f"最终候选涉及{business['cohort_users']}名用户，按更早校准组用户最高候选得分80%分位数阈值{business['eligibility_score_threshold']:.6f}识别{business['eligible_users']}名符合资格的用户。该最高分只是排序规则，不能当成用户购买任意商品的概率。候选只是抽样商品，真实上线前应统一推荐覆盖和可触达条件。",
        "正式实验在处理前按用户稳定随机分配50%对照和50%处理，同一用户所有商品使用同一分组。对照保持现有策略，处理使用概率排序推荐或意向提醒；补贴另设实验，避免无法区分推荐和优惠的作用。主要指标为预先确定资格用户的7天有效支付用户转化率ITT；同时监测退款、净收益、客诉和退订。",
        f"历史购买行为代理基线为{business['proxy_baseline_rate']:.2%}，按合格用户是否购买任意商品计算，既不是已支付订单基线，也不是用户商品对的{test_rate:.2%}购买率。按双侧alpha=0.05、功效80%的两比例近似规划样本量，正式实验应替换真实支付基线。",
        *table(
            pd.DataFrame(business["power_scenarios"]),
            {"absolute_lift": "假设绝对提升", "required_users_per_arm": "每组所需用户"},
        ),
        "计划招募14天、每用户观察7天，主要结果最早21天读取；最后一笔购买还需等待30天退款成熟，最终收益读取可能延至约51天。当前小样本不能证明小幅业务提升，历史分组平衡也不能代替真实干预效果。",
        "## 精准营销与推荐建议",
        "优先小流量验证高意向候选的排序、同类替代与库存价格承接，保留随机对照；对冷启动与长尾商品设置类目和内容属性补充、探索流量与频控。概率阈值须结合可服务人数、成本和校准漂移，不直接按高分发放高额优惠。促销期与日常期分别招募或分层随机并分别报告，避免把大促自然购买算成策略增量。",
        "## 业务收益情景",
        "假设1万名合格用户触达20%，每个增量买家1单，客单价150元、退款10%、券前毛利25%。无券触达成本每人0.05元；有券触达0.2元、券5元、核销30%。这些参数均非现有行为表测得。券成本覆盖全部自然及增量购买者，不能仅计增量买家。",
        *table(
            pd.read_csv(output / "assumed_business_scenarios.csv"),
            {
                "assumed_absolute_lift": "假设绝对增量",
                "coupon": "发券",
                "reached_users": "触达",
                "incremental_buyers": "增量买家",
                "assumed_net_gmv": "假设净GMV增量",
                "coupon_cost": "券成本",
                "assumed_incremental_contribution": "假设贡献增量",
            },
        ),
        "上述金额是条件测算而非模型预测的真实利润；无订单金额、退款、实际优惠和对照实验时，不能承诺业务提升。最终窗口已有明显概率漂移，更应先验证支付口径和校准，再决定营销资源投入。",
    ]
    save_report(output / "概率预测业务落地模拟方案", b)
