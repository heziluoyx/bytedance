"""融合、解释和业务模拟的离线中文报告与图表。"""

import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

FEATURE_CN = {
    "pair_pv": "该商品历史浏览次数",
    "pair_fav": "该商品历史收藏次数",
    "pair_cart": "该商品历史加购次数",
    "pair_buys": "该商品历史购买次数",
    "pair_recent_events": "近七天交互次数",
    "pair_days": "该商品交互天数",
    "item_events": "商品历史总交互量",
    "item_buys": "商品历史购买次数",
    "item_users": "商品历史交互人数",
    "user_events": "用户历史总交互量",
    "user_items": "用户历史交互商品数",
    "user_days": "用户历史活跃天数",
    "user_pv": "用户历史浏览次数",
    "user_fav": "用户历史收藏次数",
    "user_cart": "用户历史加购次数",
    "user_buys": "用户历史购买次数",
    "user_last_days_ago": "距用户最近行为天数",
    "user_last_buy_days_ago": "距用户最近购买天数",
    "pair_buy_share": "用户对该商品的购买占比",
    "item_buy_share": "商品历史购买行为占比",
    "user_buy_share": "用户历史购买行为占比",
    "pair_user_event_share": "该商品占用户交互的比例",
    "first_time_days_ago": "距该组合首次交互天数",
    "last_time_days_ago": "距该组合最近交互天数",
    "last_buy_days_ago": "距该组合最近购买天数",
    "category_target_encoding": "类目购买目标编码",
}
GROUP_CN = {
    "User statistics": "用户统计",
    "Item statistics": "商品统计",
    "Pair interactions": "用户商品交互",
    "Category encoding": "类目目标编码",
    "Candidate embedding": "候选商品嵌入",
    "History sequence": "历史序列与匹配标记",
}


def save_report(path, lines):
    """输出同源Markdown和本地可直接打开的HTML。"""
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    blocks, in_table = [], False
    for line in lines:
        if line.startswith("|"):
            if re.fullmatch(r"\|[-|: ]+\|", line):
                continue
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            tag = "td" if in_table else "th"
            if not in_table:
                blocks.append("<table>")
                in_table = True
            blocks.append(
                "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            )
            continue
        if in_table:
            blocks.append("</table>")
            in_table = False
        if line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            blocks.append(
                f'<figure><img src="{html.escape(match[2])}"><figcaption>{html.escape(match[1])}</figcaption></figure>'
            )
        elif line.startswith("## "):
            blocks.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line.startswith("# "):
            blocks.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line:
            content = re.sub(
                r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(line)
            )
            content = re.sub(
                r"\[(.*?)\]\((https://[^ )]+)\)", r'<a href="\2">\1</a>', content
            )
            blocks.append("<p>" + content + "</p>")
    if in_table:
        blocks.append("</table>")
    path.with_suffix(".html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'
        + html.escape(path.name)
        + '</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.9 "Microsoft YaHei",sans-serif;color:#1d2730}h1,h2{color:#172f43}table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}td,th{border:1px solid #ddd;padding:8px}th{background:#244b63;color:white}tr:nth-child(even){background:#f1f5f7}img{max-width:100%;height:auto}figure{margin:32px 0}figcaption{color:#51626d}a{color:#146095}</style><main>'
        + "\n".join(blocks)
        + "</main></html>",
        encoding="utf-8",
    )


def create_plots(run, output):
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 130})
    results = pd.read_csv(output / "stacking/model_comparison.csv")
    pred = pd.read_parquet(output / "stacking/test_predictions.parquet")
    lr = pd.read_csv(output / "interpretability/logistic_coefficients.csv")
    xgb = pd.read_csv(output / "interpretability/xgboost_shap_importance.csv")
    din = pd.read_csv(output / "interpretability/din_group_shap_importance.csv")

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["auc", "average_precision"]):
        ax.barh(
            results.model,
            results[metric],
            color=["#d9822b" if x == "Stacking" else "#326b8c" for x in results.model],
        )
        ax.set(title="Test " + metric)
        ax.grid(axis="x", alpha=0.2)
    finish(fig, "01_model_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ["LogisticRegression", "XGBoost", "LightGBM", "DIN", "Stacking"]:
        fpr, tpr, _ = roc_curve(pred.target, pred[name])
        precision, recall, _ = precision_recall_curve(pred.target, pred[name])
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].axhline(pred.target.mean(), linestyle="--", color="grey")
    axes[0].set(
        xlabel="False positive rate", ylabel="True positive rate", title="Test ROC"
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Test PR")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    finish(fig, "02_roc_pr.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    shown = lr.head(15).iloc[::-1]
    ax.barh(
        shown.feature,
        shown.coefficient_per_standard_deviation,
        color=[
            "#326b8c" if value > 0 else "#c26348"
            for value in shown.coefficient_per_standard_deviation
        ],
    )
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(
        xlabel="Log-odds coefficient per SD of transformed feature",
        title="Logistic regression coefficients",
    )
    finish(fig, "03_logistic_coefficients.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    shown = xgb.head(15).iloc[::-1]
    ax.barh(shown.feature, shown.mean_abs_shap, color="#326b8c")
    ax.set(xlabel="Mean absolute SHAP (raw margin)", title="XGBoost exact TreeSHAP")
    finish(fig, "04_xgboost_importance.png")
    arrays = np.load(run / "dataset/dataset.npz")
    summary = json.loads(
        (run / "dataset/dataset_summary.json").read_text(encoding="utf-8")
    )
    shap = pd.read_parquet(output / "interpretability/xgboost_shap_values.parquet")
    rng = np.random.RandomState(42)
    selection = rng.choice(len(shap), min(2000, len(shap)), replace=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    shown = list(reversed(xgb.feature.head(10).tolist()))
    for position, name in enumerate(shown):
        raw = arrays["x"][
            shap.sample_row.to_numpy()[selection], summary["features"].index(name)
        ]
        rank = pd.Series(raw).rank(pct=True).to_numpy()
        points = ax.scatter(
            shap[name].to_numpy()[selection],
            position + rng.uniform(-0.22, 0.22, len(selection)),
            c=rank,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            alpha=0.35,
            s=7,
        )
    ax.set(
        yticks=np.arange(len(shown)),
        yticklabels=shown,
        xlabel="SHAP contribution to raw margin",
        title="Feature direction and heterogeneous effects",
    )
    ax.axvline(0, color="grey", linewidth=0.8)
    fig.colorbar(points, ax=ax, label="Within-sample feature percentile (red=high)")
    finish(fig, "05_xgboost_beeswarm.png")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    shown = din.iloc[::-1]
    positions = np.arange(len(shown))
    ax.barh(positions, shown.mean_abs_shap, color="#326b8c", label="8-row background")
    ax.scatter(
        shown.background_first_half,
        positions,
        marker="x",
        color="#e2a03e",
        label="First four references",
    )
    ax.scatter(
        shown.background_second_half,
        positions,
        marker="o",
        facecolors="none",
        edgecolors="#cc534e",
        label="Last four references",
    )
    ax.set(
        yticks=positions,
        yticklabels=shown.group,
        xlabel="Mean absolute grouped SHAP (logit)",
        title="DIN six-group exact Shapley; 128 random test pairs",
    )
    ax.legend(fontsize=8)
    finish(fig, "06_din_group_shap.png")
    sub = pd.read_csv(output / "diagnostics/subgroup_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ["XGBoost", "Stacking"]):
        row = sub[(sub.model == name) & (sub.segment == "all")].iloc[0]
        matrix = np.array(
            [
                [row.true_negative, row.false_positive],
                [row.false_negative, row.true_positive],
            ],
            dtype=int,
        )
        ax.imshow(np.log1p(matrix), cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No buy", "Buy"],
            yticklabels=["No buy", "Buy"],
            xlabel="Predicted",
            ylabel="Observed",
            title=name + " confusion counts",
        )
    finish(fig, "07_error_counts.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    chosen_segments = [
        "all",
        "user_unseen_in_train_sample",
        "long_tail_item",
        "non_long_tail_item",
    ]
    pivot = sub[sub.segment.isin(chosen_segments)].pivot(
        index="segment", columns="model", values="recall"
    )
    pivot.plot.bar(ax=ax)
    positive_counts = sub[sub.model == "Stacking"].set_index("segment").positives
    ax.set_xticklabels(
        [name + f"\npositive n={int(positive_counts[name])}" for name in pivot.index]
    )
    ax.set(
        title="Recall by subgroup (read positive counts in report)",
        ylabel="Recall",
        xlabel="",
    )
    ax.tick_params(axis="x", rotation=12)
    ax.legend(fontsize=8)
    finish(fig, "08_segment_recall.png")
    ranking = pd.read_csv(output / "diagnostics/ranking_budget_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, group in ranking.groupby("model"):
        axes[0].plot(group.fraction, group.precision_at_k, marker=".", label=name)
        axes[1].plot(group.fraction, group.recall_at_k, marker=".", label=name)
    for ax, metric in zip(axes, ["Precision", "Recall"]):
        ax.set(
            xlabel="Fraction of candidate pairs selected",
            ylabel=metric,
            title="Fixed candidate budget: " + metric,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    finish(fig, "09_ranking_budget.png")
    scenarios = pd.read_csv(output / "business/assumed_business_benefits.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, group in scenarios.groupby("policy"):
        ax.plot(
            group.assumed_absolute_lift * 100,
            group.expected_incremental_contribution,
            marker="o",
            label=name,
        )
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(
        xlabel="ASSUMED purchase lift among reached users (percentage points)",
        ylabel="ASSUMED incremental contribution (CNY)",
        title="Scenario only: 10,000 eligible users, 20% reached; not measured uplift",
    )
    ax.legend(fontsize=8)
    finish(fig, "10_assumed_net_benefit.png")


def table(frame, columns):
    """列名到展示名的简单映射。"""
    result = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(
                ("是" if value else "否")
                if isinstance(value, (bool, np.bool_))
                else "不可估计"
                if pd.isna(value)
                else f"{value:.4f}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        result.append("| " + " | ".join(values) + " |")
    return result + [""]


def write_followup_reports(run: Path, output: Path, diagnostic: dict, business: dict):
    create_plots(run, output)
    dataset = json.loads(
        (run / "dataset/dataset_summary.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(output / "stacking/model_comparison.csv")
    stack = comparison.set_index("model").loc["Stacking"]
    frozen = json.loads(
        (output / "stacking/frozen_configuration.json").read_text(encoding="utf-8")
    )
    lr = pd.read_csv(output / "interpretability/logistic_coefficients.csv")
    xgb = pd.read_csv(output / "interpretability/xgboost_shap_importance.csv")
    din = pd.read_csv(output / "interpretability/din_group_shap_importance.csv")
    sub = pd.read_csv(output / "diagnostics/subgroup_metrics.csv")
    rank = pd.read_csv(output / "diagnostics/ranking_budget_comparison.csv")
    coefficients = "、".join(
        f"{name}：{value:+.3f}"
        for name, value in zip(frozen["base_models"], frozen["meta_coefficients"])
    )
    core = "、".join(FEATURE_CN.get(name, name) for name in xgb.feature.head(5))
    main_groups = "、".join(GROUP_CN.get(name, name) for name in din.group.head(3))
    difference = diagnostic["paired_auc_difference_stack_minus_reference"]["XGBoost"]
    conclusion = (
        "本轮融合的测试AUC高于XGBoost"
        if difference["point_difference"] > 0
        else "本轮融合未提高XGBoost的测试AUC"
    )
    confidence = (
        "区间覆盖0，尚不能确认AUC有稳定提升。"
        if difference["lower_95"] <= 0 <= difference["upper_95"]
        else "该探索性区间不包含0，但仍需新的时间外数据验证。"
    )
    lines = [
        "# 模型融合与可解释性分析报告",
        "",
        "## 核心结论",
        "",
        f"本报告预测单位为用户—商品对，预测时点为{dataset['cutoff']}，未来七天标签窗口截至{dataset['label_end_exclusive']}（不含结束时点）。候选来自此前七天已交互商品，抽样120,000对，按7:2:1分层划分；目标是购买行为标记，不是已核验支付订单。",
        f"Stacking融合逻辑回归、XGBoost和DIN后，留出集AUC为{stack.auc:.4f}，AP为{stack.average_precision:.4f}，召回率为{stack.recall:.2%}，F1为{stack.f1:.4f}。{conclusion}，不能把模型融合视为必然提升。",
        f"按用户成簇进行500次配对bootstrap，Stacking相对XGBoost的AUC差为{difference['point_difference']:+.4f}，探索性95%区间为[{difference['lower_95']:+.5f}, {difference['upper_95']:+.5f}]。{confidence}",
        "本轮沿用已经查看过结果的12,000条留出样本，其中只有77个购买正例。融合结构、二层参数和阈值未根据本次测试表现继续调整；但这些结果属于原留出集上的扩展评估，不能当作全新、未接触测试集的独立验收。",
        "",
        "## Stacking实施与隔离",
        "",
        f"将原84,000条训练样本分为三折，每次用56,000条训练基础模型，为另外28,000条产生折外预测。填补、数值缩放和有监督类目编码均在当前拟合折重新学习；类目编码对拟合折内部再做五折折外处理。三个基础模型各重训三次，DIN每折固定训练{frozen['DIN_fixed_epochs']}轮，不依据当前留出折标签挑选轮数或早停。",
        "商品和类目使用原训练期拟合的固定无监督词表，未按外层折重选热门词表；这降低额外计算，也意味着本轮采用训练期共享词表的随机分层评估。基模型超参来自上一轮验证集选择，不属于完整嵌套交叉验证。",
        "二层采用固定C=1的逻辑回归，以三种基础模型概率的logit为输入，在OOF预测上拟合缩放和系数，不使用基模型对自身训练样本的拟合概率。最终推断复用已在完整训练集拟合的三个原基模型，再接二层模型。分类阈值由原验证集最大F1决定。",
        f"二层标准化输入系数为{coefficients}。这些系数用于组合三个相关模型的输出，受尺度、相关性和正则影响，不是三个模型的业务贡献百分比。OOF折模型与最终全训练模型的概率分布也可能不同，需在未来时间窗口检查融合稳定性。",
        "",
    ]
    lines += table(
        comparison,
        {
            "model": "模型",
            "validation_auc": "验证AUC",
            "auc": "测试AUC",
            "average_precision": "AP",
            "accuracy": "准确率",
            "recall": "召回率",
            "f1": "F1",
        },
    )
    lines += [
        "![模型综合比较](plots/01_model_comparison.png)",
        "",
        "图中保留上一轮七个模型作为参照，避免只与较弱基线比较。AUC和AP刻画不同侧面的排序质量，模型在全体样本上的区分能力较好，不代表少量触达名单一定具有更高购买命中率。",
        "",
        "![ROC与PR曲线](plots/02_roc_pr.png)",
        "",
        "正例稀少时，PR曲线比单看准确率更能反映误触达问题。用户购买倾向评分只是预测结果，尚未测量推荐或营销能额外带来多少购买。",
        "",
        "## 逻辑回归系数解释",
        "",
        "输入数值经过log1p与标准化，因此系数表示变换后特征增加一个训练标准差时的对数胜算变化，其他输入保持不变。正系数提高预测购买倾向，负系数降低预测倾向；不是原始浏览次数每增加一次的影响。类目目标编码使用单独缩放，其系数同样按标准差解释。",
        "模型使用类别权重，系数适合解释该模型的判别方向，不能把未经校准的预测概率作为真实支付概率。相关特征之间可能分担或抵消系数，不能仅凭符号推断业务因果。",
        "",
    ]
    top_lr = lr.head(10).copy()
    top_lr["name"] = top_lr.feature.map(FEATURE_CN).fillna(top_lr.feature)
    lines += table(
        top_lr,
        {
            "name": "特征",
            "coefficient_per_standard_deviation": "标准化系数",
            "odds_multiplier": "对应胜算倍数",
        },
    )
    lines += [
        "本次距该组合最近交互天数的系数为负，而加购次数、用户历史购买次数的系数为正，说明近期商品意向和已有购买习惯在该线性模型中提供了不同层面的识别信息。该结果支持在候选排序中保留时效与加购信号，不能据此推断人为增加浏览或加购次数会带来相同比例的成交。",
        "",
    ]
    lines += [
        "![逻辑回归系数](plots/03_logistic_coefficients.png)",
        "",
        "## XGBoost的SHAP解释",
        "",
        f"采用XGBoost原生精确TreeSHAP，为全部12,000条测试样本计算26项特征贡献，并验证贡献加基准值等于原始margin。平均绝对贡献较高的前五项为：{core}。这些特征是当前模型预测时较依赖的信息，不是经过随机实验确认的购买驱动因素。",
        "SHAP数值处于模型原始margin空间，不能直接解读为转化率提高多少个百分点。重要性条形图表示影响幅度；蜂群图进一步展示取值高低与预测增减之间的联系。同一个特征的贡献可能随用户和商品状态变化。",
        "",
    ]
    top_xgb = xgb.head(10).copy()
    top_xgb["name"] = top_xgb.feature.map(FEATURE_CN).fillna(top_xgb.feature)
    lines += table(
        top_xgb,
        {"name": "特征", "mean_abs_shap": "平均绝对SHAP", "mean_signed_shap": "平均有符号贡献"},
    )
    lines += [
        "![XGBoost重要性](plots/04_xgboost_importance.png)",
        "",
        "![XGBoost贡献方向](plots/05_xgboost_beeswarm.png)",
        "",
        "## DIN的分组SHAP解释",
        "",
        f"将DIN全部输入拆为六个互不重叠的信息组，对128条均匀抽取的测试样本穷举64种联盟。背景为固定随机抽取的8条训练样本，使用模型logit输出计算精确分组Shapley值；不是把注意力权重冒充SHAP。当前贡献较高的三组为：{main_groups}。",
        "这是六组信息在指定背景下的贡献，不是每一个嵌入维度的独立解释，也不等同于对26个数值特征逐项展开。历史序列组包括小时内商品、类目、行为分布、时间、掩码及候选匹配标记。图中额外比较前后各4条背景的结果，提示小背景选择对解释可能产生的影响。",
        f"原商品词表仅覆盖训练期热门子集，测试候选商品嵌入未知率为{dataset['splits']['test']['target_item_oov_rate']:.2%}。类目、数值统计和历史匹配仍可使用，但不能据此认定已学到全量商品的独立语义；扩展商品属性和嵌入覆盖应作为后续验证重点。",
        "遮蔽时会组合来自不同样本的数值、候选商品和历史块，可能形成现实中少见的组合；相关特征和匹配标记使这种解释尤其依赖背景假设。结果用于诊断模型输入依赖，不能解释为更改某一项运营动作的因果收益。",
        "",
    ]
    translated = din.copy()
    translated["name"] = translated.group.map(GROUP_CN)
    lines += table(
        translated,
        {
            "name": "输入组",
            "mean_abs_shap": "平均绝对SHAP",
            "background_first_half": "前4条背景",
            "background_second_half": "后4条背景",
        },
    )
    lines += [
        "蜂群图显示，加购次数和近七天交互量较高的样本较多出现在正贡献一侧，而距离最近交互越久，贡献更常偏负。这与近期兴趣衰减的业务判断一致。运营上宜在用户意向仍较新时提供准确的商品与履约信息，同时保留对低频、长决策周期用户的观察，不能把低交互量直接等同于无购买需求。",
        "",
    ]
    lines += [
        "![DIN分组SHAP](plots/06_din_group_shap.png)",
        "",
        "## 误判、冷启动与长尾诊断",
        "",
        f"长尾定义为商品在历史窗口的交互次数不超过训练候选去重商品的25%分位，本次阈值为{diagnostic['long_tail_event_threshold']:g}次。该定义依据预测前历史和训练候选确定，不按测试标签决定。未见用户指未出现在监督训练样本中的用户；他们仍有历史行为，不能称为真正零历史新用户。本测试集中零历史用户样本数为{diagnostic['zero_history_user_rows']}。",
        "",
    ]
    selected = sub[
        (sub.model == "Stacking")
        & sub.segment.isin(
            [
                "all",
                "user_unseen_in_train_sample",
                "long_tail_item",
                "non_long_tail_item",
                "item_unseen_in_train_sample",
            ]
        )
    ]
    selected = selected.copy()
    selected["segment"] = selected.segment.map(
        {
            "all": "全部候选",
            "user_unseen_in_train_sample": "训练样本未见用户",
            "long_tail_item": "长尾商品",
            "non_long_tail_item": "非长尾商品",
            "item_unseen_in_train_sample": "训练样本未见商品",
        }
    )
    lines += table(
        selected,
        {
            "segment": "分组",
            "rows": "样本数",
            "positives": "正例数",
            "auc": "AUC",
            "precision": "精确率",
            "recall": "召回率",
            "false_positive": "误报",
            "false_negative": "漏报",
        },
    )
    tail = sub[(sub.model == "Stacking") & (sub.segment == "long_tail_item")].iloc[0]
    cold = sub[
        (sub.model == "Stacking") & (sub.segment == "user_unseen_in_train_sample")
    ].iloc[0]
    lines += [
        f"长尾组共{int(tail.rows):,}条候选对、{int(tail.positives)}个购买正例，在当前阈值下没有候选入选，正例全部漏报，提示统一阈值可能压低长尾覆盖。未见训练用户组共{int(cold.rows)}条候选对，仅{int(cold.positives)}个正例，因此即使个别模型该组AUC或召回看起来很高，也不足以代表新用户总体能力。后续应在新的验证窗口评估分层阈值、类目兜底及探索预算，而不是直接在本测试集上降低阈值。",
        "",
    ]
    profiles = pd.read_csv(output / "diagnostics/error_feature_profiles.csv")
    pivot = profiles[profiles.model == "Stacking"].pivot(
        index="feature", columns="error_type", values="median"
    )
    selected_features = [
        "pair_pv",
        "pair_cart",
        "user_events",
        "user_buys",
        "item_events",
        "last_time_days_ago",
    ]
    profile_table = pivot.loc[selected_features].reset_index()
    profile_table["name"] = profile_table.feature.map(FEATURE_CN)
    lines += table(
        profile_table,
        {"name": "特征中位数", "TP": "命中购买", "FP": "误报购买", "FN": "漏报购买", "TN": "正确未购买"},
    )
    lines += [
        f"漏报购买样本的商品历史交互量中位数为{pivot.loc['item_events','FN']:g}，低于命中购买样本的{pivot.loc['item_events','TP']:g}；距该组合最近交互中位数约{pivot.loc['last_time_days_ago','FN']*24:.1f}小时，命中样本约{pivot.loc['last_time_days_ago','TP']*24:.1f}小时。漏报用户自身历史购买次数并不更少，说明仅看用户是否活跃或是否有购买习惯，仍不足以判断会购买哪件商品。",
        f"误报与命中样本的浏览次数中位数均为{pivot.loc['pair_pv','FP']:g}，但误报样本依然没有在七天内购买。近期浏览或加购提供意向线索，却不能完整表达预算、规格、比较决策和交易阻力；下一步应补充商品属性与真实订单状态验证这些假设。",
        "",
    ]
    all_stack = sub[(sub.model == "Stacking") & (sub.segment == "all")].iloc[0]
    lines += [
        f"按固定阈值，融合模型命中{int(all_stack.true_positive)}个购买正例，误报{int(all_stack.false_positive)}条，漏报{int(all_stack.false_negative)}条。误报表示模型预测会购买、窗口内未观测到购买；它不一定代表用户永远没有需求。漏报可能来自弱历史信号、未知商品或短期意向变化，但现有数据无法逐条确认真实原因。",
        "",
        "![误判数量](plots/07_error_counts.png)",
        "",
        "![分群召回](plots/08_segment_recall.png)",
        "",
        "分群表现必须结合正例数量阅读。没有购买正例的分组不计算AUC或AP；正例很少时，召回率容易因一两个样本显著改变，不据此断言某类人群一定无法建模。对于真正的新用户，应采用类目热度、上下文和规则兜底；长尾商品可优先补充内容、属性及类目表示，并设置少量探索流量，防止仅凭历史热门度循环放大头部商品。",
        "",
        "本地误判案例和各类样本的特征均值、中位数已经保存，可核查近期交互、历史购买、商品热度及时间间隔。不能用组间特征差异直接认定缺货、运费或价格导致流失，这些原因需要订单与页面日志核实。",
        "",
        "## 固定推荐预算比较",
        "",
    ]
    lines += table(
        rank[np.isclose(rank.fraction, 0.1)],
        {
            "model": "模型",
            "selected_pairs": "前10%候选对",
            "purchased_pairs": "观测购买对",
            "precision_at_k": "命中比例",
            "recall_at_k": "购买覆盖",
            "lift_over_random": "相对随机排序倍数",
        },
    )
    lines += [
        "![推荐预算曲线](plots/09_ranking_budget.png)",
        "",
        "同样的候选数量预算下比较命中与覆盖，更接近推荐位有限的业务决策。这里的倍数是离线排序相对随机筛选的富集程度，不是A/B转化提升，也不是增量订单数。实际推荐前还需库存、商品合规、类目多样性和去重过滤。",
        "",
        "## 方法依据与交付",
        "",
        "[Stacking折外训练方法](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)",
        "[XGBoost贡献预测接口](https://xgboost.readthedocs.io/en/release_2.1.0/prediction.html)",
        "[SHAP穷举解释方法](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html)",
        "",
        "Stacking二层权重、三折检查点、折内预处理、OOF预测、SHAP值、错误样本与分群指标均已保存在本次输出目录。带用户商品标识的案例和预测仅保存在本地，不应直接上传公开仓库。",
        "",
    ]
    save_report(output / "模型融合与可解释性分析报告", lines)
    write_business_report(output, business)


def write_business_report(output, business):
    scenarios = pd.read_csv(output / "business/assumed_business_benefits.csv")
    power = pd.read_csv(output / "business/ab_sample_size_planning.csv")
    balance = pd.read_csv(output / "business/historical_qualified_user_balance.csv")
    balance["arm"] = balance.arm.map({"control": "对照组", "treatment": "实验组"})
    scenario_display = scenarios.copy()
    scenario_display["policy"] = scenario_display.policy.map(
        {
            "recommendation_without_coupon": "无券推荐",
            "coupon_marketing": "优惠券营销",
        }
    )
    lines = [
        "# 业务落地模拟方案与收益测算",
        "",
        "## 业务结论与实验状态",
        "",
        "本次完成用户级历史分组模拟、推荐名单生成、实验样本量规划和收益敏感性测算，尚未开展真实触达、推荐投放或在线A/B测试。模型能够筛选较高购买倾向的候选组合，但预测命中不等于营销增量；用户本来就会购买时，发券可能只增加成本。",
        f"模拟覆盖{business['cohort_users']:,}名测试候选用户，稳定哈希分为对照组{business['control_users']:,}人和实验组{business['treatment_users']:,}人。触达资格阈值由验证集用户最高候选分数的80%分位固定，测试人群中{business['test_contact_eligible_users']:,}人符合，实际比例为{business['test_contact_eligible_rate']:.2%}，不是强行在测试结果中挑出20%。",
        "当前评分与每用户最多3件推荐仅来自抽样的测试候选对，不是全量商品目录。用户最高分受抽到的候选数量影响，验证与测试候选规模不同，因此资格比例可以偏离20%。正式部署需在一致的候选召回与每用户候选规模下重新验证阈值；下文收益使用独立设定的20%覆盖预算，不把本次14%左右的资格比例当作已实现触达率。",
        f"完整标签窗口内，这批用户任意商品的购买行为率为{business['historical_user_purchase_proxy']:.2%}；符合触达资格人群的历史购买行为率为{business['targeted_historical_user_purchase_proxy']:.2%}。该指标来自完整SQLite的用户级购买记录，不是0.64%的候选对购买率，也不是已核验支付转化率。高活跃用户与双十二所在观察期会影响基线，正式实验需重新采集近期支付基线。",
        "",
        "## 实验设计",
        "",
        "实验单位为用户，同一用户跨设备、推荐位和会话保持固定分组。实验前按冻结评分确定触达资格，主分析人群是所有满足资格的用户，按最初分组进行意向治疗分析（ITT），不根据实验后的点击、领券或是否成功触达筛选分析样本。候选不足或不同意营销的用户应在随机分组前统一排除。",
        "对照组维持现有推荐或运营策略；实验组在相同推荐位数量和曝光机会下采用模型排序，先测试不加券的推荐方案。对于可触达的近期浏览、加购和收藏人群，可在既有授权范围内测试一次补充提醒。每名用户7天内最多一次主动触达、推荐最多3件去重商品，过滤缺货、下架和重复购买不适用的商品。",
        "第一轮不同时更改推荐模型、券额和触达频率，便于识别策略作用。如需单独验证补贴，应另开预算受限的优惠券实验，明确券领取和使用条件。内容场、搜索、商城来源字段补齐后，可按来源和活跃度预分层随机，避免不同流量结构掩盖效果；现有数据暂不支持渠道级成交归因。",
        "主指标建议为分组后7天内产生有效支付的用户比例，每名用户只计一次；实际支付成功、取消及退款口径需由订单表确认。目前仅能用购买行为标记演示计算，不能作为支付验收结果。次指标包括每用户净GMV、贡献毛利、推荐点击率、加购率和购买用户覆盖；护栏指标包括退款率、投诉或退订率、触达频率、延迟、曝光集中度及冷启动与长尾覆盖。",
        "主指标应覆盖用户在商城的全部有效支付，推荐商品的成交作为次指标单列，防止用户只是从另一件原本会买的商品转到推荐商品，却被记为总成交增量。优惠券口径也需明确适用商品范围，避免把任意商品的用户购买基线直接当作某个SKU的用券转化率。",
        "使用双侧5%显著性水平和80%统计功效，预先注册主指标、分析人群、最小可检测提升和停止规则。检查用户去重、交叉曝光和样本比例异常，不每日挑选有利日期提前结束。主指标使用用户级差值及置信区间；多指标结果需说明探索性质，订单金额等偏态指标可按用户做bootstrap。",
        "",
        "## 周期与样本量",
        "",
        "规划连续14天纳入新用户，覆盖两个完整星期；最后一批用户再等待7天购买窗口，最早第21天读取主指标。若逐笔订单另需30天退款成熟期，则最终净收益最晚需等到第51天。样本量达不到预先要求时，应在上线前延长招募周期或调整可检测提升，不因中途结果接近显著而临时延长。",
        "下表按符合触达资格人群的历史用户购买行为率规划，绝对提升0.005表示提高0.5个百分点，不是提高0.5%。这是两独立比例的正态近似，每组人数指不同用户；正式支付基线、季节性及流量结构变化后需要重新计算。",
        "",
    ]
    lines += table(
        power,
        {
            "absolute_lift": "假设绝对提升",
            "required_users_per_arm": "每组用户数",
            "required_total_users": "两组合计",
            "minimum_daily_new_eligible_users_for_14day_enrollment": "14天内每日新增合格用户",
            "current_qualified_cohort_sufficient": "当前合格人数足够",
        },
    )
    lines += [
        "当前模拟样本主要用于跑通分组、数据记录与指标核算。若样本量表显示不足，不能依靠当前几千名用户证明细小提升。面向全平台用户的收益还会被触达覆盖率稀释：例如仅触达20%人群、触达人群提高1个百分点，对全部合格用户的提升约为0.2个百分点，不能把两者混用。",
        "",
        "## 历史安慰剂分组检查",
        "",
    ]
    lines += table(
        balance,
        {
            "arm": "随机组别",
            "users": "符合资格人数",
            "observed_buyers": "历史观测买家",
            "observed_buyer_rate": "历史购买比例",
        },
    )
    lines += [
        "上述表格只检查历史随机分组后两组的规模和购买基线，未施加任何新策略。两组比例差异来自随机分组和有限样本，不能叫作实验提升，也不能当作模型产生的成交增量。已输出本地模拟分组与每用户最多3件候选推荐，未执行实际投放。",
        "",
        "## 收益测算假设",
        "",
        "以每批10,000名合格候选用户、触达其中20%即2,000人为例。假设客单价150元、优惠前贡献毛利率25%、退款率10%，每名新增转化用户产生一笔订单。对触达人群分别假设0.5、1、2个百分点的因果转化提升，得到10、20、40名新增购买用户。这些提升和金额参数均为情景输入，未从离线AUC或购买倾向概率推算。",
        "无券推荐假设每名触达用户额外成本0.05元；优惠券营销假设每名触达用户成本0.20元、券额5元、成交用户用券比例30%。补贴支出覆盖实验组全部用券成交，包括原本就会购买的人；示例保守地假设退款相关补贴不返还。净GMV先扣退款，贡献毛利按优惠前金额计算，再单独扣补贴和触达成本，避免把券额同时从金额和成本扣两次。",
        "新增购买人数＝触达人数×假设绝对提升。新增净GMV＝新增购买人数×客单价×（1－退款率）。增量贡献＝新增净GMV×毛利率－触达成本－全部用券成交补贴。结果未计固定研发成本、履约成本差异及多单复购，应使用真实成本表进一步修订。",
        "",
    ]
    lines += table(
        scenario_display,
        {
            "policy": "策略",
            "assumed_absolute_lift": "假设提升",
            "expected_incremental_buyers": "新增买家",
            "expected_incremental_net_gmv": "新增净GMV/元",
            "expected_total_cost": "总增量成本/元",
            "expected_incremental_contribution": "增量贡献/元",
        },
    )
    lines += [
        "![收益情景敏感性](plots/10_assumed_net_benefit.png)",
        "",
        "图中正负收益取决于假设的因果提升与成本，不代表已经实现收益。高购买倾向用户可能本来就会下单，因此优惠补贴易被既有购买吸收。优先验证低成本排序和信息提醒，再用随机实验判断哪些人真正受到补贴影响；不要仅按购买倾向向所有高分用户发券。",
        "assumed_business_benefits.csv同时给出各方案的盈亏平衡绝对提升，可用于上线前预算审核。最终应以实验组与对照组每用户净贡献之差测算增量，不能把实验组所有成交都算作模型新增。",
        "",
        "## 精准营销与商品推荐建议",
        "",
        "近期多次浏览但未购买的用户可优先获得规格、价格和履约信息；加购或收藏用户可验证库存、到手价及商品比较信息是否有助决策。商品推荐首先保持与已观测兴趣一致，并加入类目多样性与适度长尾探索。用户或商品嵌入未知时使用类目、商品属性、历史热度与规则兜底，记录兜底比例和后续反馈。",
        "上述策略需要通过实验区分高购买倾向与可被策略影响的人群。若后续取得随机触达数据，可进一步建立uplift模型，减少向自然购买人群发放无效补贴。对已购买用户是否推荐同品，应结合消耗周期和退货情况，当前数据不足以统一设定复购时机。",
        "",
        "## 上线前需补齐的信息",
        "",
        "订单明细需要订单ID、用户ID、商品及SKU标识、下单和支付时间、支付状态、数量、标价、实付金额、优惠券及平台与商家承担金额、取消与退款金额和时间、履约及商品成本。实验日志需要实验ID、稳定用户标识、随机分组、资格判定时间、策略版本、候选列表、分数、排序、曝光、点击、触达送达与退订信息。",
        "补充统一商品映射、库存和场景来源后，才能分别评估短视频、直播、搜索和商城推荐的承接效果，并核验净支付与毛利。缺少这些信息时，现有输出用于历史离线诊断与情景规划，不能证明真实销售收入或营销投资回报。",
        "",
        "## 统计方法依据",
        "",
        "[两独立比例的功效与样本量规划](https://support.sas.com/kb/25/012.html)",
        "",
    ]
    save_report(output / "业务落地模拟方案与收益测算", lines)
