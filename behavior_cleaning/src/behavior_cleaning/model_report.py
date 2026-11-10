"""导出模型实验的静态图表、中文结论及复现实验说明。"""

import html
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def write_report(
    run: Path,
    dataset: dict,
    results: pd.DataFrame,
    predictions: dict,
    y_test: np.ndarray,
    metadata: dict,
) -> None:
    """报告如实说明分层随机留出、稀疏样本与计算预算的适用边界。"""
    run = Path(run)
    environment = {"python": sys.version, "packages": {}}
    for name in [
        "numpy",
        "pandas",
        "scikit-learn",
        "imbalanced-learn",
        "torch",
        "xgboost",
        "lightgbm",
        "optuna",
        "threadpoolctl",
        "dask",
        "distributed",
        "alembic",
        "Mako",
    ]:
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][name] = "not installed"
    (run / "environment_versions.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 130})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, probability in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        axes[0].plot(fpr, tpr, label=name)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].axhline(
        y_test.mean(), color="grey", linestyle="--", label="positive-rate baseline"
    )
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Frozen test ROC",
    )
    axes[1].set(
        xlabel="Recall", ylabel="Precision", title="Frozen test precision recall"
    )
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "01_test_roc_pr.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["auc", "average_precision", "f1"]):
        ax.barh(results.model, results[metric], color="#326b8c")
        ax.set_title("Test " + metric)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "02_test_metrics.png")
    plt.close(fig)
    imbalance = pd.read_csv(run / "imbalance_validation_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["auc", "average_precision"]):
        pivot = imbalance.pivot(index="strategy", columns="model", values=metric)
        pivot.plot.bar(ax=ax)
        ax.set_ylabel(metric)
        ax.set_title("Validation " + metric, pad=40)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=8)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(plots / "03_imbalance_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, info in metadata["deep"].items():
        curve = pd.read_csv(run / f"{name}_trial{info['best_trial']}_history.csv")
        axes[0].plot(curve.epoch, curve.train_weighted_loss, marker=".", label=name)
        axes[1].plot(curve.epoch, curve.validation_auc, marker=".", label=name)
    axes[0].set(
        title="Selected deep trial training loss", xlabel="Epoch", ylabel="Weighted BCE"
    )
    axes[1].set(
        title="Selected deep trial validation AUC",
        xlabel="Epoch",
        ylabel="Validation AUC",
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "04_deep_training_curves.png")
    plt.close(fig)
    winner = metadata["validation_selection_winner"]
    best = results.set_index("model").loc[winner]
    auc_leader = results.loc[results.auc.idxmax()]
    f1_leader = results.loc[results.f1.idxmax()]
    budgets = "、".join(
        name + str(limit) + "轮"
        for name, limit in metadata.get("deep_epoch_limits", {}).items()
    )
    lines = [
        "# 用户商品未来七天购买预测实验报告",
        "",
        "## 实验结论",
        "",
        f"按验证AUC选定的本轮候选模型为 **{winner}**。冻结测试集AUC为{best.auc:.4f}，平均精确率AP为{best.average_precision:.4f}，F1为{best.f1:.4f}。这是单个历史截点、有限预算下的结果，不代表已经达到上线标准。",
        f"从冻结后的测试结果看，{auc_leader.model}的AUC最高，为{auc_leader.auc:.4f}；{f1_leader.model}的F1最高，为{f1_leader.f1:.4f}。验证集领先并未保证测试集领先，说明本轮深度模型的优势尚未稳定泛化。这里的测试排序仅用于评估，不据此重新选择超参或触达阈值。",
        "",
        "## 样本与目标",
        "",
        f"预测时点为{dataset['cutoff']}，标签窗口结束于{dataset['label_end_exclusive']}（不含结束时点）。同一用户—商品对在窗口内有购买行为为1，否则为0。候选只来自预测时点前7天已交互商品，不用未来是否购买决定候选资格。",
        f"候选池{dataset['candidate_pool_pairs']:,}对，均匀蓄水池抽取{dataset['sample_pairs']:,}对，抽样种子42。候选规则覆盖未来已观测购买组合的{dataset['candidate_future_purchase_pair_coverage']:.2%}，因此模型结果只适用于这一近期交互候选池，不是全商城任意商品推荐。",
        "",
        "| 数据集 | 样本对数 | 正例 | 正例比例 | 候选商品嵌入未知率 | 未在训练出现的用户数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, info in dataset["splits"].items():
        lines.append(
            f"| {name} | {info['rows']:,} | {info['positives']:,} | {info['positive_rate']:.3%} | {info['target_item_oov_rate']:.2%} | {info['users_unseen_in_train']} |"
        )
    lines += [
        "",
        f"按标签分层7:2:1随机划分，同一用户或商品可跨集合出现，用户—商品对不重复。三个集合共享预测截点；这不是按时间前推的独立上线回测。测试仅{int(y_test.sum())}个正例，指标可能有较大抽样波动；本轮不据测试结果继续选择超参。",
        "",
        "## 特征和序列",
        "",
        f"传统模型使用{dataset['feature_count']}项处理后特征，包括历史用户、商品、交互次数与购买占比、最近行为间隔及训练折外类目目标编码。填补、标准化和类别映射仅在训练集拟合。",
        "序列使用历史最近7天中的最后16个活跃小时，每小时最多选4件商品后池化，保留行为分布、距预测时点间隔和候选商品匹配标记。同小时不虚构先后；并非秒级完整行为序列。商品词表最多20,000件，由训练样本及训练用户历史拟合；其余映射未知标记，类目嵌入与数值特征仍可使用。",
        "DIN采用候选商品与历史局部激活的小时桶适配版，保留兴趣强度；DIN_MHA在此基础上加入多头自注意力。这是适应现有小时精度的实现，不是原论文逐事件数据与训练条件的完整复现。",
        "",
        "## 不平衡处理与调优",
        "",
        "传统模型分别比较不处理、SMOTE、随机欠采样及类别权重，共12组基础训练；按验证AUC选择各模型策略，再用Optuna调优。重采样只在训练折内进行，验证及测试保留自然分布。SMOTE仅处理缩放后的数值和编码特征，不生成原始用户ID、商品ID或行为序列。",
        f"逻辑回归、XGBoost、LightGBM各{metadata['trials_per_traditional_model']}次Optuna试验，方向为最大化验证AUC。LSTM、GRU、DIN、DIN_MHA各{metadata['trials_per_deep_model']}次完成的架构试验，按下述轮数预算训练，调整嵌入维度、隐藏单元、Dropout和学习率，多头扩展同时比较注意力头数。深度模型使用真实序列及训练类别权重，不对离散序列做SMOTE。",
        "深度模型用验证AUC连续3轮无改善早停，ReduceLROnPlateau动态降低学习率，恢复最佳epoch。CPU线程设为2，未使用GPU；本轮属于可扩展的基础预算，并未穷尽超参数空间。",
        "本轮各架构的实际最大轮数预算为：" + budgets + "。不同预算会影响比较，应结合完整训练轨迹解读；中断试验不参与模型选择。",
        "",
        "## 冻结测试集比较",
        "",
        "表内分类阈值由验证集最大F1选定后固定；模型与策略选择以验证AUC为准。除所列阈值指标，CSV另保留0.5阈值准确率。",
        "",
        "| 模型 | 不平衡方案 | 验证AUC | 测试AUC | 测试AP | 准确率 | 精确率 | 召回率 | F1 | 阈值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results.itertuples():
        lines.append(
            f"| {r.model} | {r.strategy} | {r.validation_auc:.4f} | {r.auc:.4f} | {r.average_precision:.4f} | {r.accuracy:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.threshold:.4f} |"
        )
    strategy_description = "；".join(
        name + "选择" + info["strategy"] for name, info in metadata["traditional"].items()
    )
    differences = []
    for name, group in imbalance.groupby("model"):
        indexed = group.set_index("strategy")
        differences.append(
            name
            + f"的SMOTE验证AUC较不处理变化{indexed.loc['smote','auc']-indexed.loc['none','auc']:+.4f}"
        )
    lr_base = imbalance[imbalance.model == "LogisticRegression"].set_index("strategy")
    tradeoff = (
        f"例如逻辑回归使用类别权重后，验证AUC由{lr_base.loc['none','auc']:.4f}变为{lr_base.loc['class_weight','auc']:.4f}，"
        + f"但AP由{lr_base.loc['none','average_precision']:.4f}变为{lr_base.loc['class_weight','average_precision']:.4f}。"
        + "这说明整体排序与高意向人群筛选不完全一致；若后续只触达少量用户，还需关注头部命中率与成本。"
    )
    lines += [
        "",
        f"测试正例比例仅{y_test.mean():.3%}。全部预测不购买也能得到{1-y_test.mean():.3%}准确率，因此准确率不能单独衡量效果；应同时看AUC、AP、召回、精确率和实际触达成本。类别权重与重采样后的原始概率没有经过独立校准，不用于直接估算订单收益。",
        "本轮按验证AUC选择的传统模型方案为："
        + strategy_description
        + "。"
        + "；".join(differences)
        + "。因此，购买样本占比较低并不意味着必须将正负样本强制平衡，仍应由未重采样验证集的结果决定方案。",
        tradeoff,
        "",
        "## 模型适用场景与局限",
        "",
        "逻辑回归训练和部署成本较低，系数便于解释，适合作为稳定基线；它对复杂非线性和特征交互的表达有限。XGBoost与LightGBM更适合结构化行为聚合及非线性组合，需控制树复杂度，并注意稀少正例和时间变化导致的过拟合。",
        "LSTM与GRU可利用跨小时的顺序信息，但当前只有16个压缩小时桶，且多数商品未拥有独立词向量。DIN通过候选条件下的局部兴趣加权建模，适合历史与目标存在相关信号的场景；多头注意力增加历史交互表达能力，也增加参数和过拟合风险。是否更好应按本表结果判断，不预设深度模型一定优于树模型。",
        "传统模型使用数值与编码特征，深度模型额外使用历史序列和嵌入。因此，本轮差异包含模型结构与输入信息的共同作用，不能全部归因于网络本身；若要识别序列的独立增益，还需同一输入条件下的消融比较。",
        "",
        "## 适用边界和下一步",
        "",
        "现有清洗异常阈值曾参考全期分布，虽然本次特征严格按截点构建，端到端正式回测仍需仅以历史拟合清洗阈值。未观测到购买不等于确认无真实交易；缺订单、金额和退款，目标是行为标记。",
        "下一步优先扩大正例与商品覆盖，增加多个预测截点的时间外回测，并按新用户、已见用户和长尾商品分层验证。需取得更细时间戳才能验证精确顺序；接入有效支付与退款后再评估收益。SMOTE是比较方案，不应直接把合成样本解释为真实用户行为。",
        "",
        "## 图表",
        "",
    ]
    for name, description, interpretation in [
        (
            "01_test_roc_pr.png",
            "测试ROC与PR曲线",
            "ROC反映不同阈值下对购买与未购买样本的排序能力；PR更直接展示少量购买正例的识别质量。购买率很低时，即使ROC表现较好，实际触达名单仍可能包含大量未购买样本，因此需要同时检查精确率与召回率。",
        ),
        (
            "02_test_metrics.png",
            "七种模型测试指标",
            f"{auc_leader.model}在本次测试的AUC为{auc_leader.auc:.4f}，{f1_leader.model}的F1为{f1_leader.f1:.4f}。各模型的分类阈值已经由验证集固定，F1反映这一具体决策规则，而AUC评估整体排序；两类指标领先的模型可以不同。",
        ),
        (
            "03_imbalance_comparison.png",
            "不平衡方案验证比较",
            "图中使用未经重采样的验证集，比较相同基础模型在四种训练策略下的表现。SMOTE和欠采样没有带来一致改善，类别权重也可能提高AUC却降低AP；业务只筛选少量高意向人群时，应特别关注头部命中质量。",
        ),
        (
            "04_deep_training_curves.png",
            "深度模型最佳试验训练轨迹",
            "训练损失持续下降而验证AUC回落，表示继续拟合训练样本未必改善泛化。各模型恢复验证AUC最高轮次的权重；曲线长短受早停与不同最大轮数共同影响，不能仅凭训练损失较低认定模型更好。",
        ),
    ]:
        lines += [f"![{description}](plots/{name})", "", interpretation, ""]
    lines += [
        "## 官方方法依据",
        "",
        "- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
        "- https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html",
        "- https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html",
        "- https://arxiv.org/abs/1706.06978",
        "",
    ]
    (run / "模型训练与评估报告.md").write_text("\n".join(lines), encoding="utf-8")
    # 只渲染本报告使用的标题、段落、表格和本地图，避免增加文档依赖。
    blocks, in_table = [], False
    for line in lines:
        if line.startswith("|"):
            if re.match(r"^\|[-|: ]+\|$", line):
                continue
            cells = [html.escape(x.strip()) for x in line.strip("|").split("|")]
            if not in_table:
                blocks.append(
                    "<table><thead><tr>"
                    + "".join("<th>" + c + "</th>" for c in cells)
                    + "</tr></thead><tbody>"
                )
                in_table = True
            else:
                blocks.append(
                    "<tr>" + "".join("<td>" + c + "</td>" for c in cells) + "</tr>"
                )
            continue
        if in_table:
            blocks.append("</tbody></table>")
            in_table = False
        if line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            blocks.append(
                '<figure><img src="'
                + html.escape(match[2])
                + '"><figcaption>'
                + html.escape(match[1])
                + "</figcaption></figure>"
            )
        elif line.startswith("## "):
            blocks.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line.startswith("# "):
            blocks.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line:
            value = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(line))
            blocks.append("<p>" + value + "</p>")
    content = "\n".join(blocks)
    (run / "模型训练与评估报告.html").write_text(
        '<meta charset="utf-8"><title>模型训练与评估报告</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.8 Microsoft YaHei,sans-serif;color:#18212a}h1,h2{color:#000}table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}td,th{border:1px solid #d9d9d9;padding:8px}th{background:#244b63;color:white}tbody tr:nth-child(even){background:#f1f5f7}img{width:100%}figure{margin:32px 0}figcaption{color:#51626d}</style><main>'
        + content
        + "</main>",
        encoding="utf-8",
    )
