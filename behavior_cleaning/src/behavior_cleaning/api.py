"""稳定的工具包入口；可选机器学习依赖在调用相应功能时加载。"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """时间窗口训练预算；日期沿用项目既有四窗口设计。

    Attributes:
        source: 清洗后Parquet，需完整覆盖2025-11-18至12-18。
        output: 新的实验目录，已完成目录不能重训覆盖。
        max_pairs: 每个快照的历史候选抽样上限。
        trials: 每个传统模型的Optuna试验次数。
        epochs: 每次深度训练最大轮数。
        deep_configurations: 深度模型配置数量，取1或2。
    """

    source: Path
    output: Path
    max_pairs: int = 80000
    trials: int = 12
    epochs: int = 8
    deep_configurations: int = 2

    def __post_init__(self):
        if self.max_pairs < 200 or self.trials < 1 or self.epochs < 1:
            raise ValueError("max_pairs至少200，trials和epochs必须为正数")
        if self.deep_configurations not in (1, 2):
            raise ValueError("deep_configurations只能取1或2")


def clean_data(input_path, output_root, *, chunk_size=100000, workers=2, rules=None):
    """运行保留正常重复的清洗，返回独立输出目录。

    参数rules可传BurstRules。输入CSV文件或目录；失败不生成SUCCESS。
    """
    from .burst_pipeline import run_burst_pipeline
    from .cleaning import Config

    return run_burst_pipeline(
        Config(Path(input_path), Path(output_root), chunk_size, workers), rules
    )


def build_features(source, output, *, max_pairs=80000):
    """构建四个有完整未来标签的训练快照，返回统计摘要。

    此函数用于训练数据集；在线/批量推理使用PurchasePredictor而非本函数。
    """
    if max_pairs < 200:
        raise ValueError("max_pairs至少200")
    from .temporal_probability_data import build_temporal_dataset

    return build_temporal_dataset(Path(source), Path(output), max_pairs)


def train_models(config: TrainingConfig, *, include_analysis=True):
    """训练七种单模型、原时间留出Stacking与概率校准，返回实验目录。

    默认继续生成解释、诊断及HTML/Markdown。只有全流程成功才写SUCCESS。
    冻结的完整结果不可覆盖；中断目录只允许相同配置恢复。
    """
    import json

    import numpy as np
    import torch

    from .temporal_probability_models import (
        frozen_prediction_and_calibration,
        train_sequence_models,
        train_traditional,
    )

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").exists():
        raise FileExistsError("该实验已完成，请使用新的output目录")
    signature = {
        "source": str(Path(config.source).resolve()),
        "max_pairs": config.max_pairs,
        "trials": config.trials,
        "epochs": config.epochs,
        "deep_configurations": config.deep_configurations,
    }
    path = output / "configuration.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != signature:
        raise ValueError("实验配置发生变化，请使用新目录")
    path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    torch.set_num_threads(2)
    summary = build_features(config.source, output, max_pairs=config.max_pairs)
    arrays = dict(np.load(output / "dataset/arrays.npz"))
    traditional = train_traditional(arrays, output, config.trials)
    sequence = train_sequence_models(
        arrays, summary, output, config.epochs, config.deep_configurations
    )
    frozen_prediction_and_calibration(arrays, summary, output, traditional, sequence)
    if include_analysis:
        from .temporal_probability_analysis import explain_and_simulate
        from .temporal_probability_report import write_reports

        explain_and_simulate(config.source, output, arrays, summary, sequence)
        write_reports(output)
    (output / "SUCCESS").write_text("ok\n", encoding="utf-8")
    return output


def load_predictor(run_dir, *, models=None):
    """加载可信本地实验或导出包；不加载训练样本或未来标签。"""
    from .inference import PurchasePredictor

    return PurchasePredictor(run_dir, models=models)
