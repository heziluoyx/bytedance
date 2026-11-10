"""严格时间窗口的购买概率预测，多模型、校准、融合与解释。"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for n in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[n] = "2"
    temporary = root / "outputs/sqlite_temp"
    temporary.mkdir(parents=True, exist_ok=True)
    for n in ["TMP", "TEMP", "SQLITE_TMPDIR"]:
        os.environ[n] = str(temporary)
    for p in [root / "src", root / ".runtime", root / ".model_runtime"]:
        sys.path.insert(0, str(p))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=root / "outputs/run_20261026_181850_f5ed88ab/cleaned_events.parquet",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/temporal_probability_20261109",
    )
    p.add_argument("--max-pairs", type=int, default=80000)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--deep-configurations", type=int, choices=[1, 2], default=2)
    p.add_argument(
        "--stage", choices=["dataset", "all", "analysis", "report"], default="all"
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run.log", encoding="utf-8"),
        ],
    )
    if args.max_pairs < 200 or args.trials < 1 or args.epochs < 1:
        p.error("样本至少200，调优和轮数至少1")
    import numpy as np
    import torch

    torch.set_num_threads(2)
    from behavior_cleaning.temporal_probability_data import build_temporal_dataset
    from behavior_cleaning.temporal_probability_models import (
        frozen_prediction_and_calibration,
        train_sequence_models,
        train_traditional,
    )

    summary = build_temporal_dataset(args.source, args.output_dir, args.max_pairs)
    if args.stage == "dataset":
        return
    configuration = {
        "source": str(args.source.resolve()),
        "max_pairs": args.max_pairs,
        "trials": args.trials,
        "epochs": args.epochs,
        "deep_configurations": args.deep_configurations,
    }
    file = args.output_dir / "configuration.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8")) != configuration:
        raise ValueError("已有运行配置不符，使用新目录")
    file.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    arrays = dict(np.load(args.output_dir / "dataset/arrays.npz"))
    if args.stage not in ["report", "analysis"]:
        if (args.output_dir / "SUCCESS").exists():
            raise FileExistsError("训练已完成；查看结果或指定新目录")
        traditional = train_traditional(arrays, args.output_dir, args.trials)
        sequence = train_sequence_models(
            arrays, summary, args.output_dir, args.epochs, args.deep_configurations
        )
        frozen_prediction_and_calibration(
            arrays, summary, args.output_dir, traditional, sequence
        )
    if args.stage != "report":
        from behavior_cleaning.temporal_probability_analysis import explain_and_simulate

        sequence = json.loads(
            (args.output_dir / "models/sequence_selection.json").read_text(
                encoding="utf-8"
            )
        )
        explain_and_simulate(args.source, args.output_dir, arrays, summary, sequence)
    from behavior_cleaning.temporal_probability_report import write_reports

    write_reports(args.output_dir)
    (args.output_dir / "SUCCESS").write_text("completed\n", encoding="utf-8")
    logging.info("全部时间概率预测任务完成")


if __name__ == "__main__":
    main()
