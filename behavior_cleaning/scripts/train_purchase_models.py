"""用户商品未来七天购买预测：分层样本与传统及深度模型训练。"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    """先设置项目依赖和线程，再加载训练模块。"""
    root = Path(__file__).resolve().parents[1]
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = "2"
    for name in ["TMP", "TEMP", "SQLITE_TMPDIR"]:
        os.environ[name] = str(root / "outputs/sqlite_temp")
    for folder in [root / "src", root / ".runtime", root / ".model_runtime"]:
        if folder.exists():
            sys.path.insert(0, str(folder))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "outputs/sql_analysis_v2/user_behavior.sqlite",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--stage", choices=["data", "train", "all"], default="all")
    parser.add_argument("--max-pairs", type=int, default=120000)
    parser.add_argument("--cutoff", default="2025-12-12 00:00:00")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--deep-trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lstm-epochs", type=int, default=None)
    args = parser.parse_args()
    if args.max_pairs < 100 or min(args.trials, args.deep_trials, args.epochs) < 1:
        parser.error("样本上限至少100，调优次数和训练轮数必须为正数")
    if args.lstm_epochs is not None and args.lstm_epochs < 1:
        parser.error("lstm-epochs必须为正数")
    try:
        cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        parser.error("cutoff须为YYYY-MM-DD HH:00:00格式")
    if cutoff.minute or cutoff.second:
        parser.error("源数据为小时精度，cutoff必须在整点")
    (root / "outputs/sqlite_temp").mkdir(parents=True, exist_ok=True)
    run = args.run_dir or root / "outputs" / (
        "purchase_models_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run / "training.log", encoding="utf-8"),
        ],
    )
    if args.stage in ["data", "all"]:
        from behavior_cleaning.pair_dataset import build_dataset

        build_dataset(args.database, run / "dataset", args.cutoff, args.max_pairs)
        (run / "run_config.json").write_text(
            json.dumps(vars(args), default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.stage in ["train", "all"]:
        from behavior_cleaning.model_experiments import run_experiments

        run_experiments(
            run, args.trials, args.deep_trials, args.epochs, args.lstm_epochs
        )
    print("运行目录：" + str(run), flush=True)


if __name__ == "__main__":
    main()
