"""运行三折Stacking、解释性分析、错误诊断与业务模拟。"""

import argparse
import logging
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = "2"
    temporary = root / "outputs/sqlite_temp"
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ["TMP", "TEMP", "SQLITE_TMPDIR"]:
        os.environ[name] = str(temporary)
    for directory in [root / "src", root / ".runtime", root / ".model_runtime"]:
        if directory.exists():
            sys.path.insert(0, str(directory))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=root / "outputs/purchase_models_20261102"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs/purchase_followup_20261102"
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "outputs/sql_analysis_v2/user_behavior.sqlite",
    )
    parser.add_argument(
        "--stage", choices=["all", "stacking", "analysis"], default="all"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run.log", encoding="utf-8"),
        ],
    )
    import torch

    torch.set_num_threads(2)
    if args.stage in ["all", "stacking"]:
        from behavior_cleaning.stacking_analysis import run_stacking

        run_stacking(args.run_dir, args.output_dir / "stacking")
    if args.stage in ["all", "analysis"]:
        from behavior_cleaning.business_analysis import run_analysis

        run_analysis(args.run_dir, args.output_dir, args.database)
    print(str(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
