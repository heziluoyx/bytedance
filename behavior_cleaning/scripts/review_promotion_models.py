"""重新训练较早窗口传统模型并冻结评估双十二覆盖窗口。"""
import argparse
import logging
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = "2"
    temporary = root / "outputs/sqlite_temp"
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ["TEMP", "TMP", "SQLITE_TMPDIR"]:
        os.environ[name] = str(temporary)
    for path in [root / "src", root / ".runtime", root / ".model_runtime"]:
        sys.path.insert(0, str(path))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "outputs/sql_analysis_v2/user_behavior.sqlite",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs/promotion_review_20261104"
    )
    parser.add_argument("--max-pairs", type=int, default=120000)
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    if args.max_pairs < 100 or args.trials < 1:
        parser.error("样本至少100，调优次数至少1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run.log", encoding="utf-8"),
        ],
    )
    from behavior_cleaning.promotion_robustness import run_promotion_review

    run_promotion_review(args.database, args.output_dir, args.max_pairs, args.trials)


if __name__ == "__main__":
    main()
