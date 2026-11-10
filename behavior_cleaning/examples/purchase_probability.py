"""仓库根目录安装pip install -e ./behavior_cleaning后运行。

python behavior_cleaning/examples/purchase_probability.py --help
"""
import argparse
from pathlib import Path

from behavior_cleaning import load_predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cutoff", required=True)
    args = p.parse_args()
    predictor = load_predictor(args.run_dir, models=["XGBoost", "Stacking"])
    predictions = predictor.predict(args.source, args.cutoff, max_pairs=1000)
    # 标识只在本地使用；示例只打印数量和概率摘要。
    print("候选数量:", len(predictions))
    print(predictions.filter(like="_probability").describe())


if __name__ == "__main__":
    main()
