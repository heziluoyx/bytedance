"""behavior-toolkit统一入口，保持原清洗CLI兼容。"""
import argparse
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description="用户行为处理与购买概率工具包")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    clean = sub.add_parser("clean", help="保留正常重复的CSV清洗")
    clean.add_argument("--input", type=Path, required=True)
    clean.add_argument("--output", type=Path, required=True)
    for command in ["features", "train"]:
        p = sub.add_parser(command)
        p.add_argument("--source", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--max-pairs", type=int, default=80000)
        if command == "train":
            p.add_argument("--trials", type=int, default=12)
            p.add_argument("--epochs", type=int, default=8)
            p.add_argument("--deep-configurations", type=int, choices=[1, 2], default=2)
    p = sub.add_parser("predict", help="只使用预测前历史，不要求未来标签")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["XGBoost"])
    p.add_argument("--max-pairs", type=int, default=80000)
    p = sub.add_parser("bundle", help="导出本地受控模型资产")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .api import (
        TrainingConfig,
        build_features,
        clean_data,
        load_predictor,
        train_models,
    )

    if args.command == "clean":
        print(clean_data(args.input, args.output))
    elif args.command == "features":
        print(build_features(args.source, args.output, max_pairs=args.max_pairs))
    elif args.command == "train":
        print(
            train_models(
                TrainingConfig(
                    args.source,
                    args.output,
                    args.max_pairs,
                    args.trials,
                    args.epochs,
                    args.deep_configurations,
                )
            )
        )
    elif args.command == "bundle":
        print(load_predictor(args.run_dir).export_bundle(args.output))
    else:
        if args.output.exists():
            parser.error("预测输出已存在，请指定新文件")
        if args.output.suffix.lower() not in [".csv", ".parquet"]:
            parser.error("预测输出须为.csv或.parquet")
        predictor = load_predictor(args.run_dir, models=args.models)
        result = predictor.predict(args.source, args.cutoff, max_pairs=args.max_pairs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            result.to_csv(args.output, index=False, encoding="utf-8-sig")
        else:
            result.to_parquet(args.output, index=False)
        print(f"预测完成: {len(result)}对；输出 {args.output}")


if __name__ == "__main__":
    main()
