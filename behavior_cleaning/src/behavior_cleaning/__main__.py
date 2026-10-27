"""命令行入口：python -m behavior_cleaning。"""

import argparse
import logging
from pathlib import Path

from behavior_cleaning.cleaning import Config
from behavior_cleaning.pipeline import run_pipeline


def main() -> None:
    """解析命令行配置并执行清洗，错误向调用方传播。"""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="用户行为 CSV 全流程清洗")
    parser.add_argument(
        "--input", type=Path, default=project_root.parent / "user_behavior.csv"
    )
    parser.add_argument("--output", type=Path, default=project_root / "outputs")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--start", help="包含端点，YYYY-MM-DD HH")
    parser.add_argument("--end", help="包含端点，YYYY-MM-DD HH")
    parser.add_argument(
        "--repeat-policy",
        choices=["burst", "legacy"],
        default="burst",
        help="默认 burst 保留正常重复；legacy 为旧版四元组全量去重",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = Config(
        input_path=arguments.input,
        output_root=arguments.output,
        chunk_size=arguments.chunk_size,
        workers=arguments.workers,
        encoding=arguments.encoding,
        start=arguments.start,
        end=arguments.end,
    )
    if arguments.repeat_policy == "burst":
        from behavior_cleaning.burst_pipeline import run_burst_pipeline

        output = run_burst_pipeline(config)
    else:
        output = run_pipeline(config)
    print(f"清洗完成，输出目录：{output}")


if __name__ == "__main__":
    main()
