"""免安装入口：将清洗结果导入 SQLite，并完成前五步核验。"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    """解析建库参数，默认选用最近一次成功清洗的 Parquet。"""
    project_root = Path(__file__).absolute().parents[1]
    temporary_directory = project_root / "outputs" / "sqlite_temp"
    temporary_directory.mkdir(parents=True, exist_ok=True)
    # 仅影响当前建库进程，避免大索引排序占满系统盘。
    for variable in ("TMP", "TEMP", "SQLITE_TMPDIR"):
        os.environ[variable] = str(temporary_directory)
    runtime = project_root / ".runtime"
    if runtime.is_dir():
        sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(project_root / "src"))

    from behavior_cleaning.database_builder import BuildConfig, build_database

    parser = argparse.ArgumentParser(description="SQLite 建库、导入、索引及核验")
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    arguments = parser.parse_args()
    source = arguments.input
    if source is None:
        candidates = sorted((project_root / "outputs").glob("run_*"), reverse=True)
        source = next(
            (
                path / "cleaned_events.parquet"
                for path in candidates
                if (path / "SUCCESS").is_file()
                and (path / "cleaned_events.parquet").is_file()
            ),
            None,
        )
        if source is None:
            parser.error("没有成功的清洗输出，请使用 --input 指定 Parquet")
    if arguments.database is None:
        quality_path = source.parent / "quality_report.json"
        quality = (
            json.loads(quality_path.read_text("utf-8")) if quality_path.exists() else {}
        )
        preserve = (
            quality.get("configuration", {}).get("duplicate_policy")
            == "preserve_regular_repetitions"
        )
        folder = "sql_analysis_v2" if preserve else "sql_analysis"
        arguments.database = project_root / "outputs" / folder / "user_behavior.sqlite"
    arguments.database.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                arguments.database.with_suffix(".build.log"), encoding="utf-8"
            ),
        ],
    )
    output = build_database(
        BuildConfig(source, arguments.database, arguments.chunk_size)
    )
    print(f"建库完成：{output}")
    print(f"校验报告：{output.with_suffix('.validation.json')}")


if __name__ == "__main__":
    main()
