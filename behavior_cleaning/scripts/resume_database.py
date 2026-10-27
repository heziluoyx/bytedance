"""在 D 盘临时目录中续建已完全导入的 SQLite 数据库。"""

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    """配置当前进程的临时目录，完成剩余索引与全部核验。"""
    root = Path(__file__).absolute().parents[1]
    scratch = root / "outputs" / "sqlite_temp"
    scratch.mkdir(parents=True, exist_ok=True)
    for variable in ("TMP", "TEMP", "SQLITE_TMPDIR"):
        os.environ[variable] = str(scratch)
    sys.path.insert(0, str(root / ".runtime"))
    sys.path.insert(0, str(root / "src"))
    from behavior_cleaning.database_builder import BuildConfig
    from behavior_cleaning.database_recovery import finish_database

    parser = argparse.ArgumentParser(description="仅续建已完整导入的临时 SQLite")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    arguments = parser.parse_args()
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
    target = finish_database(
        BuildConfig(arguments.input, arguments.database), arguments.temporary
    )
    print(f"建库完成：{target}")


if __name__ == "__main__":
    main()
