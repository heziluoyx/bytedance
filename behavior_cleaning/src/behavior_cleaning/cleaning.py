"""分块标准化与业务规则校验，导出 Config、CleanResult 和 clean_chunk。"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
    "timestamp",
)
ALIASES = {"time": "timestamp", "item_category": "category_id"}
BEHAVIOR_NAMES = {1: "pv", 2: "fav", 3: "cart", 4: "buy"}
MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a", "<na>"}


@dataclass(frozen=True)
class Config:
    """运行配置。时间界限为包含端点的本地小时，默认不臆定采样窗口。

    Attributes:
        input_path: 输入 CSV 文件或目录。
        output_root: 每次运行独立子目录的父目录。
        chunk_size: 每次读入的行数。
        workers: 并行校验线程数，待处理块数限制为 workers。
        encoding: CSV 编码。
        start: 可选的最早有效时间，格式 YYYY-MM-DD HH。
        end: 可选的最晚有效时间，格式 YYYY-MM-DD HH。
    """

    input_path: Path
    output_root: Path
    chunk_size: int = 100_000
    workers: int = 2
    encoding: str = "utf-8-sig"
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        """校验资源配置及时间范围。

        Raises:
            ValueError: 块大小、线程数或时间范围不合法。
        """
        if self.chunk_size < 1 or self.workers < 1:
            raise ValueError("chunk_size 和 workers 必须大于零")
        for bound in (self.start, self.end):
            if bound is not None:
                parsed = pd.to_datetime(bound, format="%Y-%m-%d %H", errors="raise")
                if parsed.strftime("%Y-%m-%d %H") != bound:
                    raise ValueError("时间界限必须严格为 YYYY-MM-DD HH")
        if self.start and self.end and self.start > self.end:
            raise ValueError("start 不能晚于 end")


@dataclass
class CleanResult:
    """单块清洗结果，保留拒绝记录及逐字段缺失统计。

    Attributes:
        accepted: 标准化后的有效数据。
        rejected: 原始数据及源文件、数据行号、全部拒绝原因。
        missing_counts: 逐字段缺失行数。
        input_rows: 读入行数。
    """

    accepted: pd.DataFrame
    rejected: pd.DataFrame
    missing_counts: dict[str, int]
    input_rows: int


def clean_chunk(
    raw: pd.DataFrame,
    source: str,
    offset: int,
    config: Config,
) -> CleanResult:
    """校验单个原始数据块，不修改输入。

    Args:
        raw: 原始字符串数据，列名尚未规范化。
        source: 源文件绝对路径。
        offset: 本块之前已读取的数据记录数，不含表头。
        config: 时间规则等配置。

    Returns:
        含有效记录与可追溯拒绝明细的结果。数据行号从 1 开始。

    Raises:
        ValueError: 标准字段缺失或别名映射后出现重名列。
    """
    frame = raw.copy().reset_index(drop=True)
    frame.columns = [
        ALIASES.get(str(name).strip(), str(name).strip()) for name in frame.columns
    ]
    if frame.columns.duplicated().any():
        raise ValueError(f"{source}: 字段别名映射后存在重名列")
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{source}: 缺少字段 {sorted(missing_columns)}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)].astype("string")
    reasons = pd.Series("", index=frame.index, dtype="string")
    missing_counts = {}
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].str.strip()
        missing = frame[column].isna() | frame[column].str.lower().isin(MISSING_TOKENS)
        missing_counts[column] = int(missing.sum())
        frame.loc[missing, column] = pd.NA
        reasons.loc[missing] += f"missing_{column};"

    # 标识符是离散编码，保留前导零，不对 ID 做 IQR、均值填充或数值压缩。
    for column in ("user_id", "item_id", "category_id"):
        valid = frame[column].str.fullmatch(r"[0-9]+", na=False)
        valid &= frame[column].str.contains(r"[1-9]", na=False)
        invalid = frame[column].notna() & ~valid
        reasons.loc[invalid] += f"invalid_{column};"

    invalid_behavior = frame["behavior_type"].notna() & ~frame["behavior_type"].isin(
        ["1", "2", "3", "4"]
    )
    reasons.loc[invalid_behavior] += "invalid_behavior_type;"
    parsed = pd.to_datetime(frame["timestamp"], format="%Y-%m-%d %H", errors="coerce")
    strict_format = frame["timestamp"].str.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}", na=False
    )
    invalid_time = frame["timestamp"].notna() & (~strict_format | parsed.isna())
    reasons.loc[invalid_time] += "invalid_timestamp;"
    if config.start:
        reasons.loc[parsed < pd.Timestamp(config.start)] += "before_start;"
    if config.end:
        reasons.loc[parsed > pd.Timestamp(config.end)] += "after_end;"

    valid_mask = reasons.eq("")
    accepted = frame.loc[valid_mask].copy()
    accepted["timestamp"] = parsed.loc[valid_mask].dt.strftime("%Y-%m-%d %H")
    accepted["behavior_type"] = accepted["behavior_type"].astype("int8")
    accepted["source_file"] = source
    accepted["source_row"] = accepted.index + offset + 1
    rejected = raw.reset_index(drop=True).loc[~valid_mask].copy()
    rejected["source_file"] = source
    rejected["source_row"] = rejected.index + offset + 1
    rejected["reject_reason"] = reasons.loc[~valid_mask].str.rstrip(";")
    return CleanResult(accepted, rejected, missing_counts, len(raw))
