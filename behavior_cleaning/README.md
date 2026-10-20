# 用户行为 CSV 数据清洗

适用 Python 3.10+。根据项目第一阶段的数据加载、清洗、质量校验、Parquet
格式转换和三个维度基础聚合任务实现。代码采用 4 空格缩进、88 字符行宽、
绝对导入、类型注解、Google 风格公共接口文档、日志及独立测试。

## 安装与运行

### 当前电脑的免安装启动方式

如果安装项目包时出现 `Cannot set --home and --prefix together`，可以直接执行：

```powershell
python 'D:\字节\behavior_cleaning\run_cleaning.py'
```

启动脚本自动加载源码和已有的 `.runtime` 本地依赖，不需要执行 `pip install -e`。
它接受与下述模块命令相同的参数。迁移到其他电脑时，如果缺少依赖，先执行
`python -m pip install -r requirements.txt`；本地 `.runtime` 不包含在源码压缩包内。

### 标准安装方式

在 PowerShell 中执行：

```powershell
Set-Location 'D:\字节\behavior_cleaning'
python -m pip install -e '.[dev]'
python -m behavior_cleaning
```

默认读取 `D:\字节\user_behavior.csv` 目录内的 CSV，自动忽略 `__MACOSX`
和 `._` 开头的文件。支持单文件或目录；目录内 CSV 必须都是同类行为数据。
按文件路径排序后依次处理，以保证保留第一条的结果可重现。

显式指定文件、分块行数及并行线程数：

```powershell
python -m behavior_cleaning --input 'D:\字节\user_behavior.csv\user_behavior.csv' --chunk-size 100000 --workers 2
```

仅在业务已经确认采样窗口时指定时间范围，例如：

```powershell
python -m behavior_cleaning --start '2025-11-18 00' --end '2025-12-18 23'
```

这个日期范围只是当前样本所见范围的示例，代码默认不把样本范围当成业务规则。
编码默认 `utf-8-sig`，其他编码通过 `--encoding gb18030` 等显式指定。
内存紧张时用 `--chunk-size 30000 --workers 1`。

## 清洗规则

| 原始字段 | 标准字段 | 校验和存储方式 |
| --- | --- | --- |
| `user_id` | `user_id` | 正整数字符串，保留前导零 |
| `item_id` | `item_id` | 正整数字符串，保留前导零 |
| `item_category` | `category_id` | 正整数字符串，保留前导零 |
| `behavior_type` | `behavior_type` | 仅允许 1、2、3、4，输出 int8 |
| `time` | `timestamp` | 严格 YYYY-MM-DD HH，输出 Parquet timestamp |

也接受已经规范化的标准字段名。出现重名、别名与标准名同时出现、缺少必要字段时立即报错。
其他字段从清洗明细中剔除，字段名写入报告。原始文件不修改。

1. 去除字段首尾空白，将空白、null、none、nan、na、n/a、`<na>` 识别为缺失，
   不区分大小写。必要字段缺失的行单独隔离，不填造 ID 或行为类型。
2. 检查正整数 ID、行为范围、严格时间格式和真实日期，及可选的起止范围。
   不转换时区，也不把小时数据伪造成更精细的行为时间。负数、小数形式 ID 不接受。
3. 对有效记录按 `user_id + item_id + behavior_type + timestamp` 全局去重。
   SQLite 磁盘唯一主键覆盖跨块和跨文件重复，不使用内存无限增长的 Python set。
   同小时真实重复浏览也会被此任务规定的四元组合并，需据此解释清洗后 PV。
4. 去重时稳定保留第一条。若同一四元组类目不同，在重复审计表中保留原值、
   被保留的类目和来源。对全量有效记录中的商品多类目情况输出冲突表，
   不擅自取众数修正，不删除冲突商品。报告的一致性检查可因此为 false。
5. 在全局去重后的用户日行为量上计算 IQR 上界 `Q3 + 1.5 * (Q3 - Q1)`
   及 3σ 上界 `mean + 3 * population_std`。从精确频数分布计算分位数，
   不使用分块阈值或抽样阈值；分位数采用线性插值。
   默认仅标记高活跃用户日，并把标记关联回行为明细，不自动删掉潜在真实行为。
   ID 和离散行为编码不适合套用连续变量异常值法。
6. Parquet 使用 Zstandard 压缩及字典编码；行为编码 int8，行为名称使用类别型。
   高基数用户和商品 ID 保持字符串，避免整数溢出、精度丢失和前导零丢失。

## 输出与审计

每次创建独立的 `outputs/run_日期时间_随机后缀/`，不会覆盖以前结果。
只有包含 `SUCCESS` 的目录才代表执行成功。失败时保留已有审计和日志供排查。

| 文件 | 内容 |
| --- | --- |
| `cleaned_events.parquet` | 标准五字段、来源文件、数据记录号、行为名称、两个异常标记 |
| `user_summary.parquet` | 用户行为量、四类行为量、商品数、活跃天数、首次/末次时间 |
| `item_summary.parquet` | 商品行为量、用户数、类目数、四类行为量 |
| `time_summary.parquet` | 小时行为量、独立用户数、四类行为量 |
| `user_daily_flags.parquet` | 用户日行为量和 IQR、3σ 标记 |
| `item_category_conflicts.parquet` | 存在多个类目的商品及类目数，包含去重丢弃行的类目信息 |
| `duplicates.parquet` | 被丢弃的有效重复行及保留行的来源、类目 |
| `rejected.jsonl` | 规则拒绝的原始记录、全部拒绝原因、源文件和数据记录号 |
| `quality_report.json` | SHA-256、配置、行数、缺失、拒绝、重复、时间、类目冲突、异常阈值 |
| `audit.sqlite` | 全局去重、冲突追踪和聚合使用的磁盘审计数据库 |
| `cleaning.log` | 进度及失败日志 |

`source_row` 是原始文件表头之后从 1 开始的数据记录编号，不是包含表头的物理文本行号。
拒绝原因统计可能一行对应多个原因，各原因之和不等于拒绝行数。
用户和商品 ID 均以不透明字符串处理；商品跨类目可能是脏数据，也可能是业务变化，
需由数据提供方确认。行为计数是清洗后事件数，不是订单数或销量。

自动核对：`输入行数 = 规则拒绝行数 + 重复行数 + 清洗后行数`，
导出行数等于清洗后行数，用户聚合行为总量等于清洗后行数。
唯一性由数据库主键约束保障；必要字段有效性由写入前校验保障。
CSV 语法损坏时停止，不使用 `on_bad_lines="skip"` 静默丢行。

输入使用 Pandas 分块读取和有限队列并行校验，写入数据库顺序固定。
不需要为当前单机文件额外引入 Dask。SQLite 精确去重和 DISTINCT 聚合会使用磁盘，
它们适用于当前约 492 MB 文件，但并非分布式亿级性能承诺。
保留完整重复审计意味着工作空间应容纳数据库、临时排序及导出文件；
磁盘占用会大于原始文件，运行时间取决于硬盘性能和重复率。
执行过程中不要修改输入文件，以保证报告哈希与内容一致。

## 测试及代码检查

```powershell
python -m pytest
python -m isort --check-only src tests
python -m black --check src tests
python -m mypy src
```

测试覆盖时间格式、非法行为、缺失 ID、别名冲突、端点时间、跨块和跨文件去重、
类目冲突追踪、拒绝行核对以及异常只标记不删除。测试使用临时目录，原始 CSV 不受影响。

本目录包含完整代码和测试。后续特征编码、归一化、未来七天标签和模型训练不在清洗阶段执行，
避免在全量清洗过程中引入未来信息。源代码和运行说明纳入 Git 版本管理。
