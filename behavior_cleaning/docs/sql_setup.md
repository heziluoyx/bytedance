# SQLite 建库与导入（任务前五步）

2026-10-26 补充：新版清洗保留正常重复，导入时自动使用
sql/02_indexes_preserve.sql 的普通四元组索引，并默认另存至 outputs/sql_analysis_v2/。
下文关于唯一四元组索引和本次执行数量的说明属于 2026-10-20 的旧版本。
旧数据库不会因重跑清洗而自动更新。

本阶段完成输入确认、目录整理、第三范式建模、Parquet 批量导入、索引建立及验证。
基础分析、漏斗分析、用户分群和 Python 查询接口尚未在此阶段执行。

## 当前数据和输出

输入为 `outputs/run_20261019_163522_0448a282/cleaned_events.parquet`。
原始清洗数据及 `audit.sqlite` 保持不变。

默认数据库为 `outputs/sql_analysis/user_behavior.sqlite`。
相邻的 `user_behavior.validation.json` 记录输入 SHA-256、各表行数、行为类型数量、
索引、完整性、外键校验、与清洗报告及聚合表的交叉核验结果。
`user_behavior.build.log` 保存构建进度。

## 运行

在 Anaconda PowerShell Prompt 中执行：

```powershell
python 'D:\字节\behavior_cleaning\scripts\build_database.py'
```

脚本默认查找 outputs 下最近一个带 SUCCESS 的清洗目录。
需要显式指定输入或使用其他输出文件时：

```powershell
python 'D:\字节\behavior_cleaning\scripts\build_database.py' --input 'D:\字节\behavior_cleaning\outputs\run_20261019_163522_0448a282\cleaned_events.parquet' --database 'D:\字节\behavior_cleaning\outputs\sql_analysis\user_behavior_v2.sqlite'
```

已有数据库或对应校验报告会导致脚本拒绝运行，不会覆盖现有结果。
构建使用临时数据库，仅在通过所有验证后才发布正式文件；失败的临时库保留供排查。
输出旁的 `.build.lock` 防止重复构建；若进程被强制终止，必须确认没有运行中的构建，
再手动处理遗留锁文件。不要同时在两个窗口启动同一个输出目标。

## 表结构与第三范式

| 表 | 主键 | 其他字段及说明 |
| --- | --- | --- |
| dim_user | user_id TEXT | 目前没有用户画像字段，不补造属性 |
| dim_item | item_id TEXT | category_id TEXT，只在商品表保存类目 |
| dim_time | event_time TEXT | YYYY-MM-DD HH:00:00，无时区转换 |
| user_behavior | event_id INTEGER | user_id、item_id、behavior_type、event_time |

用户、商品、时间均由外键约束关联。行为编码通过 CHECK 限定为 1、2、3、4。
行为主表不重复存储 category_id。联合唯一索引约束用户、商品、行为和时间四元组。
日期、年、月、小时、星期和周末标志由 `v_time_attributes` 视图推导，
不在时间基础表中重复保存。weekday 使用周一=1，周日=7；周末不等同于法定节假日。

时间格式补齐分钟、秒为零是为了兼容 SQLite 日期函数，原始时间仍只有小时精度。
ID 全程保留字符串及前导零。源文件名、来源行号、异常标记仍可在原始清洗 Parquet
中查询，不参与本次四张规范化基础表的业务字段。

若商品类目发生一对多冲突，导入将失败，不会悄悄保留第一个类目。
当前模型采用 item_id -> category_id，后续如果数据含类目历史变更，需要先调整模型。

## 导入与索引

PyArrow 每次读入 100,000 行。每批先插入维度，再写入行为表，通过一个事务提交。
导入时启用外键，SQLite 页缓存限制为约 256 MiB，临时排序允许落盘。
2026-10-27 起同时设置最高约 512 MiB 的内存映射窗口及两个索引排序工作线程，
各索引分别记录耗时，便于跟踪大数据量建库进度。
所有数据加载结束后统一创建索引，并运行 ANALYZE 生成优化器统计信息。
ANALYZE 属于数据库维护，不是第六步业务分析。

索引包括：四元组唯一索引、用户与时间、商品与行为及用户、行为与时间、时间与用户、
商品类目。SQL 定义位于 `sql/01_schema.sql` 和 `sql/02_indexes.sql`。

## 验收

- 行为表行数与 Parquet 元数据、实际读入行数相同。
- 行为类型分布与导入前计算结果一致。
- 与同目录清洗报告中的总行数、行为类型分布一致。
- 用户、商品、时间维度行数与上阶段对应聚合 Parquet 行数一致。
- PRAGMA foreign_key_check 没有异常。
- PRAGMA integrity_check 返回 ok。
- 四元组唯一索引成功建立，六个预期索引均存在。

校验中的 COUNT、行为分布用于核对导入正确性，不包含第六步的业务分析报表。
后续每次用 Python 新建数据库连接时都需要重新执行 `PRAGMA foreign_keys=ON`，
因为 SQLite 的外键开关按连接生效。

## 用 DB Browser 查看

1. 打开 DB Browser for SQLite。
2. 点击“打开数据库 / Open Database”。
3. 选择 `D:\字节\behavior_cleaning\outputs\sql_analysis\user_behavior.sqlite`。
4. 在“数据库结构 / Database Structure”查看四张表、视图和索引。
5. 在“浏览数据 / Browse Data”选择维度表或行为表，查看数据。

这是 SQLite 本地文件，不需要配置服务器地址、端口、用户名或密码。
构建期间不要用其他软件编辑正在生成的数据库。

## 本次执行结果

2026-10-20 已对完整清洗数据完成建库，数据库约 1.55 GB（十进制）。

| 表 | 行数 |
| --- | ---: |
| user_behavior | 6,213,379 |
| dim_user | 10,000 |
| dim_item | 2,876,947 |
| dim_time | 744 |

六个索引全部建立，外键检查通过，integrity_check 返回 ok。
总行数、行为类型分布及三个维度数量与上阶段结果全部一致。
通过 EXPLAIN QUERY PLAN 确认用户时间查询和类目商品查询均命中对应覆盖索引。
新增建库测试与原有清洗测试共 21 项通过。
详细结果以输出目录中的 user_behavior.validation.json 为准。

## Git 范围

提交 scripts/、sql/、docs/、src/ 和 tests/ 中的代码和说明。
outputs/ 中的数据库、日志、验证报告、清洗数据不提交。
