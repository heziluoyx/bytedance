# 新版清洗数据的 SQLite 建库（第 1—5 步）

## 本次范围

使用最新成功的新版清洗结果重新完成输入确认、目录整理、表结构、分批导入、
索引建立及数据库核验。第六步基础 SQL 业务分析、漏斗、用户分群、模型训练不在本次范围。
导入校验需要的 COUNT 和行为分布仅用于核对数据完整性，不作为业务分析报表。

## 输入和输出

- 输入：outputs/run_20261026_181850_f5ed88ab/cleaned_events.parquet。
- 输入清洗政策：preserve_regular_repetitions。
- 清洗后记录：12,204,330 行。正常重复不再全量去重。
- 新数据库：outputs/sql_analysis_v2/user_behavior.sqlite。
- 验证报告：outputs/sql_analysis_v2/user_behavior.validation.json。
- 进度日志：outputs/sql_analysis_v2/user_behavior.build.log。

旧版 outputs/sql_analysis/user_behavior.sqlite 及其报告、原始 CSV 和各版 Parquet 均保留。

## 数据模型

| 对象 | 字段与职责 |
| --- | --- |
| dim_user | user_id 字符串主键 |
| dim_item | item_id 字符串主键；category_id 只在本表保存 |
| dim_time | event_time 小时时间主键，保存为 YYYY-MM-DD HH:00:00 |
| user_behavior | event_id 整数主键；user_id、item_id、behavior_type、event_time |
| v_time_attributes | 从时间主键派生日期、年、月、小时、星期和周末标志的视图 |

主表的用户、商品和时间键均设置外键，行为类型通过 CHECK 限制为 1、2、3、4。
主表不重复存储类目，时间基础表不重复存储可推导属性，保持第三范式基础表设计。
ID 保留前导零。补齐分钟和秒为零只为 SQLite 日期函数兼容，不提高原始时间精度。

## 与旧版的关键差异

新版使用 event_id 标识每一条保留事件。用户—商品—行为—小时四元组允许重复，
因此 idx_behavior_quadruple 是普通索引，不再使用旧版 uq_behavior_quadruple 唯一索引。
否则正常重复将被数据库拒绝或错误合并。

其余五个索引覆盖用户与时间、商品与行为及用户、行为与时间、时间与用户、商品类目。
全量加载完成后建立索引，并执行 ANALYZE 更新优化器统计信息。

## 导入与资源

每批读取 100,000 行 Parquet，先写维度，再写行为主表，在批次事务中提交。
分批核对 ID、行为值、小时精度和跨批次商品类目一致性。
SQLite 使用约 256 MiB 页缓存，最高约 512 MiB 内存映射窗口，临时排序允许落盘。
各索引单独记录耗时；索引耗时取决于磁盘和缓存情况。
建库入口会将当前 Python 进程的 TMP、TEMP、SQLITE_TMPDIR 指向
项目 outputs/sqlite_temp，避免大索引排序占满 C 盘。此设置不改动 Windows 全局环境变量。

构建先写入独立临时数据库，通过所有核验后才发布正式文件。
已有数据库或验证报告禁止覆盖；构建锁避免同一目标被重复启动。
如果数据已经全部导入，仅在索引阶段因空间等原因中断，可用 scripts/resume_database.py
对指定 building 临时库续建。续建会核对完整行数和行为分布、跳过已有索引，
再执行全部最终核验；未完整导入的库不允许使用该方式发布。

## 重现命令

在 Anaconda PowerShell Prompt 中逐条执行完整单行命令，末尾不添加反斜杠：

```powershell
python 'D:\字节\behavior_cleaning\scripts\build_database.py' --input 'D:\字节\behavior_cleaning\outputs\run_20261026_181850_f5ed88ab\cleaned_events.parquet' --database 'D:\字节\behavior_cleaning\outputs\sql_analysis_v2\user_behavior.sqlite'
```

该文件已经存在时会拒绝重跑；如需另行重建，请更换 --database 文件名。
脚本不传参数时会寻找最近带 SUCCESS 的清洗目录，并根据报告自动选择新旧索引政策。

## 验收标准

1. user_behavior 行数与实际读入及 Parquet 元数据一致。
2. 四类行为数量与清洗报告一致。
3. 用户、商品、时间维度数量与对应的新版聚合 Parquet 一致。
4. PRAGMA foreign_key_check 无异常，PRAGMA integrity_check 返回 ok。
5. 六个索引存在，四元组索引为非唯一，event_id 主键保留。
6. 新库中正常重复确实存在，未因建库再次进行机械去重。

## 在 DB Browser 中打开

1. 打开 DB Browser for SQLite。
2. 点击“打开数据库 / Open Database”。
3. 选择 D:\字节\behavior_cleaning\outputs\sql_analysis_v2\user_behavior.sqlite。
4. “数据库结构”查看四张表、时间视图和六个索引。
5. “浏览数据”选择 user_behavior 或维度表查看记录。

此数据库使用 SQLite 本地文件，无需配置数据库服务器、账号或端口。
后续新建 Python 连接时仍须执行 PRAGMA foreign_keys=ON，因为外键开关按连接生效。

## 2026-10-27 全量验收结果

已完成本说明中的第 1—5 步，正式数据库和验证报告均已生成。

| 表 | 实际行数 |
| --- | ---: |
| user_behavior | 12,204,330 |
| dim_user | 10,000 |
| dim_item | 2,876,947 |
| dim_time | 744 |

数据库大小 2,944,778,240 字节，约 2.94 GB（十进制）。
浏览、收藏、加购、购买记录数分别为 11,498,327、242,424、343,437、120,142，
均与指定新版清洗报告一致。这些数量用于导入质量核对，不是本轮基础分析成果。

六个索引均存在，四元组索引明确为非唯一；额外抽查发现同一四元组保留 3 条记录的样本，
证明正常重复没有被旧唯一约束压缩。外键检查通过，integrity_check 返回 ok，
与清洗报告及三张聚合表的全部交叉核验均通过。

建库与安全续建测试共 11 项通过。构建期间遇到 C 盘临时排序空间不足，
已将临时目录转至 D 盘，并在完整导入的临时库上安全续建，无需重复导入。
构建锁和 building 临时库在正式发布后已清理。

第六步基础业务 SQL 分析尚未执行，旧版 SQLite 数据库未覆盖。
