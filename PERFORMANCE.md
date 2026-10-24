# SQLite向量扫描优化（2026-10-25）

已完成两个后端各147题严格回放，以及24次配对计时：热缓存Python约5.96倍、
NumPy约37.11倍。首次加载、内存开销和比较边界见[完整实测报告](RESULTS_VECTOR_SCAN_20261025.md)。

基于已上传的`f49af330c65340c7ae76dae9ae3b5e364a042d66`，分支
`optimize/sqlite-vector-scan-20261025`。本轮只改变本地存储访问与计算，
保留SQLite、原JSON向量、三路候选预算、阈值、余弦分数、RRF及回答策略。
不会自动重抽记忆、重新嵌入或调用模型。

## 改动与配置

- 按scope和通路缓存解码后的float64向量及平方范数，使用紧凑数组，避免每次解析JSON。
- 可选NumPy批量计算初步分数，保留截断边界附近候选，再用原Python运算顺序重算返回分数。
  这不是ANN或float32量化。零向量、负相似度截断和按原插入顺序打破同分保持原行为。
- 问题向量查询只读`memory_id,embedding`，不再为每个问题重复读事实向量和正文。
- 确定候选后批量读取正文与元数据；BM25不再逐条读取事实；这些读取不加载embedding。
- 为JSON中的user_id、agent_id、run_id建立三个表达式索引；显式rowid顺序保留旧同分行为。

基础安装仍只要求原来的依赖，无NumPy时使用缓存后的Python精确扫描。可选安装：

```powershell
D:\conda\envs\agent_aml\python.exe -m pip install -e '.[fast]'
```

通过store控制后端和缓存；MiniVector自动创建store时采用`auto`：

```python
from mini_vector import MiniVector
from mini_vector.memory import SQLiteVectorStore

store = SQLiteVectorStore("memory.sqlite3", vector_backend="auto",
                          vector_cache_bytes=128*1024*1024)
memory = MiniVector(store=store, rerank_mode="none")
```

`vector_backend`可为`auto`、`python`或`numpy`，显式要求numpy但未安装时清楚报错。
`vector_cache_bytes=0`关闭解码缓存。默认最多8个LRU池、合计128MiB紧凑数值载荷，
ID和分组等Python对象另有开销；单池超预算则不驻留。缓存增加RAM，不缩小磁盘数据库。
最新受测方案明确关闭LLM重排；库原有构造函数的默认重排设置未改。

## 缓存正确性

同连接写入通过`total_changes`失效，其他连接提交通过`PRAGMA data_version`失效。
事务内可以读自己的未提交修改，但不保存事务内建立的池，避免rollback后留下脏缓存。
更新事实或问题、删除、回填、元数据变化均会在下次扫描时检查。消息／历史等其他写入
也会使缓存保守失效：频繁写入场景不应套用只读热缓存的加速倍数。

已有数据库打开时添加缺失索引，不转换向量或事实。原有事务、AML幂等、数据库身份与
服务操作锁保持现状；多进程高并发和事务缩短不属于本轮。

## 验证和性能回放

新增测试覆盖精确分数与同分、近似同分边界、空／零／负向量、scope隔离、生命周期、
跨连接提交、事务回滚、重启、缓存预算和缺少NumPy的路径。

离线回放工具读取已完成edited147运行的真实问题向量和检索trace，
逐题比较三路候选／分数、完整trace、最终20条事实、原文窗口和回答上下文。
旧实现来自冻结Git提交，所有可写访问只针对新目录内的SQLite副本；不打开模型客户端。

```powershell
D:\conda\envs\agent_aml\python.exe evals\benchmark_vector_scan.py --run-dir E:\AML\mini-vector\eval_outputs\edited147_dual_20261024_01 --output-dir E:\AML\mini-vector\eval_outputs\vector_scan_20261025_new
```

每次使用新的输出目录。计时包含本地三路检索、候选读取、融合及explain／trace，
不含嵌入API、原文选择、回答或裁判。冷缓存指解码缓存为空，不代表OS文件缓存为空；
开库和首次创建索引时间不计入查询。实际结果另存聚合报告，不将离线回放当作新LLM-J分数。

## 本轮之外

磁盘BLOB／float32迁移、近似向量索引、远程嵌入缓存、写事务缩短和服务并发仍待单独验证。
当前扫描复杂度仍随scope内向量数和维度线性增长，NumPy与缓存主要减少常数开销。
