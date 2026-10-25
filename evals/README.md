# mini-vector LoCoMo 第一组正式评测

2026-10-22：runner 默认使用增强检索（记忆侧问题 + 三路召回 + 逐路 LLM 重排 + RRF）。`run_locomo.py --baseline` 或 PowerShell 启动器的 `-Baseline` 可选择旧基线。下列“实验口径”描述的是 2026-10-21 已完成的旧版实验；新实验必须使用新目录，不能对历史运行绕过源码哈希检查续跑。生成问题与重排均有额外模型调用成本。

2026-10-23：使用用户提供的完整新抽取提示词和增强检索完成重测，**135/152（88.82%）**，1,759 条记忆、4,920 条记忆侧问题，独立验证通过。419 次抽取请求均核对为完整新提示词及原始 session 日期。三路共 456 次重排全部因模型未返回候选 ID 完整排列而回退，实际结果为三路原排序经 RRF 融合。相比旧版 126/152 增加 9 题、5.92 个百分点，但抽取与检索同时改变，不是仅提示词消融。见 [本次报告](E:/AML/mini-vector/eval_outputs/locomo_conv26_newprompt_enhanced_20261023_01/REPORT.md)。

增强版还会快照 `mini_vector/retrieval.py`，在运行身份中记录功能开关，并导出 `memory_queries_embeddings.json`；trace schema 2 记录每路重排及 RRF。向量模型仍为历史实验的 `text-embedding-3-small`，不是 AML 接口的 `text-embedding-v4`。

使用 `D:\conda\envs\agent_aml\python.exe`，无需 Mem0、Qdrant 或 AMB 依赖。

## 2026-10-21 旧版实验口径

- 原始数据：`E:\AML\graphiti\locomo_eval\data\locomo.json`，索引 0、`conv-26`。
- 摄取全部 19 个 session、419 条话语；按会话日期排序，逐条摄取，保留图片检索词和图片描述文本。
- 问题仅包含类别 1–4，共 152 题；类别 5 的 47 题排除。不是整个 LoCoMo 数据集评测。
- 摄取输入格式、speaker 到 user/assistant 角色的映射、回答上下文、回答提示、宽松裁判提示，直接复用 `mem0/locomo_eval/run_locomo.py` 中现有函数。
- 抽取、回答、裁判模型均为 `gpt-4o-mini`；向量模型为 `text-embedding-3-small`。密钥和服务地址由 mini-vector 既有配置加载器读取，不保存到实验产物。
- 使用真实 `MiniVector.add(infer=True)` 和 `MiniVector.search`。抽取不额外设置 temperature（保持库默认），回答/裁判 temperature=0。
- 原生精确余弦扫描 + SQLite FTS5 BM25，`top_k=20`、`threshold=0.1`、候选上限 80；无实体加分、无独立 reranker。
- 摄取和检索顺序执行，冻结记忆后最多四个普通线程并发回答/判分。
- 问题和金标不参与记忆摄取；回答模型只看到该题问题及实际返回记忆；金标仅供裁判使用。

## 运行与续跑

新实验使用独立目录，在可见 Windows PowerShell 窗口运行：

```powershell
Start-Process -FilePath 'powershell.exe' -WindowStyle Normal -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', 'E:\AML\mini-vector\evals\run_locomo.ps1', '-RunDir', 'E:\AML\mini-vector\eval_outputs\locomo_conv26_enhanced_20261022_01')
```

中断后在同一目录续跑（先确认原进程已结束；运行锁会拒绝重复启动）：

```powershell
Start-Process -FilePath 'powershell.exe' -WindowStyle Normal -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', 'E:\AML\mini-vector\evals\run_locomo.ps1', '-RunDir', 'E:\AML\mini-vector\eval_outputs\locomo_conv26_enhanced_20261022_01', '-Resume')
```

每次实验应使用新的目录，不能把新实验写入已有结果目录。续跑会校验数据、模型、参数及实现/提示源码的哈希。

## 产物与完成判定

- `run_metadata.json`：运行身份、模型、参数、数据和源码 SHA-256。
- `source_code/`：本次运行的 runner、mini-vector 源码及共享评测帮助函数快照。
- `input_conversation_0.json`、`source_snapshot.json`：实际输入和按时间排序的来源话语。
- `memory.sqlite3`：实际记忆、向量、FTS5、历史、近期消息及摄取完成标记。
- `ingestion/*.json`：逐条实际输入、抽取结果和耗时。
- `memory_snapshot.json`、`memories_embeddings.json`：冻结记忆及完整存储向量。
- `query_details/*.json`：每题检索候选、原始/归一化 BM25、阈值排除、综合排名、实际 top-k、回答上下文、答案和裁判。
- `api_calls/`：模型请求与响应（包括 token 用量）；不记录认证头、API key 或环境变量。live preflight 使用单独内存库，与正式记忆和计分分开。
- `final_results.json`、`summary.json`：152 题汇总和分类型正确率。
- `api_usage_summary.json`、`console.log`、`run_status.json`、`process_exit_code.txt`：调用统计、日志、阶段/进度和退出码。

仅当 `run_status.json` 为 `complete`、摄取 419/419、计分 152/152 且退出码为 0 时，视为完整正式运行。摄取通过包装 SQLite 事务，把一条话语的记忆、FTS、历史、近期消息和完成标记一起提交，异常时整体回滚；不改变核心抽取或检索算法。回答和判分分阶段落盘，续跑可以复用已保存答案。

完成后运行独立验证（只读取保存的模型结果和数据库，不调用模型）：

```powershell
& 'D:\conda\envs\agent_aml\python.exe' 'E:\AML\mini-vector\evals\verify_results.py' 'E:\AML\mini-vector\eval_outputs\locomo_conv26_20261021_01'
```

2026-10-21 首次正式运行已完成：**126/152，82.89%**，850 条记忆，见该运行目录的 `REPORT.md` 和 `verification.json`。运行曾因网络连接失败及本地状态文件 PermissionError 续跑，所有摄取和逐题判分均保留。完成后 runner 新增了 Windows 瞬时文件锁的原子写入重试；本次实验实际使用的旧源码仍原样保存在 `source_code/`，新源码适用于新实验，不能绕过已有运行的源码哈希校验。

## 解释结果时的边界

这是一组对话、一个模型配置下的单次 LLM 裁判评测。裁判沿用历史宽松标准，因此准确率不是严格逐字匹配或人工评估。

为复用既有回答口径，共享提示把 `session_date` 标记为 `event_time`；这个字段实际是对话日期，并不一定是事件发生日期。抽取事实里的时间文本也会进入上下文。此语义问题保留并记录在 metadata，避免把本次改变提示的结果混进历史对比。

`source_dialogue_ids` 记录产生新增记忆时输入的那一条话语；LLM 同时看过最近消息和相似记忆，这些 ID 不是经过独立验证的完整事实证据集合。`gold_origin_hit` 只表示返回记忆的摄取来源 ID 与题目 gold evidence 的交集，不能当作完整证据召回率。
# Dedicated Qwen reranker comparison (2026-10-25)

`run_qwen_ablation.py` prepares147 frozen RRF60 pools from the independently
verified edited147 run. It compares none/default-instruction/support-instruction,
top20, identical temporal/source answer flow, fresh answers and both four-class
judges. No ingestion or embeddings. Exact instructions and input/source hashes
are frozen before calls. Details: [predeclared protocol](../QWEN_RERANK_EXPERIMENT_20261025.md).
The default invocation is offline preparation; real calls need the explicitly
authorized `--run-models` switch. Credentials are read from the existing files,
never copied into artifacts. A failed Qwen ranking stops the benchmark rather
than being silently counted as a successful arm.

From the mini-vector folder, prepare without API calls:

```powershell
& 'D:\conda\envs\agent_aml\python.exe' evals/run_qwen_ablation.py --run-dir eval_outputs/qwen_instruct_147_20261025_01
```

After authorization, the PowerShell wrapper saves logs, exit status and runs the
independent verifier after successful completion:

```powershell
& .\evals\run_qwen_ablation.ps1 -RunDir 'E:\AML\mini-vector\eval_outputs\qwen_instruct_147_20261025_01' -RunModels
```

Run verification again without API calls:

```powershell
& 'D:\conda\envs\agent_aml\python.exe' evals/verify_qwen_ablation.py eval_outputs/qwen_instruct_147_20261025_01
```
