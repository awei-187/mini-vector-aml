# 记忆侧 Query 与逐通路 LLM 重排

更新：2026-10-22。普通 `MiniVector` 和 AML 服务默认启用这两项能力。

## 写入与问题索引

1. `infer=True` 仍先从新消息抽取事实，并在当前 scope 内按精确文本哈希去重。
2. 对本次确需新增的记忆调用一次 LLM，逐条生成 1–3 个该记忆可回答的问题。问题必须独立成句，仅使用记忆中已知的信息，不猜答案、人物或事件。
3. 每个问题单独生成向量，保存在 `memory_queries(memory_id, position, query, embedding)`。事实文本及其原向量仍保存在 `memories`。
4. 单条记忆的正文、事实向量、问题向量、FTS 和 ADD 历史一起提交。问题结构非法或向量生成失败会明确报错，不把缺失索引伪装成成功。

示例：`Alice cycles to work.` → `How does Alice commute to work?`。搜索问题与生成问题向量匹配，但返回的 `memory` / AML `content` 仍是原事实。

生成结果按短 ID 对应原记忆，校验未知/重复/缺失 ID、空数组、重复问题、非法元素和超过数量上限。不拼接多问题后生成一个向量。精确重复事实不重复生成问题。`infer=False` 跳过事实抽取，但默认仍对原始记忆生成问题；只有关闭问题功能时才不调用此 LLM。

## 三路检索

| 通路 | 召回对象 | 阈值 |
| --- | --- | --- |
| `semantic` | 原记忆向量 | cosine ≥ `threshold` |
| `memory_query` | 记忆侧生成问题向量 | 每条记忆取问题 cosine 最大值，再应用同一阈值 |
| `bm25` | 原记忆文本的 FTS5 BM25 | 独立召回，不受事实向量阈值限制 |

搜索时保留原始用户问题，仅生成一次查询向量，不在线改写用户问题。三路均按相同 scope 隔离，各取最多 `max(top_k * 4, 60)` 条记忆候选。问题路先按 memory ID 取最大分并去重，再截断候选数量，避免一条记忆的多个问题挤占名额。

每路候选分别进入同一个 LLM listwise reranker。默认仅重排该路前 **30** 条（`rerank_limit`），其余按原检索顺序追加。候选不足两条时跳过模型。三路顺序执行，每次 Search 最多增加 3 次重排调用。

提示词输入原始问题与原记忆文本；模型只返回短 ID 排列，不生成最终答案、不修改证据。响应必须为完整排列；空、重复、未知、缺失 ID、非法 JSON 或调用异常都使该路回退原顺序，其他路继续。研究 trace 记录回退状态和异常类型，不记录上游异常正文。

各路重排后按 memory ID 去重，以等权 RRF 融合：

```text
rrf(memory) = sum(1 / (60 + rank_in_channel))
score = rrf(memory) / (number_of_nonempty_channels / 61)
```

最后取全局 `top_k`。`score` 在 `[0,1]` 内，是查询内部的融合排序分数，不是 LLM 置信度，也不是原始 cosine。同分按 memory ID 排序。BM25 独有候选现在可以进入最终结果。

## 配置与消融

```python
from mini_vector import MiniVector

memory = MiniVector(
    memory_query_enabled=True,
    queries_per_memory=3,
    rerank_mode="llm",
    rerank_limit=30,
)
```

`queries_per_memory` 和 `rerank_limit` 均须为正整数。普通库使用既有 `llm` / `embedder` 注入点；默认模型来源未变。AML 服务的生成问题、抽取、重排统一使用既有 `gpt-4o-mini` 聊天客户端，问题与事实向量统一走独立 Qwen `text-embedding-v4` 客户端。

| 配置 | 行为 |
| --- | --- |
| 默认 | 生成问题 + 三路独立召回 + 逐路 LLM 重排 + RRF |
| `rerank_mode="none"` | 保留问题通路，关闭 LLM 重排，以原检索排名做 RRF |
| `memory_query_enabled=False` | 不生成或搜索问题；原记忆向量和 BM25 两路分别重排后 RRF |
| 两项都关闭 | 使用旧版“向量候选 + BM25 加分”的检索算法，保留历史基线语义 |

独立 BM25 召回和 RRF 也是相对旧基线的结构变化；两项功能开关的比较不能单独证明某一种改进的因果收益。

## 现有数据库、更新与删除

普通库打开旧 SQLite 时只添加缺失的索引表，不自动调用模型重建旧记忆的问题。旧事实依然可由向量和 BM25 检索。确需为旧库补建问题时，使用原向量空间兼容的 embedder，显式按 scope 执行：

```python
memory.backfill_memory_queries(filters={"user_id": "alice"})
```

该方法逐条提交，已有问题的记忆会跳过，可在异常后再次调用；不会改变事实文本、事实向量或历史记录。回填可能产生大量模型调用。本次实现没有对任何现有实验库执行回填。

文本更新时重新生成问题及向量，并与正文一起替换；只更新 metadata 保留问题索引。关闭问题功能后更新文本会清除旧问题，避免过期匹配。`delete` / `delete_all` 同时清理问题索引。

AML 数据库身份现在绑定检索版本和这些选项。旧版 AML 数据库或不同检索配置不能直接复用，需要新的 `--db-path`，原库保留。

## Trace、评测与参考实现

增强检索的 `research_trace` 为 schema 2，记录三路原始候选、阈值后 ID、逐路重排输入/输出、耗时、成功或回退状态、RRF 顺序及实际返回结果。`explain=True` 返回原向量/BM25/问题向量分数、通路排名和 RRF 分数。旧基线路径仍使用 schema 1。

LoCoMo runner 默认评测增强模式；`--baseline` 显式选择旧算法。新运行记录生成问题与重排提示源码哈希，并导出 `memory_queries_embeddings.json`。请使用新的运行目录，不能对历史运行绕过源码哈希校验续跑。该 runner 仍沿用历史 `text-embedding-3-small` 配置，不能把它描述成 AML 指定模型的正式比赛成绩。

参考本地既有代码的两处模式：

- [query_contract.py](../query-rewrite-research/query_contract.py)：独立有序 query 数组、非空与去重校验；原项目方向是用户问题改写，本次实现的是记忆侧生成问题。
- [mini-hindsight memory.py](../mini-hindsight/mini_hindsight/memory.py) 与 [prompts.py](../mini-hindsight/mini_hindsight/prompts.py)：短 ID 的有界 listwise 重排、异常回退与 RRF。这里应用到每条检索通路，并将响应校验收紧为完整排列。

运行时不导入上述研究项目，不读取已保存的 benchmark 问题、金标或评测结果。

2026-10-22 验证：`agent_aml` 环境中全部 **50 项离线测试通过**（新增 16 项）。使用 SQLite + 注入模型，覆盖问题侧独有召回、BM25 独有召回、逐通路重排与融合、非法输出、隔离、生命周期、旧库补建、AML 原子性与配置保护。真实模型的效果提升、延迟与成本尚未在本次改动后重新测量。
