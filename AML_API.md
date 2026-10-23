# mini-vector 的 AML 文本赛道接口

本服务包装 `MiniVector.add(infer=True)` 和 `search(threshold=0.1)`。默认启用记忆侧问题索引、原记忆向量/问题向量/BM25 三路召回、逐路 LLM 重排与 RRF；只返回原记忆证据。完整行为见 [RETRIEVAL.md](RETRIEVAL.md)。协议依据：https://agentmemoryleaderboard.ai/api-guide 。

## 当前验证状态（2026-10-22）

- 本次增强版全部 **50 项离线测试通过**，使用注入模型，覆盖问题索引、逐路重排、事务、隔离和配置保护；尚未重新执行真实模型 HTTP 联调或效果评测。
- 以下真实 HTTP 验证记录属于此次检索升级之前的版本，不能视为增强版模型效果或真实调用已验证。
- 升级前现有回归、HTTP/SQLite、Qwen 配置和客户端隔离测试共 **34 项通过**。
- 实际 Uvicorn localhost 服务启动、Health、鉴权及格式校验通过，未调用模型，见 `runtime/aml/transport_validation.json`。
- **真实本地 HTTP 联调已通过**：LLM 使用原服务的 `gpt-4o-mini`，Embedding 使用 Qwen 服务的 `text-embedding-v4`（1024 维）。Health、鉴权、Add、Search、用户隔离、重复请求、重启后检索和重放全部通过；返回 3 条合成记忆，临时数据库已删除。
- 先前将 Embedding 发往 LLM 服务产生的 404 报告另存为 `runtime/aml/local_validation_shared_provider_404.json`。现已按用户指定，读取 `E:\.env` 中的 Qwen 配置，没有替换模型，也没有复制凭据。
- 脱敏验证报告：`runtime/aml/local_validation.json`。报告不保存服务地址、凭据、问题、候选记忆或上游原始报错。
- 本地接口接通的验收条件已满足。本次不部署公网、不申请 Eval Key、不运行官方 Smoke 或 Full，不产生比赛成绩。

## 安装与启动

使用现有 `agent_aml` 环境。进入项目目录：

```powershell
cd E:\AML\mini-vector
```

安装 API 可选依赖（已在本次实施中安装）：

```powershell
& 'D:\conda\envs\agent_aml\python.exe' -m pip install -e '.[api]'
```

LLM 使用既有 `OPENAI_BASE_URL`、`OPENAI_API_KEY`，沿用原有 `.env` 查找方式。Embedding 使用单独的 Qwen 客户端，从 `E:\.env` 读取配置；该文件中的 `OPENAI_*` 字段不会覆盖 LLM 配置。两个客户端分别绑定服务地址和密钥，不共享认证。比赛服务显式固定 `gpt-4o-mini`、`text-embedding-v4`，忽略普通库的模型默认值。工作区根 `.env`、`E:\.env` 和原有实验配置未修改。

启动前在当前 PowerShell 设置接口鉴权令牌。下列命令生成随机令牌并仅存入本会话环境；后续请求使用同一变量。若需要跨会话使用，按自己的密钥管理方式安全保存，不能每次重启都重新生成并忘记同步调用方。

```powershell
$env:AML_API_TOKEN = & 'D:\conda\envs\agent_aml\python.exe' -c 'import secrets; print(secrets.token_urlsafe(32))'
```

启动服务，前台运行：

```powershell
& 'D:\conda\envs\agent_aml\python.exe' -m mini_vector.aml_api
```

若需要独立窗口观察长期运行，可在设置令牌后启动可见 PowerShell（继承当前环境）：

```powershell
Start-Process -FilePath 'powershell.exe' -WindowStyle Normal -ArgumentList @('-NoExit', '-Command', "& 'D:\conda\envs\agent_aml\python.exe' -m mini_vector.aml_api") -WorkingDirectory 'E:\AML\mini-vector'
```

默认地址 `http://127.0.0.1:8000`，默认数据库 `E:\AML\mini-vector\runtime\aml\memory.sqlite3`，不受当前工作目录影响。可用 `--host`、`--port`、`--db-path` 修改；`AML_DB_PATH` 也可指定数据库，优先级为命令行路径 > 环境变量 > `.env` > 默认。令牌优先使用环境变量，回退到 `.env` 的 `AML_API_TOKEN`；为空则拒绝启动。

本次检索配置改变了 AML 数据库身份。升级旧服务时须指定新的数据库文件，例如：

```powershell
& 'D:\conda\envs\agent_aml\python.exe' -m mini_vector.aml_api --db-path 'E:\AML\mini-vector\runtime\aml-enhanced\memory.sqlite3'
```

可选参数：`--disable-memory-queries` 关闭问题索引；`--rerank-mode none` 关闭 LLM 重排；两者同时使用时走旧基线检索。`--queries-per-memory 3` 设置问题数量上限，`--rerank-limit 30` 设置逐路重排的头部大小。各配置绑定数据库身份，配置变化需新数据库路径。

问题生成与重排也使用 `gpt-4o-mini`；所有向量继续由 Qwen `text-embedding-v4` 客户端提供。Search 现在最多增加三次 LLM 重排调用；每路异常回退原排序。生成问题或问题 embedding 失败则 Add 失败并回滚，不能返回虚假写入成功。

普通模型配置保留原有 `setting()` 行为：显式服务模型固定值优先；`OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 从当前目录、项目目录、工作区根目录的 `.env` 依次查找，再回退到环境变量。不要将根 `.env` 当成启动脚本复制或输出。

### Qwen Embedding 配置

比赛服务与真实验证脚本默认只读 `E:\.env`。可通过环境变量 `AML_EMBEDDING_ENV_FILE` 指向其他位置，也支持在原有配置查找路径的 `.env` 中设置该指针；环境变量优先。外部文件不会被整体导入进程环境。

- 模型：`QWEN_EMBEDDING_MODEL` 必须为 `text-embedding-v4`；未声明时使用该固定值。
- 地址：优先使用外部文件的 `QWEN_EMBEDDING_BASE_URL`。未设置时，读取 `QWEN_RERANK_ENDPOINT`，保留同一 DashScope/百炼业务空间的协议与主机，将路径改为 `/compatible-mode/v1`。不会向 rerank 路径发送向量请求。
- 密钥变量名：优先读取 `QWEN_EMBEDDING_API_KEY_ENV`，其次 `QWEN_RERANK_API_KEY_ENV`，默认 `DASHSCOPE_API_KEY`。密钥值优先取同名进程环境变量，否则读取选定外部文件；显式空环境值阻止回退。缺失时明确失败，不回退到 LLM 的密钥。
- 请求固定 `dimensions=1024`、`encoding_format=float`，按每次最多 10 条文本拆批，按返回索引恢复输入顺序，并验证数量与维度。依据 [阿里云 Embedding 官方文档](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)。

如部署机器上的配置文件位置不同，在启动或验证前设置：

```powershell
$env:AML_EMBEDDING_ENV_FILE = 'E:\.env'
```

配置文件只引用本机文件，不需要把凭据复制进项目。若已有使用旧服务配置标记的比赛数据库，应换用新的 `--db-path`；服务不会自动改写旧库或重建已有向量。

## 请求与响应

`GET /health` 无需鉴权，返回 `{"status":"ok"}`。它检查应用已初始化，不调用模型，也不证明上游模型可用。

```powershell
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/health'
```

`POST /add` 必填 `request_id`、`user_id`、`session_id`、非空 `messages`。每条消息的 `role` 为 `user` 或 `assistant`，`content` 为非空文本，可带整数 Unix 毫秒 `timestamp`。输入顺序保留；时间戳作为消息来源时间进入抽取上下文，不作为事件时间或内部创建时间。

在保留 `AML_API_TOKEN` 的原 PowerShell 中写入一条合成记忆（需上游支持指定模型）：

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/add' -Headers @{Authorization="Bearer $env:AML_API_TOKEN"} -ContentType 'application/json; charset=utf-8' -Body (@{request_id='local-demo-1';user_id='local-demo-user';session_id='local-demo-session';messages=@(@{role='user';content='My name is Vera Lin. My project code is LANTERN-471.';timestamp=1767225600000})} | ConvertTo-Json -Depth 6 -Compress)
```

成功返回：

```json
{"success":true,"request_id":"local-demo-1","user_id":"local-demo-user","session_id":"local-demo-session"}
```

`POST /search` 必填 `query`、`user_id`、正整数 `top_k`，可带选择题 `options` 字符串数组。选项在本阶段不参与检索或生成。正式 AML 传入 `top_k=100`，服务始终返回不超过请求上限的条目，不人为补足数量。

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/search' -Headers @{Authorization="Bearer $env:AML_API_TOKEN"} -ContentType 'application/json; charset=utf-8' -Body (@{query="What is Vera Lin's project code?";user_id='local-demo-user';top_k=100} | ConvertTo-Json -Compress)
```

响应仅包含按检索分数排序的 `data`，每条有 `id`、`content`、`score`、`created_at`；空结果为 `{"data":[]}`。`created_at` 为记忆入库时间；源消息时间保存在内部上下文与来源元数据，不伪装成事件时间。Search 不按 `session_id` 过滤。同一 `user_id` 的不同会话可以共同检索，所有标识符原样保留，首尾空白也参与用户隔离。

## 幂等、故障和容量

- 同一 `request_id`、相同规范化请求体：返回已保存的成功响应，不再次调用模型；JSON 对象键顺序不影响摘要，消息数组顺序影响摘要。可选时间戳缺省与 `null` 视为相同。
- 同一 `request_id`、不同用户/会话/内容：返回 409。请求 ID 在整个服务数据库内唯一。
- 一批 Add 的记忆、事实/问题向量、FTS、历史、近期上下文和幂等记录使用一个 SQLite 事务；错误或进程中断时未提交写入回滚，重试使用原请求 ID。上游已经发生的调用费用无法回滚。
- 一个数据库只允许一个服务进程持有锁；单 worker，Add/Search 串行，建议申报 Add 并发 1、Search 并发 1。Health 不等待模型操作锁。增加 worker 不会提高该版本吞吐。
- 422 为格式错误，401 为本服务鉴权失败，409 为请求 ID 冲突，503 为上游连接/HTTP 错误，504 为上游超时，500 为内部错误。上游拒绝指定模型同样是失败，不能当作空结果成功。HTTP 错误正文和日志不转发上游原始报错。
- SDK 单次调用超时 90 秒、内部自动重试关闭，由调用方按协议决定重试；队列等待与一批内多次模型调用会增加整次请求耗时。
- 数据库标记绑定模型名、LLM 与 Embedding 两个服务地址的独立摘要、指定向量维度、接口版本与检索配置，同时检查实际向量维度。拒绝接管没有 AML 标记的旧数据库，拒绝在配置不匹配时继续写入；更换配置应使用新的数据库路径。
- `runtime/`、数据库与 `.env` 均被忽略。日志只保留操作名、状态和耗时。评测数据及派生副本须按官方要求在任务结束后 30 天内清理；该版本不自动计时删除数据。清理应停服务后覆盖数据库、WAL/SHM、上下文、历史及任何自行保存的副本。

## 验证

离线回归，全部模型使用注入替身：

```powershell
& 'D:\conda\envs\agent_aml\python.exe' -m unittest discover -s tests -v
```

真实验证会产生少量模型调用费用。使用合成数据和随机接口令牌，启动临时 localhost HTTP 服务，执行模型预检、Health、鉴权、Add、Search、隔离、重试和重启恢复验证；进程停止后删除临时数据库，仅留下脱敏结果。它不会接触默认正式数据库，也不会执行 Answer 或评分：

```powershell
& 'D:\conda\envs\agent_aml\python.exe' scripts\verify_aml_api.py --live
```

默认报告为 `runtime/aml/local_validation.json`；`--report` 可指定另一份报告。重新联调增强版时应选择新报告路径以保留旧记录。只有退出码 0、`status=passed` 且 `temporary_data_deleted=true` 才视为完整通过。若指定模型请求被拒绝，保留 HTTP 状态和失败阶段，不回退模型。已有报告为升级前版本，使用 Qwen Embedding 与原 LLM 服务。

## 后续公网部署

1. 在部署机器安装相同源码和 API 依赖，分别配置 LLM 服务与 Qwen Embedding 配置文件路径、独立数据库及强随机 `AML_API_TOKEN`；保留固定代码版本和运行配置。部署机器应自行配置密钥，不应复制整个共享 `.env`。
2. 将服务置于 HTTPS 反向代理后，公开 `/add`、`/search`、`/health`。代理连接与读取超时应覆盖官方允许的最长请求时间；避免代理、SDK 和平台重试相互叠加。
3. 若代理位于另一台机器，显式设置 `--host 0.0.0.0`，同时配置访问限制。不要向公网直接暴露数据库或模型密钥，不开启请求正文日志。
4. 提交完整 Add/Search URL、Bearer 鉴权方式、接口专用令牌、容量与运行限制。Memory System Key 是本服务令牌，Eval Key 由 AML 签发，两者不同。
5. 当前版本仍需针对正式数据规模做容量验证。SQLite 对用户范围内向量做精确扫描，不保证完整赛道的吞吐或总耗时；本地测试通过不代表官方 Smoke 通过或具备 Full 容量。

公网部署、平台申请、官方 Smoke 与 Full 属于后续工作，不包含在本次本地实现内。
