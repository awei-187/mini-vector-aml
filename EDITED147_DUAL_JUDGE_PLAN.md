# 最新优化版：修改版147题与GPT／JEV双四分类

目标：测试已固定的“时间解释修正＋原文补充”版本，分别使用现有GPT四分类与
JEV四分类评估同一批新答案。本轮不新增其他优化或比较条件，不按新评分回调参数。

## 数据与范围

- 数据文件：`E:/LLM judge/locomo_first_conversation_edited.json`。
- 数据SHA-256：`e7c79b10f81e26fc08ea5da246cb86cfb6c5fbe62920e7d947080434fc0dc94d`。
- conv-26，19个会话、419条话语、当前147题，全部题均为category 1–4。
- 相对之前原始数据，D4:1、D7:8、D10:8、D16:16的正文发生变化，因此完整重建一次库，
  不把原版1,711条记忆的运行身份套用到新数据。
- 这里的“新数据集”是本项目已采用的修改版第一组，不是LongMemEval或独立未见会话；
  本轮结果不能证明跨基准泛化。

模型与检索维持已测方案：gpt-4o-mini抽取／生成记忆问题／回答，
text-embedding-3-small嵌入，最多3个问题每记忆，事实向量／问题向量／独立BM25三路RRF，
关闭LLM重排，top_k=20、threshold=0.1。
回答使用已固定的observation_time语义；原文最多4个同session相邻窗口，前后各1条，
去除重叠，原文最多12,000字符。金标只用于本地对齐和最后的裁判阶段。

## 两套裁判与LLM-J口径

| 裁判 | 模型与接口 | 规则来源 | 输出 |
| --- | --- | --- | --- |
| GPT | 现有api.relayrouter.ai，gpt-4o-mini，temperature=0 | `mem0_locomo_dual_eval/prompts/answer_direct.txt`与DirectAnswerResult schema，原answer_coverage_v3逻辑 | correct／partial／missing／incorrect及简短中文理由 |
| JEV | 现有openrouter.ai/api/v1/systemone，typesafe/jev-1.13，响应固定typesafe/jev-1.13-20260917 | 已有prepared/rubric.json中的System One Choice | 四类编码、完整概率分布与confidence；不生成文字理由 |

两边使用完全相同的query、原始gold_answer和完整candidate_answer，
都不读取检索记忆、来源、方法名、历史标签或另一裁判的结果。
规则保留“错误优先；无错误时区分全覆盖、部分覆盖和零覆盖”。
主LLM-J比例定义为`correct / 147`；同时报告四类计数和比例、分类别结果、
两裁判一致率及混淆矩阵，不把partial加权为任意分数，不把JEV编码当连续数值。

JEV使用的Choice选项、概率和confidence结构参照
[TypeSafe官方Choice文档](https://docs.typesafe.ai/primitives/choice)。
协议解析或供应商调用失败不记为incorrect；实际模型版本不符时停止，不能自动替换模型。

## 授权与执行

用户已在本对话明确授权完整重建、147份回答、GPT与JEV两套判分、少量协议预检、
对应数据外发及新增费用。密钥从本地既有配置读取，不进入源码、日志或公开产物。

默认命令只做离线身份准备：

```powershell
& 'D:\conda\envs\agent_aml\python.exe' 'E:\AML\mini-vector\evals\run_edited_dual.py' --run-dir 'E:\AML\mini-vector\eval_outputs\edited147_dual_20261024_01'
```

加入`--run-models`才会实际调用模型。可见窗口启动器为`evals/run_edited_dual.ps1`，
需要显式传入`-RunModels`。同一目录仅可在数据、源码、提示、服务配置不变时恢复。
每条话语完整事务提交；检索、回答、两个裁判分别保存检查点。
已成功但尚未写入逐题结果的GPT／JEV响应可恢复利用，避免重复收费。

## 核验与当前状态

75项离线测试通过。注入客户端的全流程演练完成419条话语、147份回答和双裁判，
独立核验向量池／BM25／RRF、原文保真、来源边界、实际模型消息隔离和同答案双判分，
恢复新增调用为0，真实网络请求为0；模拟标签不是正式成绩。

真实运行先以少量合成数据预检记忆、嵌入、GPT分类和JEV固定版本，成功后才完整建库。
最终要求run_status complete、147题均有两个合法裁判结果、process_exit_code 0、
verification_exit_code 0以及verification.json passed；不以“调用返回成功”替代完整验收。

完整审计保存在新的eval_outputs目录，原数据及既有结果不覆盖。
最终公开仅保存聚合指标与方法记录；原始问答、gold、记忆、模型响应日志、数据库不上传GitHub。
不得将原始152题宽松裁判的140/152与本轮147题四分类直接比较成算法升降。
