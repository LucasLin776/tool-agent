# ToolBench API 推荐后训练

这个项目把用户任务转换成 API 规划：模型从 ToolBench API 候选目录中选择所需 API，说明用途、调用顺序和参数。服务只返回推荐结果，不执行 API，也不需要本地工具沙箱。

训练使用 PyTrio 远程算力。代码负责下载和整理数据、构建 API 目录、生成 SFT 样本、提交训练并保存远程权重；实际训练由你启动。

## 环境

```bash
conda create -n post_training python=3.11 -y
conda activate post_training
cd /Users/shuoqinlin/tool-agent-post-training
pip install -e ".[dev,service]"
trio login
```

如果当前网络使用 SOCKS 代理，项目已经包含 `httpx[socks]`，可以直接重试 `trio login`。

## 下载数据并启动 SFT

默认取 ToolBench 训练集的 1000 条样本，下载只发生在数据文件层面，随后一次性生成目录和推荐样本：

```bash
chmod +x scripts/train_api_sft.sh
TOOLBENCH_SAMPLES=1000 ./scripts/train_api_sft.sh
```

脚本执行顺序：

1. 下载固定 revision 的 ToolBench 数据到 `data/processed/toolbench_sft.jsonl`。
2. 从系统消息提取 API 定义，写入 `data/processed/toolbench_api_catalog.jsonl`。
3. 把 API 轨迹转换为推荐格式，写入 `data/processed/api_recommendation_sft.jsonl`。
4. 调用 `tool_agent.training.sft`，将 SFT 作业提交到 PyTrio。

可用环境变量调整规模和训练参数：`TOOLBENCH_SAMPLES`、`BASE_MODEL`、`SFT_EPOCHS`、`SFT_BATCH_SIZE`、`RUN_TAG`。默认基础模型是 `Qwen/Qwen3.5-4B`。训练完成后，`artifacts/sft_paths.json` 保存 `sampler_path` 和 `state_path`。

数据准备也可以单独运行：

```bash
python -m tool_agent.data.download_toolbench --max-samples 1000 --output data/processed/toolbench_sft.jsonl
python -m tool_agent.data.build_api_catalog --input data/processed/toolbench_sft.jsonl --output data/processed/toolbench_api_catalog.jsonl
python -m tool_agent.data.build_recommendation_data --input data/processed/toolbench_sft.jsonl --output data/processed/api_recommendation_sft.jsonl
python -m tool_agent.data.validate data/processed/api_recommendation_sft.jsonl
```

## 可选的强化学习阶段

项目现在也提供了与 API 推荐目标匹配的在线 GRPO 风格强化学习。每个任务采样一组推荐结果，使用确定性奖励计算组内相对优势，再通过 PyTrio 的 `importance_sampling` 更新 LoRA。奖励由四部分组成：候选 API 命中 F1、API 顺序、严格 JSON 格式和候选目录约束。整个奖励过程只比较 API 名称和输出格式，不执行外部 API。

先完成 SFT，再把 SFT 的 `state_path` 作为 RL 的起点：

```bash
export SFT_STATE_URI="$(python -c 'import json; print(json.load(open("artifacts/sft_paths.json"))["state_path"])')"
chmod +x scripts/train_api_rl.sh
SFT_STATE_URI="$SFT_STATE_URI" RL_STEPS=50 RL_BATCH_SIZE=2 RL_GROUP_SIZE=4 ./scripts/train_api_rl.sh
```

RL 完成后，`artifacts/api_rl_paths.json` 中的 `sampler_path` 用于服务和评估，`state_path` 用于继续 RL。首次建议使用 `RL_STEPS=2 RL_BATCH_SIZE=1 RL_GROUP_SIZE=2` 做链路验证，再增加步数。采样数量、生成长度和 RL 步数会直接影响 PyTrio 用量。

## 启动推荐服务

训练结束后，把 PyTrio 返回的 sampler URI 放入环境变量。下面的写法会读取训练脚本生成的路径文件：

```bash
export TOOL_AGENT_BASE_MODEL=Qwen/Qwen3.5-4B
export TOOL_AGENT_CHECKPOINT="$(python -c 'import json; print(json.load(open("artifacts/sft_paths.json"))["sampler_path"])')"
export TOOL_AGENT_API_CATALOG=data/processed/toolbench_api_catalog.jsonl
uvicorn tool_agent.serving.api:app --host 127.0.0.1 --port 8000
```

发送任务：

```bash
curl -s http://127.0.0.1:8000/recommend \
  -H 'content-type: application/json' \
  -d '{"task":"帮我查明天上海到北京的航班并比较价格","top_k":8}'
```

返回值包含 `apis`、候选 API、模型原文和延迟。`/chat` 是同一推荐逻辑的兼容别名；推荐结果不会触发任何外部请求。

## 评估

```bash
python -m tool_agent.evaluation.run_api_eval \
  --base-model Qwen/Qwen3.5-4B \
  --checkpoint "$TOOL_AGENT_CHECKPOINT" \
  --catalog data/processed/toolbench_api_catalog.jsonl \
  --data data/processed/toolbench_sft.jsonl \
  --output reports/api_recommendation.json
```

默认 `candidate-mode=row`，会复现训练时每条任务自己的候选 API 集合，适合测量 SFT/RL 模型本身的推荐能力。若要单独测试线上全局目录检索，可以加上 `--candidate-mode global --top-k 8`。评估报告包括 API 命中率和平均召回率，只比较推荐名称与 ToolBench 轨迹中的 API，不执行网络 API。

## 项目结构

```text
src/tool_agent/api_recommender.py             API 抽取、BM25 检索、提示词和输出解析
src/tool_agent/data/download_toolbench.py     下载并规范化 ToolBench
src/tool_agent/data/build_api_catalog.py      构建 API 候选目录
src/tool_agent/data/build_recommendation_data.py 生成 SFT 推荐样本
src/tool_agent/training/sft.py                PyTrio LoRA SFT
src/tool_agent/training/rl.py                 API 推荐 GRPO 风格强化学习
src/tool_agent/serving/recommender.py         远程 sampler 推理
src/tool_agent/serving/api.py                 FastAPI 服务
src/tool_agent/evaluation/run_api_eval.py     API 推荐评估
```

运行本地检查：

```bash
pytest -q
python -m compileall -q src
```
