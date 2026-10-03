# 数据目录

项目使用固定 revision 的 `tuandunghcmut/toolbench-v1` 训练集。

- `processed/toolbench_sft.jsonl`：下载后的规范化对话，保留系统中的 API 定义和用户、助手轨迹。
- `processed/toolbench_api_catalog.jsonl`：从规范化对话中去重得到的 API 候选目录。
- `processed/api_recommendation_sft.jsonl`：用于 SFT 的 API 推荐样本，助手目标是严格 JSON。
- 对应的 `.manifest.json`：记录数据集、revision、请求数量和实际写入数量。

重新生成 1000 条数据：

```bash
./scripts/train_api_sft.sh
```

训练数据只用于学习 API 选择、用途说明和参数规划。服务阶段会把用户任务与候选目录交给模型，返回推荐结果，不执行候选 API。
