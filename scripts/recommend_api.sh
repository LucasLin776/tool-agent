#!/usr/bin/env bash
set -euo pipefail

: "${TASK:?Usage: TASK='describe your task' ./scripts/recommend_api.sh}"
python - <<'PY'
import os
from pathlib import Path

from tool_agent.serving.recommender import RemoteApiRecommender

agent = RemoteApiRecommender(
    base_model=os.getenv("TOOL_AGENT_BASE_MODEL", "Qwen/Qwen3.5-4B"),
    checkpoint=os.environ.get("TOOL_AGENT_CHECKPOINT"),
    catalog_path=Path(os.getenv("TOOL_AGENT_API_CATALOG", "data/processed/toolbench_api_catalog.jsonl")),
)
print(agent.recommend(os.environ["TASK"]))
PY
