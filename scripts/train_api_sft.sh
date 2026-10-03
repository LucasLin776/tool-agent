#!/usr/bin/env bash
set -euo pipefail

# API recommendation training: download ToolBench, build the candidate catalog,
# prepare recommendation examples, then submit the remote SFT job.
TOOLBENCH_SAMPLES="${TOOLBENCH_SAMPLES:-1000}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
SFT_DATA="${SFT_DATA:-data/processed/toolbench_sft.jsonl}"
API_CATALOG="${API_CATALOG:-data/processed/toolbench_api_catalog.jsonl}"
RECOMMENDATION_DATA="${RECOMMENDATION_DATA:-data/processed/api_recommendation_sft.jsonl}"

python -m tool_agent.data.download_toolbench --max-samples "$TOOLBENCH_SAMPLES" --output "$SFT_DATA"
python -m tool_agent.data.build_api_catalog --input "$SFT_DATA" --output "$API_CATALOG"
python -m tool_agent.data.build_recommendation_data --input "$SFT_DATA" --output "$RECOMMENDATION_DATA"
python -m tool_agent.training.sft \
  --data "$RECOMMENDATION_DATA" \
  --base-model "$BASE_MODEL" \
  --epochs "$SFT_EPOCHS" \
  --batch-size "$SFT_BATCH_SIZE" \
  --name "tool-agent-api-sft-${RUN_TAG}"

echo "API catalog: $API_CATALOG"
echo "SFT paths: artifacts/sft_paths.json"
