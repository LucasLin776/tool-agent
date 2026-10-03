#!/usr/bin/env bash
set -euo pipefail

: "${SFT_STATE_URI:?Set SFT_STATE_URI to the state_path from artifacts/sft_paths.json}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
RL_DATA="${RL_DATA:-data/processed/api_recommendation_sft.jsonl}"
RL_STEPS="${RL_STEPS:-50}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-2}"
RL_GROUP_SIZE="${RL_GROUP_SIZE:-4}"
RL_MAX_TOKENS="${RL_MAX_TOKENS:-384}"
RL_TEMPERATURE="${RL_TEMPERATURE:-0.8}"
RL_TOP_P="${RL_TOP_P:-0.95}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-5e-6}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

python -m tool_agent.training.rl \
  --data "$RL_DATA" \
  --base-model "$BASE_MODEL" \
  --resume "$SFT_STATE_URI" \
  --steps "$RL_STEPS" \
  --batch-size "$RL_BATCH_SIZE" \
  --group-size "$RL_GROUP_SIZE" \
  --max-tokens "$RL_MAX_TOKENS" \
  --temperature "$RL_TEMPERATURE" \
  --top-p "$RL_TOP_P" \
  --learning-rate "$RL_LEARNING_RATE" \
  --name "tool-agent-api-rl-${RUN_TAG}"

echo "RL paths: artifacts/api_rl_paths.json"
