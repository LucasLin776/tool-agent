#!/usr/bin/env bash
set -euo pipefail

python -m tool_agent.evaluation.run_api_eval "$@"
