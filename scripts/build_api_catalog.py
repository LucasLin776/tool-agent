#!/usr/bin/env bash
set -euo pipefail

python -m tool_agent.data.build_api_catalog "$@"
