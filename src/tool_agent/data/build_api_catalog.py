from __future__ import annotations

import argparse
from pathlib import Path

from tool_agent.api_recommender import build_api_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an API catalog from normalized ToolBench conversations")
    parser.add_argument("--input", type=Path, default=Path("data/processed/toolbench_sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/toolbench_api_catalog.jsonl"))
    args = parser.parse_args()
    count = build_api_catalog(args.input, args.output)
    print(f"apis={count} output={args.output}")


if __name__ == "__main__":
    main()
