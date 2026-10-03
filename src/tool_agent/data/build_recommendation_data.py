from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tool_agent.api_recommender import (
    extract_api_names_from_text,
    extract_api_specs,
    is_terminal_api,
    recommendation_messages,
)


def first_user_message(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(row.get("user_query", ""))


def action_names(row: dict[str, Any], known: set[str]) -> list[str]:
    names: list[str] = []
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        for name in extract_api_names_from_text(content, known):
            if name not in names:
                names.append(name)
    return names


def normalize_row(row: dict[str, Any], max_candidates: int) -> dict[str, Any] | None:
    specs = [spec for spec in extract_api_specs(row.get("messages", [])) if not is_terminal_api(spec.name)]
    if not specs:
        return None
    by_name = {spec.name: spec for spec in specs}
    gold_names = [name for name in action_names(row, set(by_name)) if not is_terminal_api(name)]
    task = first_user_message(row)
    if not task or not gold_names:
        return None

    selected_names = list(gold_names)
    selected_names.extend(spec.name for spec in specs if spec.name not in selected_names)
    candidates = [by_name[name] for name in selected_names[:max_candidates]]
    candidate_names = {spec.name for spec in candidates}
    gold_names = [name for name in gold_names if name in candidate_names]
    if not gold_names:
        return None
    target = {
        "apis": [
            {"name": name, "purpose": by_name[name].description[:280], "order": index, "parameters": {}}
            for index, name in enumerate(gold_names, start=1)
        ],
        "notes": "Select the listed APIs in order; do not execute them.",
    }
    messages = recommendation_messages(task, candidates, compact=True)
    messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))})
    return {
        "task_id": row.get("task_id", ""),
        "source": "api_recommendation",
        "revision": row.get("revision"),
        "user_query": task,
        "gold_api_names": gold_names,
        "candidate_apis": [spec.as_dict() for spec in candidates],
        "messages": messages,
    }


def main(args: argparse.Namespace) -> None:
    accepted = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as source, args.output.open("w") as target:
        for line in source:
            if not line.strip():
                continue
            normalized = normalize_row(json.loads(line), args.max_candidates)
            if normalized is not None:
                target.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                accepted += 1
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"input": str(args.input), "accepted_samples": accepted, "max_candidates": args.max_candidates}, ensure_ascii=False, indent=2))
    print(f"recommendation_examples={accepted} output={args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ToolBench traces into API recommendation SFT examples")
    parser.add_argument("--input", type=Path, default=Path("data/processed/toolbench_sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/api_recommendation_sft.jsonl"))
    parser.add_argument("--max-candidates", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
