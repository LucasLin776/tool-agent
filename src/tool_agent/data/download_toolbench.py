"""Download a pinned ToolBench revision and normalize API recommendation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO = "tuandunghcmut/toolbench-v1"
REVISION = "36de9b189753ad5de276181974f97df15e8c3202"


def _as_pairs(conversations: Any) -> list[tuple[str, str]]:
    if isinstance(conversations, list):
        pairs = []
        for item in conversations:
            if isinstance(item, dict):
                pairs.append((str(item.get("from", "")), str(item.get("value", ""))))
        return pairs
    if isinstance(conversations, dict):
        roles = conversations.get("from", [])
        values = conversations.get("value", [])
        return [(str(role), str(value)) for role, value in zip(roles, values, strict=False)]
    return []


def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    messages = []
    role_map = {"system": "system", "user": "user", "assistant": "assistant", "function": "tool", "tool": "tool"}
    for source_role, content in _as_pairs(row.get("conversations")):
        role = role_map.get(source_role.lower())
        if role and content.strip():
            messages.append({"role": role, "content": content.strip()})
    if not messages or not any(message["role"] == "user" for message in messages):
        return None
    if messages[-1]["role"] != "assistant":
        # SFT should end at a model response, not a tool observation.
        assistant_indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        if not assistant_indices:
            return None
        messages = messages[: assistant_indices[-1] + 1]
    return {
        "task_id": str(row.get("id", f"toolbench-{index}")),
        "source": "toolbench",
        "revision": REVISION,
        "messages": messages,
    }


def main(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    dataset = load_dataset(REPO, split=args.split, revision=REVISION)
    if args.max_samples > 0:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.max_samples, len(dataset))))
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with output.open("w") as file:
        for index, row in enumerate(tqdm(dataset, desc="Normalize ToolBench")):
            normalized = normalize_row(row, index)
            if normalized is None:
                continue
            file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            accepted += 1
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"dataset": REPO, "revision": REVISION, "split": args.split, "requested_samples": args.max_samples, "accepted_samples": accepted, "license": "apache-2.0"}, ensure_ascii=False, indent=2))
    print(f"dataset={REPO} revision={REVISION} accepted={accepted} output={output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize pinned ToolBench")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--max-samples", type=int, default=5000, help="0 means all rows; default keeps the first experiment affordable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/processed/toolbench_sft.jsonl"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
