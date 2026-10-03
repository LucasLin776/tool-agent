from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_recommendation_target(messages: list[dict], candidate_names: set[str]) -> list[str] | None:
    if not messages:
        return None
    try:
        value = json.loads(messages[-1].get("content", ""))
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("apis"), list):
        return None
    names: list[str] = []
    for item in value["apis"]:
        if not isinstance(item, dict) or item.get("name") not in candidate_names:
            return None
        if item["name"] in names:
            return None
        names.append(item["name"])
    return names


def validate_api_recommendation(path: Path) -> tuple[int, int]:
    total, valid = 0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        if row.get("source") != "api_recommendation":
            continue
        candidate_names = {
            item["name"]
            for item in row.get("candidate_apis", [])
            if isinstance(item, dict) and item.get("name")
        }
        if parse_recommendation_target(row.get("messages", []), candidate_names) is not None:
            valid += 1
    return total, valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate API recommendation SFT JSONL")
    parser.add_argument(
        "path", type=Path, default=Path("data/processed/api_recommendation_sft.jsonl"), nargs="?"
    )
    args = parser.parse_args()
    total, valid = validate_api_recommendation(args.path)
    print(json.dumps({"total": total, "valid": valid, "valid_rate": valid / max(total, 1)}, ensure_ascii=False))
    if valid != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
