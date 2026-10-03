from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tool_agent.api_recommender import (
    ApiRetriever,
    ApiSpec,
    load_catalog,
    parse_api_recommendation,
    recommendation_messages,
)
from tool_agent.data.build_recommendation_data import normalize_row
from tool_agent.training.common import encode_messages, load_jsonl, require_pytrio


def _specs_from_dicts(values: list[dict[str, Any]]) -> list[ApiSpec]:
    return [
        ApiSpec(
            name=str(value["name"]),
            description=str(value.get("description", "")),
            parameters=value.get("parameters", {}) if isinstance(value.get("parameters", {}), dict) else {},
            tool=value.get("tool"),
        )
        for value in values
        if isinstance(value, dict) and value.get("name")
    ]


def prepare_row(row: dict[str, Any], max_candidates: int) -> tuple[str, list[ApiSpec], list[str]] | None:
    normalized = normalize_row(row, max_candidates)
    if normalized is None:
        return None
    return (
        str(normalized["user_query"]),
        _specs_from_dicts(normalized["candidate_apis"]),
        [str(name) for name in normalized["gold_api_names"]],
    )


def main(args: argparse.Namespace) -> None:
    trio = require_pytrio()
    rows = load_jsonl(args.data)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    retriever = ApiRetriever(load_catalog(args.catalog)) if args.candidate_mode == "global" else None
    tokenizer = None
    service = trio.ServiceClient()
    sampler = service.create_sampling_client(base_model=args.base_model, model_path=args.checkpoint)
    tokenizer = sampler.get_tokenizer()
    records = []
    for row in rows:
        if args.candidate_mode == "row":
            prepared = prepare_row(row, args.max_candidates)
            if prepared is None:
                continue
            task, candidates, gold = prepared
        else:
            assert retriever is not None
            task = next(
                (str(message.get("content", "")) for message in row.get("messages", []) if message.get("role") == "user"),
                str(row.get("user_query", "")),
            )
            candidates = retriever.search(task, args.top_k)
            known = {spec.name for spec in retriever.specs}
            gold = []
            for message in row.get("messages", []):
                if message.get("role") != "assistant":
                    continue
                for spec in candidates:
                    if spec.name in str(message.get("content", "")) and spec.name not in gold:
                        gold.append(spec.name)
            if not task or not gold:
                continue

        prompt = encode_messages(
            tokenizer,
            recommendation_messages(task, candidates, compact=True),
            add_generation_prompt=True,
        )
        response = sampler.sample(
            prompt=trio.ModelInput.from_ints(prompt),
            sampling_params=trio.SamplingParams(max_tokens=384, temperature=0.0, top_p=1.0),
            num_samples=1,
        ).result().sequences[0]
        raw = getattr(response, "text", "") or ""
        predicted = [item["name"] for item in parse_api_recommendation(raw, candidates)]
        overlap = set(gold).intersection(predicted)
        records.append(
            {
                "task": task,
                "gold": gold,
                "predicted": predicted,
                "completion": raw,
                "hit": bool(overlap),
                "recall": len(overlap) / len(gold),
            }
        )
    count = len(records)
    metrics = {
        "examples": count,
        "hit_rate": sum(item["hit"] for item in records) / max(count, 1),
        "mean_recall": sum(item["recall"] for item in records) / max(count, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "candidate_mode": args.candidate_mode,
                "metrics": metrics,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate API recommendation against ToolBench traces")
    parser.add_argument("--data", type=Path, default=Path("data/processed/toolbench_sft.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/toolbench_api_catalog.jsonl"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--candidate-mode", choices=["row", "global"], default="row")
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("reports/api_recommendation.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
