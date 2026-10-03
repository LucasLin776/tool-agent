from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tool_agent.api_recommender import ApiRetriever, load_catalog, parse_api_recommendation, recommendation_messages
from tool_agent.training.common import encode_messages, require_pytrio


class RemoteApiRecommender:
    """Recommend ToolBench APIs without executing any tool."""

    def __init__(
        self,
        base_model: str,
        checkpoint: str | None,
        catalog_path: Path,
        default_top_k: int = 8,
    ) -> None:
        trio = require_pytrio()
        self.trio = trio
        service = trio.ServiceClient()
        self.sampler = service.create_sampling_client(base_model=base_model, model_path=checkpoint)
        self.tokenizer = self.sampler.get_tokenizer()
        self.retriever = ApiRetriever(load_catalog(catalog_path))
        self.default_top_k = default_top_k

    def recommend(self, task: str, top_k: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        candidates = self.retriever.search(task, top_k or self.default_top_k)
        messages = recommendation_messages(task, candidates)
        prompt = encode_messages(self.tokenizer, messages, add_generation_prompt=True)
        params = self.trio.SamplingParams(max_tokens=384, temperature=0.0, top_p=1.0)
        sequence = self.sampler.sample(
            prompt=self.trio.ModelInput.from_ints(prompt),
            sampling_params=params,
            num_samples=1,
        ).result().sequences[0]
        raw = getattr(sequence, "text", "") or ""
        return {
            "task": task,
            "apis": parse_api_recommendation(raw, candidates),
            "candidates": [item.as_dict() for item in candidates],
            "raw_completion": raw,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
