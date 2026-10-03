from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def require_pytrio() -> Any:
    try:
        import pytrio as trio
    except ImportError as exc:  # pragma: no cover - exercised only on a training machine
        raise RuntimeError("Install pytrio and set PYTRIO_API_KEY before remote training") from exc
    return trio


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/train_api_sft.sh or build the data files first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encode_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError("Chat template encoded to an empty sequence")
    return list(tokens)


def build_sft_datum(trio: Any, tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> Any:
    tokens = encode_messages(tokenizer, messages, add_generation_prompt=False)
    if len(tokens) > max_length:
        return None
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray(tokens[1:], dtype=np.int64),
            "weights": np.ones(len(tokens) - 1, dtype=np.float32),
        },
    )


def loss_per_token(result: Any, data: list[Any]) -> float:
    logprobs: list[float] = []
    weights: list[float] = []
    for output, datum in zip(result.loss_fn_outputs, data, strict=True):
        values = output["logprobs"]
        logprobs.extend(values.tolist() if hasattr(values, "tolist") else values)
        datum_weights = datum.loss_fn_inputs.get("weights")
        if datum_weights is None:
            weights.extend([1.0] * len(values))
        else:
            weights.extend(datum_weights.tolist() if hasattr(datum_weights, "tolist") else datum_weights)
    denom = max(float(np.sum(weights)), 1.0)
    return float(-np.dot(np.asarray(logprobs), np.asarray(weights)) / denom)


def save_path_json(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2))
