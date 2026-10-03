"""Online GRPO-style reinforcement learning for API recommendation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tool_agent.api_recommender import ApiSpec, parse_api_recommendation, recommendation_messages

from .common import encode_messages, load_jsonl, require_pytrio, save_path_json


@dataclass(frozen=True)
class Rollout:
    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float


def _strict_json(text: str) -> bool:
    try:
        value = json.loads(text.strip())
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict) and isinstance(value.get("apis"), list)


def recommendation_reward(text: str, candidates: list[ApiSpec], gold_names: list[str]) -> float:
    """Score one completion using only deterministic API-planning criteria."""
    parsed = parse_api_recommendation(text, candidates)
    predicted = [item["name"] for item in parsed]
    if not predicted or not gold_names:
        return 0.0

    gold = list(dict.fromkeys(gold_names))
    overlap = set(predicted).intersection(gold)
    precision = len(overlap) / len(predicted)
    recall = len(overlap) / len(gold)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    ordered_hits = sum(
        1 for index, name in enumerate(predicted) if index < len(gold) and name == gold[index]
    )
    order_score = ordered_hits / len(gold)
    format_score = 1.0 if _strict_json(text) else 0.0
    return float(0.65 * f1 + 0.25 * order_score + 0.10 * format_score)


def build_prompt_tokens(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], list[ApiSpec], list[str]]:
    candidates = [
        ApiSpec(
            name=str(item["name"]),
            description=str(item.get("description", "")),
            parameters=item.get("parameters", {}) if isinstance(item.get("parameters", {}), dict) else {},
            tool=item.get("tool"),
        )
        for item in row.get("candidate_apis", [])
        if isinstance(item, dict) and item.get("name")
    ]
    task = str(row.get("user_query") or "").strip()
    gold = [str(name) for name in row.get("gold_api_names", []) if name]
    if not task or not candidates or not gold:
        raise ValueError("Each RL row needs user_query, candidate_apis, and gold_api_names")
    messages = recommendation_messages(task, candidates, compact=True)
    return encode_messages(tokenizer, messages, add_generation_prompt=True), candidates, gold


def build_rl_datum(trio: Any, prompt_tokens: list[int], rollout: Rollout) -> Any:
    """Build the aligned Datum expected by PyTrio's importance_sampling loss."""
    if not rollout.tokens:
        raise ValueError("Cannot train on an empty completion")
    if len(rollout.tokens) != len(rollout.logprobs):
        raise ValueError("Rollout token/logprob lengths differ")
    observation_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + rollout.tokens[:-1]
    target_tokens = [0] * observation_len + rollout.tokens
    old_logprobs = [0.0] * observation_len + rollout.logprobs
    advantages = [0.0] * observation_len + [rollout.reward] * len(rollout.tokens)
    if not (len(input_tokens) == len(target_tokens) == len(old_logprobs) == len(advantages)):
        raise ValueError("RL datum fields must have the same token length")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )


def _sequence_text(tokenizer: Any, sequence: Any) -> str:
    text = getattr(sequence, "text", None)
    if text is not None:
        return str(text)
    return tokenizer.decode(list(getattr(sequence, "tokens", [])), skip_special_tokens=True)


def main(args: argparse.Namespace) -> None:
    trio = require_pytrio()
    rows = load_jsonl(args.data)
    if not rows:
        raise RuntimeError("No API recommendation rows found")
    service = trio.ServiceClient()
    if args.resume:
        client = service.create_training_client_from_state(args.resume)
    else:
        client = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed)
    tokenizer = client.get_tokenizer()
    rng = np.random.default_rng(args.seed)
    params = trio.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
    step = 0
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(rows), size=args.batch_size)
        sampler = client.save_weights_and_get_sampling_client()
        datums = []
        rollout_rewards: list[float] = []
        degenerate = 0
        updates = 0
        started = time.time()
        for index in indices:
            prompt_tokens, candidates, gold = build_prompt_tokens(tokenizer, rows[int(index)])
            result = sampler.sample(
                prompt=trio.ModelInput.from_ints(prompt_tokens),
                sampling_params=params,
                num_samples=args.group_size,
            ).result()
            rollouts: list[Rollout] = []
            for sequence in result.sequences:
                raw_tokens = getattr(sequence, "tokens", None)
                raw_logprobs = getattr(sequence, "logprobs", None)
                if not raw_tokens or raw_logprobs is None:
                    continue
                tokens = list(raw_tokens)
                logprobs = [float(value) for value in raw_logprobs]
                if len(tokens) != len(logprobs):
                    continue
                text = _sequence_text(tokenizer, sequence)
                rollouts.append(Rollout(tokens, logprobs, text, recommendation_reward(text, candidates, gold)))
            if not rollouts:
                continue
            rollout_rewards.extend(item.reward for item in rollouts)
            group_mean = float(np.mean([item.reward for item in rollouts]))
            advantages = [item.reward - group_mean for item in rollouts]
            if max(advantages, default=0.0) - min(advantages, default=0.0) < args.min_reward_spread:
                degenerate += 1
                continue
            for rollout, advantage in zip(rollouts, advantages, strict=True):
                datums.append(build_rl_datum(trio, prompt_tokens, Rollout(rollout.tokens, rollout.logprobs, rollout.text, advantage)))
        if datums:
            forward = client.forward_backward(datums, loss_fn="importance_sampling")
            optim = client.optim_step(trio.AdamParams(learning_rate=args.learning_rate))
            forward_result = forward.result()
            optim.result()
            metrics = getattr(forward_result, "metrics", {}) or {}
            updates = 1
        else:
            metrics = {}
        mean_reward = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
        print(
            f"step={step}/{args.steps} datums={len(datums)} reward={mean_reward:.4f} "
            f"degenerate_groups={degenerate} updates={updates} metrics={metrics} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        if args.save_every and step % args.save_every == 0:
            sampler_path = client.save_weights_for_sampler(name=f"{args.name}-step-{step}").result()
            state_path = client.save_state(name=f"{args.name}-state-{step}").result()
            save_path_json(args.paths, sampler_path=sampler_path.path, state_path=state_path.path, step=step)
    sampler_path = client.save_weights_for_sampler(name=f"{args.name}-final").result()
    state_path = client.save_state(name=f"{args.name}-final-state").result()
    save_path_json(args.paths, sampler_path=sampler_path.path, state_path=state_path.path, step=step)
    print(f"sampler_path={sampler_path.path}\nstate_path={state_path.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote PyTrio GRPO-style RL for API recommendation")
    parser.add_argument("--data", type=Path, default=Path("data/processed/api_recommendation_sft.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--resume", required=True, help="SFT or RL PyTrio state URI")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-reward-spread", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--name", default="tool-agent-api-rl")
    parser.add_argument("--paths", type=Path, default=Path("artifacts/api_rl_paths.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
