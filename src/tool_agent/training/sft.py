from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .common import build_sft_datum, load_jsonl, loss_per_token, require_pytrio, save_path_json


def main(args: argparse.Namespace) -> None:
    trio = require_pytrio()
    rows = load_jsonl(args.data)
    service = trio.ServiceClient()
    if args.resume:
        client = service.create_training_client_from_state(args.resume)
    else:
        client = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed)
    tokenizer = client.get_tokenizer()
    datums = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Each recommendation example must contain a non-empty messages list")
        datum = build_sft_datum(trio, tokenizer, messages, args.max_length)
        if datum is not None:
            datums.append(datum)
    if not datums:
        raise RuntimeError("No recommendation examples fit within --max-length")
    rng = np.random.default_rng(args.seed)
    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(datums))
        for start in range(0, len(order), args.batch_size):
            batch = [datums[int(index)] for index in order[start : start + args.batch_size]]
            started = time.time()
            forward = client.forward_backward(batch, loss_fn="cross_entropy")
            optim = client.optim_step(trio.AdamParams(learning_rate=args.learning_rate))
            result = forward.result()
            optim.result()
            step += 1
            print(f"epoch={epoch + 1} step={step} loss={loss_per_token(result, batch):.5f} elapsed={time.time() - started:.1f}s", flush=True)
            if args.save_every and step % args.save_every == 0:
                sampler = client.save_weights_for_sampler(name=f"{args.name}-step-{step}").result()
                state = client.save_state(name=f"{args.name}-state-{step}").result()
                save_path_json(args.paths, sampler_path=sampler.path, state_path=state.path, step=step)
    sampler = client.save_weights_for_sampler(name=f"{args.name}-final").result()
    state = client.save_state(name=f"{args.name}-final-state").result()
    save_path_json(args.paths, sampler_path=sampler.path, state_path=state.path, step=step)
    print(f"sampler_path={sampler.path}\nstate_path={state.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote PyTRIO SFT for API recommendation")
    parser.add_argument("--data", type=Path, default=Path("data/processed/api_recommendation_sft.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--resume", default=None, help="PyTRIO Train checkpoint URI")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--name", default="tool-agent-sft")
    parser.add_argument("--paths", type=Path, default=Path("artifacts/sft_paths.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
