"""API catalog extraction, retrieval, prompting, and output normalization."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


API_MARKERS = (
    "Specifically, you have access to the following APIs:",
    "following APIs:",
    "available APIs:",
)
TERMINAL_API_NAMES = {"finish", "final_answer", "ask_user"}


def is_terminal_api(name: str) -> bool:
    """Return whether a ToolBench control action should be excluded from recommendations."""
    return name.strip().lower() in TERMINAL_API_NAMES


@dataclass(frozen=True)
class ApiSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    tool: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def search_text(self) -> str:
        return " ".join(
            value
            for value in (self.name, self.tool or "", self.description, json.dumps(self.parameters, ensure_ascii=False))
            if value
        )


def _bracketed_literal(text: str, start: int) -> str | None:
    """Return the balanced list literal beginning at start."""
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _find_api_literal(text: str) -> str | None:
    for marker in API_MARKERS:
        marker_index = text.lower().find(marker.lower())
        if marker_index >= 0:
            start = text.find("[", marker_index + len(marker))
            if start >= 0:
                literal = _bracketed_literal(text, start)
                if literal:
                    return literal
    return None


def extract_api_specs(messages: Iterable[dict[str, Any]]) -> list[ApiSpec]:
    """Extract API definitions embedded in ToolBench system messages."""
    found: dict[str, ApiSpec] = {}
    for message in messages:
        content = str(message.get("content", ""))
        literal = _find_api_literal(content)
        if not literal:
            continue
        try:
            values = ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict) or not value.get("name"):
                continue
            name = str(value["name"])
            if is_terminal_api(name):
                continue
            spec = ApiSpec(
                name=name,
                description=str(value.get("description", "")),
                parameters=value.get("parameters") if isinstance(value.get("parameters"), dict) else {},
                tool=str(value.get("tool")) if value.get("tool") else None,
            )
            # Keep the richest definition if the same API appears in several rows.
            previous = found.get(name)
            if previous is None or len(spec.search_text) > len(previous.search_text):
                found[name] = spec
    return list(found.values())


def build_api_catalog(input_path: Path, output_path: Path) -> int:
    specs: dict[str, ApiSpec] = {}
    with input_path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            for spec in extract_api_specs(row.get("messages", [])):
                specs[spec.name] = spec
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as target:
        for spec in sorted(specs.values(), key=lambda item: item.name):
            target.write(json.dumps(spec.as_dict(), ensure_ascii=False) + "\n")
    return len(specs)


def load_catalog(path: Path) -> list[ApiSpec]:
    specs: list[ApiSpec] = []
    with path.open() as source:
        for line in source:
            if line.strip():
                value = json.loads(line)
                specs.append(
                    ApiSpec(
                        name=str(value["name"]),
                        description=str(value.get("description", "")),
                        parameters=value.get("parameters", {}),
                        tool=value.get("tool"),
                    )
                )
    if not specs:
        raise ValueError(f"API catalog is empty: {path}")
    return specs


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


class ApiRetriever:
    """BM25 retrieval with a deterministic token-overlap fallback."""

    def __init__(self, specs: list[ApiSpec]) -> None:
        if not specs:
            raise ValueError("API catalog must contain at least one API")
        self.specs = specs
        self._corpus = [_tokens(spec.search_text) for spec in specs]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            self._bm25 = None

    def search(self, query: str, top_k: int = 8) -> list[ApiSpec]:
        top_k = max(1, min(top_k, len(self.specs)))
        query_tokens = _tokens(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            ranked = sorted(range(len(self.specs)), key=lambda index: (-float(scores[index]), self.specs[index].name))
        else:
            query_set = set(query_tokens)
            ranked = sorted(
                range(len(self.specs)),
                key=lambda index: (-len(query_set.intersection(self._corpus[index])), self.specs[index].name),
            )
        return [self.specs[index] for index in ranked[:top_k]]


RECOMMENDER_SYSTEM_PROMPT = """You are an API planning assistant.
Given a user task and a candidate API catalog, select the APIs needed to complete the task.
Do not execute any API and do not invent API names outside the candidate catalog.
Return JSON only in this format:
{"apis":[{"name":"exact_api_name","purpose":"why it is needed","order":1,"parameters":{}}],"notes":"optional planning note"}
If no candidate is relevant, return {"apis":[],"notes":"no suitable API in the catalog"}.
"""


def _compact_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    return {
        "type": parameters.get("type", "object"),
        "properties": {
            str(name): value.get("type", "string") if isinstance(value, dict) else "string"
            for name, value in properties.items()
        },
        "required": parameters.get("required", []),
    }


def recommendation_messages(task: str, candidates: list[ApiSpec], *, compact: bool = False) -> list[dict[str, str]]:
    catalog = [
        {
            "name": spec.name,
            "tool": spec.tool,
            "description": spec.description[:280] if compact else spec.description,
            "parameters": _compact_parameters(spec.parameters) if compact else spec.parameters,
        }
        for spec in candidates
    ]
    return [
        {"role": "system", "content": RECOMMENDER_SYSTEM_PROMPT + "\nCandidate APIs:\n" + json.dumps(catalog, ensure_ascii=False)},
        {"role": "user", "content": task},
    ]


def _json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_api_recommendation(text: str, candidates: list[ApiSpec]) -> list[dict[str, Any]]:
    """Normalize strict JSON or legacy ToolBench Action output."""
    known = {spec.name: spec for spec in candidates}
    raw = _json_object(text)
    values = []
    if raw:
        values = raw.get("apis") or raw.get("api_names") or raw.get("required_apis") or []
    if not isinstance(values, list):
        values = []
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values, start=1):
        if isinstance(value, str):
            name, purpose, parameters = value, "", {}
        elif isinstance(value, dict):
            name = str(value.get("name", ""))
            purpose = str(value.get("purpose", ""))
            parameters = value.get("parameters", {}) if isinstance(value.get("parameters", {}), dict) else {}
        else:
            continue
        if name not in known or name in seen:
            continue
        seen.add(name)
        parsed.append({"name": name, "purpose": purpose or known[name].description, "order": index, "parameters": parameters})
    if parsed:
        return parsed

    # ToolBench traces use: Action: api_name\nAction Input: {...}
    for name in re.findall(r"(?im)^\s*Action\s*:\s*([A-Za-z0-9_]+)", text):
        if name in known and name not in seen:
            seen.add(name)
            parsed.append({"name": name, "purpose": known[name].description, "order": len(parsed) + 1, "parameters": {}})
    return parsed


def extract_api_names_from_text(text: str, known_names: set[str]) -> list[str]:
    """Extract API names from JSON, Action lines, or ToolBench action prose."""
    found: list[tuple[int, str]] = []
    for name in sorted(known_names, key=lambda value: (-len(value), value)):
        match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
        if match:
            found.append((match.start(), name))
    result: list[str] = []
    seen: set[str] = set()
    for _, name in sorted(found):
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result
