import json
from pathlib import Path

from tool_agent.api_recommender import ApiRetriever, ApiSpec, extract_api_specs, parse_api_recommendation
from tool_agent.data.build_recommendation_data import normalize_row
from tool_agent.training.rl import Rollout, build_rl_datum, recommendation_reward


def test_extract_and_parse_toolbench_api_catalog():
    messages = [{"role": "system", "content": "Specifically, you have access to the following APIs: [{'name': 'weather_api', 'description': 'weather lookup', 'parameters': {'type': 'object'}}]"}]
    specs = extract_api_specs(messages)
    assert specs[0].name == "weather_api"
    text = '{"apis":[{"name":"weather_api","purpose":"weather","parameters":{}}]}'
    assert parse_api_recommendation(text, specs)[0]["name"] == "weather_api"


def test_retriever_returns_known_api_without_bm25_dependency():
    specs = [
        ApiSpec("weather_api", "weather city forecast", {"type": "object"}),
        ApiSpec("music_api", "search songs", {"type": "object"}),
    ]
    assert ApiRetriever(specs).search("weather forecast", 1)[0].name == "weather_api"


def test_build_recommendation_example_uses_json_target():
    row = {
        "task_id": "x",
        "messages": [
            {"role": "system", "content": "Specifically, you have access to the following APIs: [{'name': 'weather_api', 'description': 'weather lookup', 'parameters': {}}]"},
            {"role": "user", "content": "find weather"},
            {"role": "assistant", "content": "Action: weather_api\nAction Input: {}"},
        ],
    }
    result = normalize_row(row, 8)
    assert result and result["source"] == "api_recommendation"
    assert json.loads(result["messages"][-1]["content"])["apis"][0]["name"] == "weather_api"


def test_terminal_control_api_is_excluded_from_recommendations():
    row = {
        "task_id": "x",
        "messages": [
            {"role": "system", "content": "Specifically, you have access to the following APIs: [{'name': 'weather_api', 'description': 'weather lookup', 'parameters': {}}, {'name': 'Finish', 'description': 'final answer', 'parameters': {}}]"},
            {"role": "user", "content": "find weather"},
            {"role": "assistant", "content": "Action: weather_api\nAction Input: {}\nAction: Finish\nAction Input: {}"},
        ],
    }
    result = normalize_row(row, 8)
    assert result and result["gold_api_names"] == ["weather_api"]
    assert all(item["name"] != "Finish" for item in result["candidate_apis"])


def test_recommendation_reward_prefers_exact_ordered_json():
    specs = [
        ApiSpec("weather_api", "weather lookup", {"type": "object"}),
        ApiSpec("maps_api", "map lookup", {"type": "object"}),
    ]
    exact = '{"apis":[{"name":"weather_api"},{"name":"maps_api"}]}'
    reversed_result = '{"apis":[{"name":"maps_api"},{"name":"weather_api"}]}'
    assert recommendation_reward(exact, specs, ["weather_api", "maps_api"]) > recommendation_reward(
        reversed_result, specs, ["weather_api", "maps_api"]
    )


def test_build_rl_datum_aligns_prompt_and_completion():
    class FakeTrio:
        class ModelInput:
            @staticmethod
            def from_ints(tokens):
                return list(tokens)

        class Datum:
            def __init__(self, model_input, loss_fn_inputs):
                self.model_input = model_input
                self.loss_fn_inputs = loss_fn_inputs

    datum = build_rl_datum(FakeTrio, [10, 11, 12], Rollout([20, 21], [-0.2, -0.3], "x", 0.5))
    assert datum.model_input == [10, 11, 12, 20]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [0, 0, 20, 21]
    assert datum.loss_fn_inputs["advantages"].tolist() == [0.0, 0.0, 0.5, 0.5]
