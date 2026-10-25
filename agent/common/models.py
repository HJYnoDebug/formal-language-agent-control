"""Model table: aliases, pinned provider, prices and active parameters (config/model_prices.json)."""
import json
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRICES_PATH = os.path.join(ROOT, "config", "model_prices.json")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MODEL_ALIASES = {"ministral-3b": "mistralai/ministral-3b-2512", "ministral-8b": "mistralai/ministral-8b-2512",
                 "ministral-14b": "mistralai/ministral-14b-2512", "large-3": "mistralai/mistral-large-2512",
                 "small-4": "mistralai/mistral-small-2603",          # candidate (cost test 2026-10-17), not in the plan
                 # Qwen3 (model set until 02b v1.9.3; kept to read the earlier smoke records)
                 "14b": "qwen/qwen3-14b", "32b": "qwen/qwen3-32b",
                 "30b-a3b": "qwen/qwen3-30b-a3b-instruct-2507", "235b": "qwen/qwen3-235b-a22b-2507"}
LINE = {"ministral-3b": "dense", "ministral-8b": "dense", "ministral-14b": "dense", "large-3": "moe", "small-4": "moe",
        "14b": "dense", "32b": "dense", "30b-a3b": "moe", "235b": "moe"}


@dataclass
class ModelSpec:
    model_id: str
    provider: str
    no_think: bool
    input_price: float
    output_price: float
    active_params_b: float
    cache_price: float = None             # price of cached input tokens, None without prompt caching
    route: str = None                     # OpenRouter endpoint slug (e.g. "mistral/eu"); None routes to `provider`
    request_extra: dict = None            # extra request fields for this model (e.g. reasoning switched off)
    max_concurrency: int = None           # at most this many episodes of this model at once (None: only `workers`)


def load_models(path=PRICES_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for mid, p in data.items():
        if mid.startswith("_"):
            continue
        out[mid] = ModelSpec(mid, p["provider"], "/no_think" in p.get("reasoning", ""), p["input"], p["output"],
                             p["active_params_b"], p.get("cache_read"), p.get("route"), p.get("request_extra"),
                             p.get("max_concurrency"))
    return out


def resolve_model(name, models):
    mid = MODEL_ALIASES.get(name, name)
    if mid not in models:
        raise KeyError(f"unknown model {name!r}")
    return models[mid]
