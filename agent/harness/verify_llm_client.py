"""Scripted verification of llm_client.py -- NO real network call, NO API
key needed. A FakeTransport stands in for HTTPTransport: it returns
hand-constructed OpenAI-compatible response dicts, so this tests the
parsing/cost logic (the part actually worth testing) in complete isolation
from network/vendor concerns, same spirit as every other verify_*.py in
this harness.
"""
import io
import json
import sys
import urllib.error

import llm_client as llm_client_module
from llm_client import HTTPTransport, LLMClient, PriceTable, Transport, parse_text_action

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


class FakeTransport(Transport):
    def __init__(self, response):
        self.response = response
        self.last_payload = None

    def post_chat_completion(self, base_url, api_key, payload, timeout=120):
        self.last_payload = payload
        return self.response


PRICES = PriceTable({
    "test-model": {"input": 1.0, "output": 2.0, "cached_input": 0.1},
})

# ---------------------------------------------------------------------------
# 1. PriceTable arithmetic.
# ---------------------------------------------------------------------------
from llm_client import Usage

u = Usage(prompt_tokens=1_000_000, completion_tokens=500_000, cached_tokens=0, reasoning_tokens=0)
check("PriceTable: pure uncached-input + output cost",
      PRICES.cost_of("test-model", u) == 1.0 * 1_000_000 / 1e6 + 2.0 * 500_000 / 1e6)

u_cached = Usage(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000, reasoning_tokens=0)
check("PriceTable: fully-cached input uses cached_input price, not input price",
      abs(PRICES.cost_of("test-model", u_cached) - 0.1) < 1e-9)

u_mixed = Usage(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=400_000, reasoning_tokens=0)
expected_mixed = (600_000 * 1.0 + 400_000 * 0.1) / 1e6
check("PriceTable: mixed cached/uncached input splits correctly",
      abs(PRICES.cost_of("test-model", u_mixed) - expected_mixed) < 1e-9)

try:
    PRICES.cost_of("unpriced-model", u)
    check("PriceTable: unpriced model raises (not silently free)", False)
except KeyError:
    check("PriceTable: unpriced model raises (not silently free)", True)

# ---------------------------------------------------------------------------
# 2. parse_text_action (G6 JSON-text fallback).
# ---------------------------------------------------------------------------
check("parse_text_action: {'command': X} extracts X",
      parse_text_action('{"command": "take apple"}') == "take apple")
check("parse_text_action: {'action': X} extracts X",
      parse_text_action('{"action": "take apple"}') == "take apple")
check("parse_text_action: non-JSON plain text passes through, stripped",
      parse_text_action("  take apple  ") == "take apple")
check("parse_text_action: JSON with neither key falls back to raw content",
      parse_text_action('{"foo": "bar"}') == '{"foo": "bar"}')
check("parse_text_action: None content does not crash", parse_text_action(None) == "")

# ---------------------------------------------------------------------------
# 3. LLMClient.call() with tools -- tool_calls path.
# ---------------------------------------------------------------------------
tool_response = {
    "choices": [{
        "message": {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "act", "arguments": '{"command": "go to kitchen"}'},
            }],
        },
    }],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 30},
        "completion_tokens_details": {"reasoning_tokens": 5},
    },
}
client = LLMClient(
    base_url="https://example.invalid/v1", api_key="fake-key", model="test-model",
    price_table=PRICES, transport=FakeTransport(tool_response),
)
result = client.call(
    [{"role": "user", "content": "obs text"}],
    tools=[{"type": "function", "function": {"name": "act", "parameters": {}}}],
)
check("LLMClient.call (tools): extracts action_text from the tool call's 'command' argument",
      result.action_text == "go to kitchen")
check("LLMClient.call (tools): tool_call is populated", result.tool_call["name"] == "act")
check("LLMClient.call (tools): usage.prompt_tokens/completion_tokens parsed",
      result.usage.prompt_tokens == 100 and result.usage.completion_tokens == 20)
check("LLMClient.call (tools): cached_tokens read from nested prompt_tokens_details",
      result.usage.cached_tokens == 30)
check("LLMClient.call (tools): reasoning_tokens read from nested completion_tokens_details",
      result.usage.reasoning_tokens == 5)
expected_cost = (70 * 1.0 + 30 * 0.1 + 20 * 2.0) / 1e6
check("LLMClient.call (tools): cost reflects the cached/uncached split",
      abs(result.cost - expected_cost) < 1e-9)
check("LLMClient.call: payload sent to the transport carries tool_choice='required' when tools given",
      client.transport.last_payload["tool_choice"] == "required")

# ---------------------------------------------------------------------------
# 4. LLMClient.call() without tools -- G6 text fallback path.
# ---------------------------------------------------------------------------
text_response = {
    "choices": [{"message": {"role": "assistant", "content": '{"command": "open door to kitchen"}'}}],
    "usage": {"prompt_tokens": 50, "completion_tokens": 10},
}
client2 = LLMClient(
    base_url="https://example.invalid/v1", api_key="fake-key", model="test-model",
    price_table=PRICES, transport=FakeTransport(text_response),
)
result2 = client2.call([{"role": "user", "content": "obs text"}], tools=None)
check("LLMClient.call (no tools): action_text comes from parse_text_action on message content",
      result2.action_text == "open door to kitchen")
check("LLMClient.call (no tools): tool_call is None", result2.tool_call is None)
check("LLMClient.call (no tools): cached/reasoning default to 0 when usage has no nested details",
      result2.usage.cached_tokens == 0 and result2.usage.reasoning_tokens == 0)
check("LLMClient.call (no tools): payload sent to the transport has no 'tools' key at all",
      "tools" not in client2.transport.last_payload)

# ---------------------------------------------------------------------------
# 5. reasoning (OpenRouter-shaped dict) is passed through to the payload
# only when set. Replaces an earlier flat 'reasoning_effort' string field --
# that field was EMPIRICALLY CONFIRMED NOT TO SUPPRESS THINKING for at
# least one real ladder model routed through OpenRouter (2026-09-28 live
# preflight against qwen/qwen3-8b, see llm_client.LLMClient.__init__'s
# docstring and runs/preflight_reasoning_control.json); {"enabled": False}
# was the shape that actually worked (0 reasoning tokens vs 132).
# ---------------------------------------------------------------------------
client3 = LLMClient(
    base_url="https://example.invalid/v1", api_key="fake-key", model="test-model",
    price_table=PRICES, reasoning={"enabled": False}, transport=FakeTransport(text_response),
)
client3.call([{"role": "user", "content": "x"}], tools=None)
check("LLMClient.call: reasoning dict is forwarded when set",
      client3.transport.last_payload.get("reasoning") == {"enabled": False})

client4 = LLMClient(
    base_url="https://example.invalid/v1", api_key="fake-key", model="test-model",
    price_table=PRICES, transport=FakeTransport(text_response),
)
client4.call([{"role": "user", "content": "x"}], tools=None)
check("LLMClient.call: reasoning key is absent when not set (not sent as null)",
      "reasoning" not in client4.transport.last_payload)

# ---------------------------------------------------------------------------
# 6. max_tokens is ALWAYS sent, never omitted. Found necessary the hard way
# (2026-09-28, real paid run): without it, every call to
# qwen/qwen3-235b-a22b-2507 via OpenRouter's GMICloud provider failed with
# HTTP 400 because that provider defaulted the completion budget to the
# model's ENTIRE 131072-token context window, which alone exceeds the
# context limit once any input tokens are added -- failed on turn 1 of
# every episode, regardless of conversation length, until this was fixed.
# See LLMClient.__init__'s docstring for the full incident.
# ---------------------------------------------------------------------------
client5 = LLMClient(
    base_url="https://example.invalid/v1", api_key="fake-key", model="test-model",
    price_table=PRICES, transport=FakeTransport(text_response),
)
client5.call([{"role": "user", "content": "x"}], tools=None)
check("LLMClient.call: max_tokens is always present in the payload (default 2048)",
      client5.transport.last_payload.get("max_tokens") == 2048)

client6 = LLMClient(
    base_url="https://example.invalid/v1", api_key="fake-key", model="test-model",
    price_table=PRICES, max_tokens=256, transport=FakeTransport(text_response),
)
client6.call([{"role": "user", "content": "x"}], tools=None)
check("LLMClient.call: max_tokens is overridable via the constructor",
      client6.transport.last_payload.get("max_tokens") == 256)

# ---------------------------------------------------------------------------
# 7. HTTPTransport retry/backoff (added 2026-09-29, before the first real
# batch of more than a handful of episodes -- see HTTPTransport's own
# docstring for the real incident this fixes: a single 429/5xx/network
# blip used to kill the WHOLE episode, silently losing its trace, and
# wrongly count against the run's circuit breaker). Monkeypatches
# urllib.request.urlopen directly (the one place HTTPTransport actually
# touches the network) and llm_client.time.sleep (so this test doesn't
# really wait through the backoff delays) -- still fully network-free,
# same spirit as every other check in this file, just one level closer to
# the real transport than the FakeTransport-based checks above.
# ---------------------------------------------------------------------------
class _FakeHTTPResponse:
    def __init__(self, body_bytes):
        self._body = body_bytes

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_urlopen(events):
    """events: list consumed in order; each item is either an Exception
    instance to raise, or a dict to return as a successful parsed JSON
    response. Returns the list of (args, kwargs) urlopen was actually
    called with, appended to live as calls happen."""
    events = list(events)
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        event = events.pop(0)
        if isinstance(event, Exception):
            raise event
        return _FakeHTTPResponse(json.dumps(event).encode("utf-8"))

    llm_client_module.urllib.request.urlopen = fake_urlopen
    return calls


_real_urlopen = llm_client_module.urllib.request.urlopen
_real_sleep = llm_client_module.time.sleep
llm_client_module.time.sleep = lambda seconds: None  # skip real backoff waits in this test

_ok_response = {
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}

# 7a. No failure at all -- baseline sanity check the refactor didn't
# change the happy path.
calls = _install_fake_urlopen([_ok_response])
transport = HTTPTransport(max_retries=3, base_delay=0.01)
result = transport.post_chat_completion("https://example.invalid/v1", "key", {"model": "x"})
check("HTTPTransport: happy path (no failure) still returns the parsed response",
      result == _ok_response)
check("HTTPTransport: happy path makes exactly 1 call", len(calls) == 1)

# 7b. One retryable HTTP 429, then success -- should retry once and
# return the eventual success.
calls = _install_fake_urlopen([
    urllib.error.HTTPError("url", 429, "rate limited", {}, io.BytesIO(b"rate limited")),
    _ok_response,
])
transport = HTTPTransport(max_retries=3, base_delay=0.01)
result = transport.post_chat_completion("https://example.invalid/v1", "key", {"model": "x"})
check("HTTPTransport: retries once on HTTP 429 then returns the eventual success",
      result == _ok_response and len(calls) == 2)

# 7c. A retryable 5xx, then success.
calls = _install_fake_urlopen([
    urllib.error.HTTPError("url", 503, "unavailable", {}, io.BytesIO(b"unavailable")),
    _ok_response,
])
transport = HTTPTransport(max_retries=3, base_delay=0.01)
result = transport.post_chat_completion("https://example.invalid/v1", "key", {"model": "x"})
check("HTTPTransport: retries once on HTTP 503 then returns the eventual success",
      result == _ok_response and len(calls) == 2)

# 7d. A URLError (network-level, not an HTTP status), then success.
calls = _install_fake_urlopen([
    urllib.error.URLError("connection reset"),
    _ok_response,
])
transport = HTTPTransport(max_retries=3, base_delay=0.01)
result = transport.post_chat_completion("https://example.invalid/v1", "key", {"model": "x"})
check("HTTPTransport: retries once on URLError then returns the eventual success",
      result == _ok_response and len(calls) == 2)

# 7e. A PERMANENT HTTP error (401) must NOT be retried -- fails on the
# first attempt, exactly 1 call made.
calls = _install_fake_urlopen([
    urllib.error.HTTPError("url", 401, "unauthorized", {}, io.BytesIO(b"bad key")),
    _ok_response,  # would prove a retry happened if this got consumed
])
transport = HTTPTransport(max_retries=3, base_delay=0.01)
try:
    transport.post_chat_completion("https://example.invalid/v1", "key", {"model": "x"})
    check("HTTPTransport: HTTP 401 raises rather than retrying", False)
except RuntimeError as e:
    check("HTTPTransport: HTTP 401 raises rather than retrying",
          len(calls) == 1 and "401" in str(e))

# 7f. A retryable failure that NEVER succeeds must give up after
# max_retries and raise -- exactly max_retries+1 total attempts. (2026-10-17, freeze log changes after the freeze 2:
# a 429 has its own, longer window, HTTPTransport.RATE_LIMIT_RETRIES, so the check uses a 5xx for max_retries and
# checks the 429 window separately.)
calls = _install_fake_urlopen([
    urllib.error.HTTPError("url", 503, "unavailable", {}, io.BytesIO(b"unavailable"))
    for _ in range(10)
])
transport = HTTPTransport(max_retries=2, base_delay=0.01)
try:
    transport.post_chat_completion("https://example.invalid/v1", "key", {"model": "x"})
    check("HTTPTransport: persistent 503 eventually raises after max_retries", False)
except RuntimeError:
    check("HTTPTransport: persistent 503 eventually raises after max_retries",
          len(calls) == 3)  # 1 initial attempt + 2 retries

calls = _install_fake_urlopen([
    urllib.error.HTTPError("url", 429, "rate limited", {}, io.BytesIO(b"rate limited"))
    for _ in range(HTTPTransport.RATE_LIMIT_RETRIES + 5)
])
transport = HTTPTransport(max_retries=2, base_delay=0.01)
try:
    transport.post_chat_completion("https://example.invalid/v1", "key", {"model": "x"})
    check("HTTPTransport: persistent 429 eventually raises after RATE_LIMIT_RETRIES", False)
except RuntimeError:
    check("HTTPTransport: persistent 429 eventually raises after RATE_LIMIT_RETRIES",
          len(calls) == HTTPTransport.RATE_LIMIT_RETRIES + 1)

llm_client_module.urllib.request.urlopen = _real_urlopen
llm_client_module.time.sleep = _real_sleep

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
else:
    print("All checks passed.")
