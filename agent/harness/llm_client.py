"""Vendor-agnostic OpenAI-compatible chat client + cost metering
(harness_design.md sec 2.2; spec sec 9's model table, sec 6's per-episode
cost breakdown, sec 12 item 4's frozen model/price/date requirement).

Split into three pieces specifically so the parsing/cost logic -- the part
that is actually worth unit-testing -- never needs a network call or an API
key to test (see verify_llm_client.py, which tests all of it against a
FakeTransport):

  - Transport: the actual HTTP POST. HTTPTransport (stdlib urllib only, no
    extra pip package -- agent/common/episode_runner.py's own docstring
    notes this harness has to import unmodified into three separately
    managed per-environment venvs with different package sets, so this file
    avoids adding a new dependency to all three just for the client) is the
    real one; a FakeTransport in verify_llm_client.py stands in for tests.
  - PriceTable: config/model_prices.json -> USD cost of a Usage. Pure
    arithmetic, no network.
  - LLMClient: wires a Transport + PriceTable together and turns a raw
    OpenAI-compatible response into (action_text, usage, cost) -- what
    scaffolds.py and episode_runner_v2.py actually consume.

This file does NOT decide which vendor/model/base_url to use -- that is a
LLMClient CONSTRUCTOR argument, supplied by whatever wires up a real run.
Nothing in this file changes if the vendor decision changes (see the
"aforce" conversation note in control.py's own docstring, which made the
same point about control.py -- this is the same property carried one layer
further, into the piece that actually IS vendor-specific in the sense of
"needs a base_url and a key", but not in the sense of "needs different
code").

Cost formula and its honest limitations (documented rather than silently
assumed correct for every vendor):
  - usage.cached_tokens and usage.reasoning_tokens are read from whichever
    of a few known nested-field conventions the response actually uses
    (OpenAI's prompt_tokens_details.cached_tokens /
    completion_tokens_details.reasoning_tokens; a flatter cached_tokens /
    reasoning_tokens at the top of usage, seen on some OpenAI-compatible
    servers). If a future vendor's response uses a third shape, these will
    silently read as 0 rather than crash -- this is a deliberate fail-open
    choice (a missing cache/reasoning breakdown should not stop the run),
    but it means cost accounting should be spot-checked against that
    vendor's own dashboard during the smoke test (G4/G5), not trusted
    blindly on day one.
  - reasoning tokens are folded into the 'output' price tier by default
    (most OpenAI-compatible servers already include them inside
    completion_tokens, i.e. double-counting them separately would
    overcharge) -- PriceTable.cost_of() does not bill the 'reasoning' price
    tier from model_prices.json at all yet; that field exists in the price
    table only so a vendor that DOES bill reasoning tokens separately (and
    reports them OUTSIDE completion_tokens) can be wired in later without
    changing the table's schema.
"""
import http.client
import json
import random
import time
import urllib.error
import urllib.request
from collections import namedtuple

Usage = namedtuple(
    "Usage", ["prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"]
)
CallResult = namedtuple("CallResult", ["action_text", "tool_call", "usage", "cost", "raw"])


class PriceTable:
    """Wraps config/model_prices.json (USD per 1,000,000 tokens)."""

    def __init__(self, prices_by_model):
        self.prices_by_model = prices_by_model

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(data)

    def cost_of(self, model, usage):
        """USD cost of one call's Usage, at `model`'s price. Raises KeyError
        (not a silent 0) for an unpriced model -- spec sec 12 item 4 treats
        the price table as something that must be complete and frozen
        before a real run, so a missing entry is a setup bug to fix, not a
        free call to quietly allow."""
        p = self.prices_by_model.get(model)
        if p is None:
            raise KeyError(
                f"no price entry for model {model!r} in the price table -- "
                f"add it to config/model_prices.json (spec sec 12 item 4: "
                f"model prices must be frozen before a real run) before "
                f"calling this model"
            )
        cached = usage.cached_tokens or 0
        uncached_input = max(0, usage.prompt_tokens - cached)
        cached_price = p.get("cached_input", p["input"])
        return (
            uncached_input * p["input"]
            + cached * cached_price
            + usage.completion_tokens * p["output"]
        ) / 1_000_000


class Transport:
    """The network boundary. See module docstring for why this exists."""

    def post_chat_completion(self, base_url, api_key, payload, timeout=120):
        raise NotImplementedError


class HTTPTransport(Transport):
    """stdlib-only (urllib) POST to <base_url>/chat/completions, OpenAI's
    request/response shape (the de facto standard every vendor spec sec 9
    lists advertises compatibility with -- Qwen3, DeepSeek, GPT-5-class).

    Retries with exponential backoff + jitter on TRANSIENT failures only
    (added 2026-09-29, before the first real budget-bearing batch of more
    than a handful of episodes): a 429 (rate limit), a 5xx, or a URLError
    (DNS hiccup, connection reset, timeout) is exactly the kind of thing a
    multi-hundred-call real batch WILL hit sooner or later on a shared
    OpenRouter account, and until now a single one of these killed the
    WHOLE episode outright -- episode_runner_v2.run_episode_v2() only
    writes its trace file once, after the loop ends, so every already-
    accepted step in that episode (and the money already spent on them,
    which the caller's manifest.add_cost() calls from earlier turns already
    recorded correctly) was lost from the trace with nothing to show for
    it, and the run's consecutive-failure circuit breaker took the hit for
    what was really just one dropped connection, not a real problem with
    the model/harness. A permanent failure (401 bad key, 400 malformed
    request, 404 wrong model id, ...) is NOT retried -- retrying those
    would just burn wall-clock time re-sending a request that can never
    succeed, and would mask a real configuration bug (e.g. a mistyped
    model id) behind several minutes of silent retrying instead of failing
    fast and visibly.
    """

    RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

    # 2026-10-13 (desk pilot, freeze record amendment 2): a DNS outage of several minutes outlasted the old window
    # (5 retries, max 20 s), and an http.client.RemoteDisconnected raised while reading the response escaped the
    # retry loop and aborted the run. Now every transient network error is retried, over about 3 minutes.
    TRANSIENT_ERRORS = (http.client.HTTPException, ConnectionError, TimeoutError, json.JSONDecodeError)

    # 2026-10-17 (freeze log, changes after the freeze 2): Mistral rate-limits OpenRouter's shared quota upstream for
    # minutes at a time; a 429 now gets its own, longer window (20 retries, delays up to 120 s, about 30 minutes) and
    # waits at least as long as a Retry-After header asks, so that an episode waits instead of failing half-way.
    RATE_LIMIT_RETRIES = 20
    RATE_LIMIT_MAX_DELAY = 120.0

    def __init__(self, max_retries=8, base_delay=1.0, max_delay=60.0):
        """max_retries: number of ADDITIONAL attempts after the first (so
        max_retries=5 means up to 6 total tries) -- 5 is generous enough to
        ride out a minute or two of rate-limiting/provider flakiness
        without making a genuinely broken run hang for a long time.
        base_delay/max_delay: exponential backoff bounds in seconds
        (1, 2, 4, 8, 16, capped at max_delay), each with up to 50% random
        jitter added so many concurrent callers hitting the same 429 don't
        all retry in lockstep."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def post_chat_completion(self, base_url, api_key, payload, timeout=120):
        url = base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                limited = e.code == 429
                limit = self.RATE_LIMIT_RETRIES if limited else self.max_retries
                if e.code not in self.RETRYABLE_HTTP_CODES or attempt >= limit:
                    raise RuntimeError(f"LLM API call failed: HTTP {e.code}: {body}") from e
                # 2026-10-17: the body says whose limit it is (OpenRouter account vs. "rate-limited upstream")
                self._sleep_before_retry(attempt, f"HTTP {e.code}: {' '.join(body.split())[:160]}",
                                         self.RATE_LIMIT_MAX_DELAY if limited else None,
                                         _retry_after(e.headers) if limited else None, limit)
                attempt += 1
            except urllib.error.URLError as e:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"LLM API call failed: could not reach {url}: {e.reason}"
                    ) from e
                self._sleep_before_retry(attempt, f"network error: {e.reason}")
                attempt += 1
            except self.TRANSIENT_ERRORS as e:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"LLM API call failed: {type(e).__name__}: {e}") from e
                self._sleep_before_retry(attempt, f"{type(e).__name__}")
                attempt += 1

    def _sleep_before_retry(self, attempt, reason, max_delay=None, at_least=None, limit=None):
        delay = min(self.base_delay * (2 ** attempt), max_delay or self.max_delay)
        delay += random.uniform(0, delay * 0.5)  # jitter, avoids retry lockstep
        if at_least:
            delay = max(delay, min(at_least, 600.0))
        print(
            f"[HTTPTransport] transient failure ({reason}), "
            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{limit or self.max_retries})...",
            flush=True,
        )
        time.sleep(delay)


def _retry_after(headers):
    """Seconds from a Retry-After header (the integer form only), else None."""
    try:
        v = (headers or {}).get("Retry-After")
        return float(v) if v is not None and float(v) >= 0 else None
    except (TypeError, ValueError):
        return None


def _extract_cached_tokens(usage_raw):
    details = usage_raw.get("prompt_tokens_details") or {}
    if "cached_tokens" in details:
        return details["cached_tokens"]
    return usage_raw.get("cached_tokens", 0)


def _extract_reasoning_tokens(usage_raw):
    details = usage_raw.get("completion_tokens_details") or {}
    if "reasoning_tokens" in details:
        return details["reasoning_tokens"]
    return usage_raw.get("reasoning_tokens", 0)


def parse_text_action(content):
    """alpha for the tools=None / G6 fallback path (spec sec 5: "工具调用在
    某供应商不稳定时,全组改 JSON 文本输出"). Tries {"command": "..."} (or
    "action") first, matching that fallback's own name ("改 JSON 文本
    输出"); if the content is not valid JSON (or is JSON but has neither
    key), falls back to the raw, stripped text itself -- this is a last
    resort so a model that just wrote the bare command as plain text is
    still usable rather than producing an unparseable-action error every
    single turn.
    """
    content = (content or "").strip()
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(obj, dict):
        for key in ("command", "action"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
    return content


class LLMClient:
    """model / base_url / api_key / temperature / reasoning are all
    constructor arguments -- nothing about which vendor is being used is
    hard-coded anywhere in this class's methods.
    """

    def __init__(
        self,
        base_url,
        api_key,
        model,
        price_table,
        temperature=0.0,
        reasoning=None,
        max_tokens=2048,
        transport=None,
        extra_payload=None,
    ):
        """max_tokens: ALWAYS sent in the payload (never omitted) -- found
        necessary the hard way (2026-09-28): without it, every single call
        to qwen/qwen3-235b-a22b-2507 through OpenRouter's GMICloud provider
        failed with HTTP 400 ("Requested token count exceeds the model's
        maximum context length of 131072 tokens... 637 tokens from the
        input messages and 131072 tokens for the completion"). That is not
        a typo: with no max_tokens in the request, this provider defaulted
        the COMPLETION budget to the model's entire context window, which by
        itself already exceeds the context limit once any input tokens are
        added -- so it failed on turn 1 of every episode, every arm,
        regardless of conversation length, until a real value was set here.
        gpt-oss-20b's provider (the earlier quicktest's model) apparently
        does not have this quirk, which is why that run never hit it --
        this was a real gap in the client, not a fluke of one model. 2048 is
        a generous, cheap ceiling (a provider bills for tokens actually
        generated, not for how high max_tokens is set) -- comfortably above
        anything a short ALFWorld/ScienceWorld/CookingWorld action, a
        PlanPolicy update_plan list, or a brief P self-report sentence
        should ever need, while staying nowhere near any ladder model's real
        context limit. Override per LLMClient instance if a scaffold or
        model genuinely needs more (e.g. a reasoning model whose thinking
        eats into the same completion budget).

        reasoning: an OpenRouter-shaped dict merged into the payload as
        payload["reasoning"] = reasoning, e.g. {"enabled": False} (turns a
        hybrid thinking/non-thinking model's thinking fully off) or
        {"effort": "low"/"minimal"} (for a model that always reasons but
        exposes an effort knob). None (the default) means "send nothing,
        let the vendor's own default apply" -- which for some models (see
        below) means thinking stays ON.

        This REPLACES an earlier `reasoning_effort` (flat, OpenAI-style
        top-level field) constructor argument. That flat field was replaced
        because it was EMPIRICALLY CONFIRMED NOT TO WORK for at least one
        real ladder member routed through OpenRouter: a live preflight call
        to qwen/qwen3-8b (2026-09-28, runs/preflight_reasoning_control.json)
        with payload["reasoning_effort"]="minimal" still produced 132
        hidden reasoning tokens for a 1-word reply (STILL THINKING); the
        same call with payload["reasoning"]={"enabled": False} produced 0
        reasoning tokens (SUPPRESSED). spec sec 9's "关闭思考模式或设最低
        推理强度" requirement -- and therefore spec sec 10's whole per-episode
        token-budget arithmetic, which assumes no hidden thinking overhead
        -- depends on getting this right, so this was changed to the
        confirmed-working shape rather than left as a plausible-looking
        guess. Only actually verified against qwen/qwen3-8b so far (not
        every ladder model) -- deliberately not over-generalized from a
        single data point; whichever new model config/model_prices.json's
        `_reasoning` field has NOT been checked against a real call should
        get the same kind of preflight before it is trusted for a real
        budget-bearing batch.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.price_table = price_table
        self.temperature = temperature
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.transport = transport or HTTPTransport()
        # extra_payload (added 2026-10-04, v5.4 runner): merged into every request payload, e.g.
        # {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}, "usage": {"include": True}}
        # for OpenRouter provider pinning and billed-cost reporting. None keeps the old payload unchanged.
        self.extra_payload = dict(extra_payload or {})

    def call(self, messages, tools=None):
        """(messages, tools=None) -> CallResult. tools=None is the G6
        fallback path (harness_design.md sec 2.2): same method, same
        return shape, action_text comes from parse_text_action(content)
        instead of a tool_call -- callers (scaffolds.py) never need an
        if/else of their own for this."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        payload.update(self.extra_payload)
        raw = self.transport.post_chat_completion(self.base_url, self.api_key, payload)
        return self._parse(raw, used_tools=bool(tools))

    def _parse(self, raw, used_tools):
        choice = raw["choices"][0]
        message = choice["message"]
        usage_raw = raw.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            cached_tokens=_extract_cached_tokens(usage_raw),
            reasoning_tokens=_extract_reasoning_tokens(usage_raw),
        )
        cost = self.price_table.cost_of(self.model, usage)

        tool_call = None
        tool_calls = message.get("tool_calls") or []
        if used_tools and tool_calls:
            tc = tool_calls[0]
            tool_call = {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
            try:
                args = json.loads(tc["function"]["arguments"])
                action_text = args.get("command")
                if action_text is None:
                    action_text = args.get("action")
                if action_text is None and args:
                    action_text = next(iter(args.values()))
            except (json.JSONDecodeError, AttributeError):
                action_text = tc["function"]["arguments"]
        else:
            action_text = parse_text_action(message.get("content"))

        return CallResult(action_text=action_text, tool_call=tool_call, usage=usage, cost=cost, raw=raw)
