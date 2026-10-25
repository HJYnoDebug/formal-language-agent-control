"""Native tool calling through OpenRouter, a metered call wrapper, and a text fallback parser.

Generic: the tools are passed per call in meta["tools"]; nothing here knows the task.
"""
import json
import re
import time
from dataclasses import dataclass, field

from agent.harness.llm_client import HTTPTransport, PriceTable, Usage, _extract_reasoning_tokens

from .ledger import ServiceFailure
from .models import OPENROUTER_URL, PRICES_PATH

SERVICE_RETRY_PER_CALL = 1
_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class ToolReply:
    content: str
    tool_calls: list            # [(id, name, args dict or None if unparsable)]
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    provider: str
    latency_s: float
    assistant_message: dict = field(default=None)
    finish_reason: str = None   # "finish_reason/native_finish_reason" as the provider reports it (2026-10-18, logging)
    provider_error: bool = False  # an error reply that persisted through every retry (freeze log changes 12)


def tool_call_json(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def parse_tool_calls(message):
    out = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                args = None
        except json.JSONDecodeError:
            args = None
        out.append((tc.get("id") or f"call_{i}", fn.get("name"), args))
    return out


def text_calls(content, tool_names):
    """Fallback when a reply has no tool call: find the first `name(arg, arg)` with a known tool name in the text."""
    text = _THINK.sub("", content or "")
    pat = re.compile(r"\b(" + "|".join(map(re.escape, tool_names)) + r")\s*\(([^()]*)\)")
    m = pat.search(text)
    if not m:
        return []
    raw = [a.strip().strip("'\"") for a in m.group(2).split(",") if a.strip()]
    return [("text_0", m.group(1), {"_positional": raw})]


def _finish(raw):
    choice = (raw.get("choices") or [{}])[0]
    return f"{choice.get('finish_reason')}/{choice.get('native_finish_reason')}"


def provider_error_reply(raw):
    """A reply that the provider failed to produce: finish_reason "error", or no tool call, no text and no output
    token (2026-10-19, freeze log changes after the freeze 11: every such reply logged so far carried "error/error"
    and was not billed). It is retried like a transient failure instead of reaching the agent as its own reply."""
    choices = raw.get("choices") or []
    if not choices:
        return True
    choice, msg = choices[0], choices[0].get("message") or {}
    if choice.get("finish_reason") == "error":
        return True
    out = (raw.get("usage") or {}).get("completion_tokens", 0)
    return not msg.get("tool_calls") and not (msg.get("content") or "").strip() and not out


class OpenRouterToolBackend:
    EMPTY_RETRIES = 6            # retries of an empty provider error reply (free: such replies are not billed)

    def __init__(self, spec, api_key, price_table=None, transport=None, base_url=OPENROUTER_URL,
                 tool_choice="required", temperature=0.0):
        self.spec, self.api_key, self.base_url = spec, api_key, base_url
        self.tool_choice, self.temperature = tool_choice, temperature
        self.price_table = price_table or PriceTable.load(PRICES_PATH)
        self.transport = transport or HTTPTransport()

    def call(self, messages, max_tokens, meta=None):
        payload = {"model": self.spec.model_id, "messages": messages, "temperature": self.temperature,
                   "max_tokens": max_tokens, "tools": (meta or {})["tools"], "tool_choice": self.tool_choice,
                   "provider": {"order": [self.spec.route or self.spec.provider], "allow_fallbacks": False},
                   "usage": {"include": True}}
        # 2026-10-17 (freeze log, changes after the freeze): a model may name one OpenRouter endpoint of its provider
        # (spec.route) and extra request fields (spec.request_extra); both default to the earlier request.
        payload.update(self.spec.request_extra or {})
        t0 = time.time()
        persistent = False
        for attempt in range(self.EMPTY_RETRIES + 1):
            raw = self.transport.post_chat_completion(self.base_url, self.api_key, payload)
            if not provider_error_reply(raw):
                break
            if attempt == self.EMPTY_RETRIES:
                # 2026-10-20 (freeze log, changes after the freeze 12): the identical request failed every time, so
                # this is the model's reply to this context that the provider cannot deliver (temperature 0), not a
                # passing fault: it reaches the runner as a reply without a tool call (a format error, as before
                # change 11), marked provider_error, instead of failing the episode on every resume.
                persistent = True
                break
            delay = min(2 ** attempt, 30)
            print(f"[OpenRouterToolBackend] empty provider reply ({_finish(raw)}), retrying in {delay}s "
                  f"(attempt {attempt + 1}/{self.EMPTY_RETRIES})...", flush=True)
            time.sleep(delay)
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        u = raw.get("usage") or {}
        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), 0, _extract_reasoning_tokens(u))
        billed = u.get("cost")
        cost = float(billed) if isinstance(billed, (int, float)) else self.price_table.cost_of(self.spec.model_id, usage)
        content = msg.get("content") or ""
        am = {"role": "assistant", "content": content or None}
        if msg.get("tool_calls"):
            am["tool_calls"] = [{"id": tc.get("id") or f"call_{i}", "type": "function",
                                 "function": {"name": tc["function"]["name"],
                                              "arguments": tc["function"].get("arguments") or "{}"}}
                                for i, tc in enumerate(msg["tool_calls"])]
        return ToolReply(content, parse_tool_calls(msg), usage.prompt_tokens, usage.completion_tokens,
                         usage.reasoning_tokens or 0, cost, raw.get("provider") or "", time.time() - t0, am,
                         f"{choice.get('finish_reason')}/{choice.get('native_finish_reason')}", persistent)


class MeteredCaller:
    """One model call with a budget reservation and one service retry; accumulates usage into an episode record."""

    def __init__(self, backend, spec, ledger, rec):
        self.backend, self.spec, self.ledger, self.rec = backend, spec, ledger, rec

    def __call__(self, messages, max_tokens, meta):
        est = (sum(len(m.get("content") or "") for m in messages) / 3.0 * self.spec.input_price
               + max_tokens * self.spec.output_price) / 1e6 + 0.00002
        last = None
        for _ in range(1 + SERVICE_RETRY_PER_CALL):
            if self.ledger is not None:
                self.ledger.reserve(est)
            try:
                rep = self.backend.call(messages, max_tokens, meta)
            except RuntimeError as e:
                if self.ledger is not None:
                    self.ledger.settle(est, 0.0)
                last = e
                self.rec["service_errors"] += 1
                continue
            if self.ledger is not None:
                self.ledger.settle(est, rep.cost_usd)
            self.rec["calls"].append({"prompt_tokens": rep.prompt_tokens, "completion_tokens": rep.completion_tokens,
                                      "reasoning_tokens": rep.reasoning_tokens, "cost_usd": rep.cost_usd,
                                      "provider": rep.provider, "latency_s": round(rep.latency_s, 3),
                                      "n_tool_calls": len(rep.tool_calls), "prompt_chars": meta.get("prompt_chars"),
                                      "control_chars": meta.get("control_chars"), "finish_reason": rep.finish_reason,
                                      "provider_error": rep.provider_error})
            self.rec["cost_usd"] += rep.cost_usd
            self.rec["tokens_in"] += rep.prompt_tokens
            self.rec["tokens_out"] += rep.completion_tokens
            self.rec["flops"] += 2 * self.spec.active_params_b * 1e9 * (rep.prompt_tokens + rep.completion_tokens)
            return rep
        raise ServiceFailure(str(last))


def chars_of(messages):
    return sum(len(m.get("content") or "") + (len(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0)
               for m in messages)
