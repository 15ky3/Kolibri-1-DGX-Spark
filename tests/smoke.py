#!/usr/bin/env python3
"""Smoke test for a running Kolibri-1 server. Stdlib only.

Usage: tests/smoke.py [--base http://127.0.0.1:8895] [--key API_KEY]

Checks, in order: /health, /v1/models, a German answer with reasoning off,
the reasoning split with reasoning on (reasoning in its own field, answer in
content), and one tool call parsed into structured tool_calls. Exit code is
the number of failed checks.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=f"http://127.0.0.1:{os.environ.get('PORT', '8895')}")
ap.add_argument("--key", default=os.environ.get("API_KEY", ""))
ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "kolibri-1"))
args = ap.parse_args()

HDR = {"Content-Type": "application/json"}
if args.key:
    HDR["Authorization"] = f"Bearer {args.key}"
failed = 0


def call(path, body=None, timeout=600):
    req = urllib.request.Request(
        args.base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=HDR,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)


def check(name, fn):
    global failed
    t0 = time.time()
    try:
        detail = fn()
        print(f"PASS  {name}  ({time.time() - t0:.1f}s)  {detail or ''}")
    except Exception as e:  # noqa: BLE001 - report every failure the same way
        failed += 1
        print(f"FAIL  {name}  ({time.time() - t0:.1f}s)  {e}")


def chat(messages, **extra):
    body = {"model": args.model, "messages": messages, "max_tokens": 2048}
    body.update(extra)
    _, r = call("/v1/chat/completions", body)
    return r


def t_health():
    status, _ = call("/health", timeout=10)
    assert status == 200, status


def t_models():
    _, r = call("/v1/models", timeout=10)
    ids = [m["id"] for m in r["data"]]
    assert args.model in ids, ids
    return f"max_model_len={r['data'][0].get('max_model_len')}"


def t_no_reasoning():
    r = chat(
        [{"role": "user", "content": "Was ist die Hauptstadt von Deutschland? Antworte in einem Satz."}],
        chat_template_kwargs={"reasoning_effort": "none"},
        temperature=0,
    )
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    assert "Berlin" in content, repr(content)
    assert not (msg.get("reasoning") or msg.get("reasoning_content")), "reasoning present with effort=none"
    u = r["usage"]
    return f"{u['completion_tokens']} tok: {content.strip()[:80]!r}"


def t_reasoning():
    t0 = time.time()
    r = chat(
        [{"role": "user", "content": "Wie viel ist 17 * 23? Gib am Ende nur die Zahl an."}],
        chat_template_kwargs={"reasoning_effort": "low"},
        temperature=1.0, top_p=0.97, top_k=128,
    )
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    assert reasoning.strip(), "no reasoning field"
    assert "391" in content, repr(content[-200:])
    assert "</think>" not in content, "think tag leaked into content"
    n = r["usage"]["completion_tokens"]
    return f"{n} tok in {dt:.1f}s (~{n / dt:.1f} tok/s incl. prefill), reasoning {len(reasoning)} chars"


def t_tool_call():
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    r = chat(
        [{"role": "user", "content": "Wie ist das Wetter gerade in Heidelberg?"}],
        tools=tools,
        chat_template_kwargs={"reasoning_effort": "low"},
    )
    calls = r["choices"][0]["message"].get("tool_calls") or []
    assert calls, f"no tool_calls; content={r['choices'][0]['message'].get('content')!r}"
    fn = calls[0]["function"]
    a = json.loads(fn["arguments"])
    assert fn["name"] == "get_weather" and "heidelberg" in a.get("city", "").lower(), fn
    return f"{fn['name']}({fn['arguments']})"


check("health", t_health)
check("models", t_models)
check("chat, reasoning off (de)", t_no_reasoning)
check("chat, reasoning low", t_reasoning)
check("tool call", t_tool_call)
print(f"\n{5 - failed}/5 passed")
sys.exit(failed)
