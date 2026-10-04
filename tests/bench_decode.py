#!/usr/bin/env python3
"""Decode throughput at 1/2/4/8 concurrent streams. Stdlib only.

Every request generates exactly --tokens tokens (ignore_eos) with reasoning
off and the model's default sampling (temperature 1.0, top_p 0.97, top_k 128),
so runs are comparable across launch configurations. Each level is run
--repeat times; the median is reported.

Usage: tests/bench_decode.py [--streams 1 2 4 8] [--tokens 600] [--repeat 2]
"""
import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=f"http://127.0.0.1:{os.environ.get('PORT', '8895')}")
ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "kolibri-1"))
ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--tokens", type=int, default=600)
ap.add_argument("--repeat", type=int, default=2)
args = ap.parse_args()


def one(i: int, out: dict) -> None:
    body = {"model": args.model, "max_tokens": args.tokens, "min_tokens": args.tokens,
            "ignore_eos": True,
            "messages": [{"role": "user", "content":
                          f"Schreibe eine ausführliche Geschichte über einen Kolibri namens Nr. {i}."}],
            "chat_template_kwargs": {"reasoning_effort": "none"}}
    req = urllib.request.Request(args.base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    out[i] = r["usage"]["completion_tokens"]


for n in args.streams:
    rates = []
    for _ in range(args.repeat):
        out: dict = {}
        ts = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        rates.append(sum(out.values()) / (time.time() - t0))
    agg = statistics.median(rates)
    print(f"{n} streams: {agg:6.1f} tok/s aggregate | {agg / n:5.1f} per stream", flush=True)
