#!/usr/bin/env python3
"""Perplexity of fixed German and English reference texts via prompt_logprobs.

A deployment integrity check: run it before and after changing load strategy,
kernels or MoE tables. Identical weights and kernels give identical numbers;
anything that corrupts the computation moves them. Stdlib only.

The text is scored as an assistant answer inside the chat template. Kolibri has
no BOS token and is lost without a leading <|im_start|>: the same Grundgesetz
passage scores perplexity ~72 as raw text and ~1.5 inside the chat frame
(measured 2026-10-04), so raw-text perplexity says nothing about the deployment.
"""
import json
import math
import os
import urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('PORT', '8895')}"
TEXTS = {
    # Grundgesetz Art. 1-3 (amtliches Werk, gemeinfrei)
    "de": "Die Würde des Menschen ist unantastbar. Sie zu achten und zu schützen ist "
          "Verpflichtung aller staatlichen Gewalt. Das Deutsche Volk bekennt sich darum zu "
          "unverletzlichen und unveräußerlichen Menschenrechten als Grundlage jeder "
          "menschlichen Gemeinschaft, des Friedens und der Gerechtigkeit in der Welt. "
          "Jeder hat das Recht auf die freie Entfaltung seiner Persönlichkeit, soweit er "
          "nicht die Rechte anderer verletzt und nicht gegen die verfassungsmäßige Ordnung "
          "oder das Sittengesetz verstößt. Jeder hat das Recht auf Leben und körperliche "
          "Unversehrtheit. Die Freiheit der Person ist unverletzlich. Alle Menschen sind vor "
          "dem Gesetz gleich. Männer und Frauen sind gleichberechtigt.",
    # US Declaration of Independence, preamble (public domain)
    "en": "When in the Course of human events, it becomes necessary for one people to "
          "dissolve the political bands which have connected them with another, and to "
          "assume among the powers of the earth, the separate and equal station to which "
          "the Laws of Nature and of Nature's God entitle them, a decent respect to the "
          "opinions of mankind requires that they should declare the causes which impel "
          "them to the separation. We hold these truths to be self-evident, that all men "
          "are created equal, that they are endowed by their Creator with certain "
          "unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.",
}

MODEL = os.environ.get("SERVED_MODEL_NAME", "kolibri-1")
ASK = {"de": "Zitiere den Anfang des Grundgesetzes ab Artikel 1.",
       "en": "Quote the opening of the US Declaration of Independence."}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


for lang, text in TEXTS.items():
    prefix = (f"<|im_start|>user\n{ASK[lang]}<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    start = len(post("/tokenize", {"model": MODEL, "prompt": prefix})["tokens"])
    r = post("/v1/completions", {"model": MODEL, "prompt": prefix + text, "max_tokens": 1,
                                 "temperature": 0, "prompt_logprobs": 0})
    plp = r["choices"][0]["prompt_logprobs"][start:]
    lps = [next(iter(d.values()))["logprob"] for d in plp if d]
    nll = -sum(lps) / len(lps)
    print(f"{lang}: {len(lps)} tokens, mean NLL {nll:.4f}, perplexity {math.exp(nll):.3f}")
