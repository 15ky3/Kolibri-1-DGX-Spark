# Kolibri-1 on ONE DGX Spark (stock vLLM 0.30.0)

Serves [`Aleph-Alpha/Kolibri-1`](https://huggingface.co/Aleph-Alpha/Kolibri-1) —
Aleph Alpha's 78 B / 3.46 B-active MoE reasoning model (German + English, tool
calling) — on a single GB10 (sm121) with the **stock `vllm/vllm-openai:v0.30.0`
image**. No quantization, no image build: the official FP8 checkpoint as published.

```
./download.sh              # ~79 GB into $HF_HOME (resumable)
./start.sh                 # pre-flight, launch, wait for /health; serves on :8895
tests/smoke.py             # health, chat (reasoning off/on), tool call
./stop.sh                  # graceful, archives the container log
```

`./start.sh --no-launch` prints the memory budget and the full docker command
without running anything (also written to `.last_launch.sh`).

## Why stock vLLM 0.30 works

Kolibri needs Aleph Alpha's vLLM plugin
([`aleph-alpha-inference`](https://github.com/Aleph-Alpha/aleph-alpha-inference) 1.0.0,
Apache-2.0), whose wheel pins `vllm>=0.29,<0.30`. The pin is the only obstacle:

- Every symbol the plugin imports exists in 0.30.0, and
  `vllm/model_executor/models/qwen3_moe.py` — the base classes `Kolibri1*`
  derive from — is **byte-identical** between v0.29.0 and v0.30.0.
- The other touched files (`fused_moe/layer.py`, `router/gate_linear.py`,
  `attention/attention.py`, `parser/qwen3.py`, `parser/engine/adapters.py`,
  `models/config.py`, `models/utils.py`, `rotary_embedding`, `config/parallel.py`)
  changed only additively; no signature the plugin calls changed.
- The image's transformers 5.17 parses the config through the plugin's
  `Kolibri1Config(Qwen3MoeConfig)` with all extra fields intact
  (`layer_types`, `sliding_window=513`, `head_dim`, `shared_expert_intermediate_size`)
  — tested on the host against the real `config.json`.

So the plugin is vendored unmodified in `files/plugin/` (package + dist-info,
from the PyPI wheel, sha256 `5a0ca118…d6f211`) and **bind-mounted** into the
image's `dist-packages`. The dist-info carries the `vllm.general_plugins` entry
point, which is all vLLM needs to load it. Nothing is pip-installed, so no
dependency resolver ever sees the pin.

## Memory

GB10 has one 121.6 GiB pool for CPU and GPU. The budget is computed, not guessed:

| | GiB |
|---|---|
| FP8 weights (+ BF16 embeddings, LM head, norms, routers) | 73.4 |
| KV cache, pinned via `--kv-cache-memory-bytes` (`KV_CACHE_GIB`) | 20 |
| activations, CUDA graphs, workspaces (`OVERHEAD_GIB`) | 7 |
| **budget → `--gpu-memory-utilization`** | **100.4 → 0.825** |

`start.sh` refuses to launch unless `MemAvailable ≥ budget + HOST_MIN_FREE_GIB`
(10), i.e. ~110 GiB — **it does not run beside another large model**; stop that
one first (`--force` overrides).

KV is cheap here: only 10 of 50 layers are full attention (4 KV heads × 128,
FP8); the other 40 keep a 513-token sliding window. Measured pools: 8 GiB =
666,780 tokens, 16 GiB = 1,576,035 (at 1M), 20 GiB (default) = 1,666,977 (at
262k). At 20 the host keeps ~14 GiB available (16: ~17–18) — don't go higher. One 1M request needs ≥ 12.6 GiB, and
`start.sh` refuses a `MAX_MODEL_LEN` the pool cannot hold once.

A watchdog (`scripts/memwatch.sh`) stops the container if `MemAvailable` stays
below 4 GiB for three samples, before the driver or the OOM killer pick a victim.

## Kernels on GB10 (sm121)

- **MoE**: FP8 128×128 block weights + dynamic activations → **Triton** fused MoE.
  FlashInfer CUTLASS supports block FP8 only on Hopper (SM90), DeepGEMM is off
  (Kolibri's fp32 block scales must not be rounded to UE8M0 — the plugin and
  `start.sh` both force `VLLM_USE_DEEP_GEMM=0`).
- **Triton tile config**: vLLM ships no GB10 table for Kolibri's E=384.
  `files/moe_configs/` carries one tuned on this box (`tools/tune_moe.sh`),
  picked up via `VLLM_TUNED_CONFIG_FOLDER`; `orig/` keeps the E=512 copy it
  started from.
- **Dense FP8 linear**: CUTLASS block-scaled GEMM (auto). `LINEAR_BACKEND`
  switches it; b12x was tried and measured no gain.
- **Attention**: left to vLLM (FP8 KV + per-layer sliding window).
  `ATTENTION_BACKEND=TRITON_ATTN` is the fallback if the auto choice fails.
- **Router**: Kolibri's own sigmoid "select on logits + bias, weight by unbiased
  sigmoid" routing (plugin `custom_routing_function`), fp32 router logits.

## Configuration

Precedence: exported environment > `.env` > `.env.sample`. All knobs are in
`.env.sample` with comments; the ones that matter most:

| Variable | Default | |
|---|---|---|
| `PORT` | 8895 | |
| `MAX_MODEL_LEN` | 262144 | > 262144 adds `--hf-overrides max_position_embeddings` (card: extrapolated, ≤ 262 k recommended) |
| `KV_CACHE_GIB` | 8 | the one memory lever; GMU follows it |
| `MAX_NUM_SEQS` | 8 | |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | chunked-prefill chunk |
| `KV_CACHE_DTYPE` | fp8 | what Aleph Alpha evaluated with |
| `REASONING` / `TOOLS` | 1 / 1 | `kolibri1` reasoning parser; `kolibri1` (Hermes) tool parser |
| `API_KEY` | empty | empty + `BIND=0.0.0.0` → start.sh warns |
| `MODEL_REVISION` | empty | pin a snapshot sha |

Sampling defaults come from the checkpoint's `generation_config.json`
(temperature 1.0, top_p 0.97, top_k 128 — Aleph Alpha's recommendation).

## Reasoning and tools

Thinking is **on by default** (high effort). Per request, through the template:

```json
{"chat_template_kwargs": {"reasoning_effort": "none"}}      // or low | medium | high
```

`none` (or `enable_thinking: false`) answers immediately. The reasoning text is
returned in `message.reasoning`, the answer in `message.content`. Tool calls use
the standard `tools` field and come back as structured `tool_calls`.

## Dashboard

Registered in `~/dashboard` as recipe `kolibri-1` (family "Kolibri 1"): Download,
Start/Stop, context / KV-pool / concurrency switches. It is marked `local` there —
there is no upstream repo, so "Update" re-runs prepare instead of `git pull`.

## Measured (2026-10-04)

| | |
|---|---|
| time to `/health`, cold page cache | ~2 min with `LOAD_STRATEGY=eager` (weights 80–92 s); torch.compile cached |
| load strategy, cold | eager 80 s · lazy mmap 523 s (~150 MB/s) · prefetch 458 s (cache evicts itself) — NVMe does 5.5 GB/s |
| weights on the GPU | 73.55 GiB |
| attention / MoE / dense FP8 | FlashInfer (bf16 query, fp8 KV, xqa decode) / Triton FP8 / CUTLASS block FP8 |
| decode | 1 stream 47–49 tok/s · 4: 127–130 · 8: 177–182 tok/s aggregate |
| prefill | ~5.3k tok/s @21k → 1.9k @238k → 578 avg over 978k |
| needle in a haystack | 6/6 from 5k to **978k tokens** (1M context, KV 16 GiB) |
| smoke test | 5/5 |

### Why decode does not go faster

One decoded token reads ~3.84 GB of weights: attention projections 1.70 GB
(FP8), six routed experts 1.18, shared expert + router 0.30, BF16 LM head 0.66.
At 48.5 tok/s that is ~186 GB/s — about 85 % of what GB10's LPDDR5X delivers in
practice. Kernel changes therefore measure as noise:

| change | 1 / 4 / 8 streams tok/s |
|---|---|
| baseline | 47.3 / 130.4 / 182.0 |
| E=384 MoE table tuned on the box (`tools/tune_moe.sh`, 1.22× at M=1, ≤ 7 % elsewhere) | 48.1 / 127.2 / 179.3 |
| + `b12x` (SM12x CuTe-DSL) dense FP8 GEMM instead of CUTLASS | 48.5 / 127.5 / 180.8 |

At M ≥ 2 the MoE kernels already stream ~230 GB/s (M=8: 46 experts × 3.9 MB in
0.78 ms). The tuned table is kept (it is never slower); b12x is not.

**N-gram speculation** (`SPEC=ngram`): 2.26× on output that copies its input
(code rewrite: 49 → 111 tok/s, output identical) but it halves free prose
(48 → 24 tok/s; k=2/min 4 still 32): vLLM falls back to the V1 runner without
async scheduling, and a rejected draft is a 5-token MoE verify. Opt-in only.

The remaining single-stream lever would be quantizing the LM head (0.66 GB of
the 3.84 GB per token) — deliberately not done.

## Open points

1. MoE tile table: re-run `tools/tune_moe.sh` after an image change.
2. `OVERHEAD_GIB=7` held at 262k and 1M; not tested lower.
