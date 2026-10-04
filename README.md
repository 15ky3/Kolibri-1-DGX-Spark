# Kolibri-1 on ONE DGX Spark (stock vLLM 0.30.0)

Serves [`Aleph-Alpha/Kolibri-1`](https://huggingface.co/Aleph-Alpha/Kolibri-1),
Aleph Alpha's 78 B / 3.46 B-active MoE reasoning model (German + English, tool
calling, up to 1M context), on a single DGX Spark (GB10, sm121) with the
**stock `vllm/vllm-openai:v0.30.0` image**. No quantization, no image build:
the official FP8 checkpoint as published.

```
./download.sh              # ~79 GB into $HF_HOME (resumable)
./start.sh                 # pre-flight, launch, wait for /health (~10 min); serves on :8895
tests/smoke.py             # health, chat (reasoning off/on), tool call
./stop.sh                  # graceful, archives the container log
```

`./start.sh --no-launch` prints the memory budget and the full docker command
without running anything (also written to `.last_launch.sh`).

At a glance (measured on one GB10, 2026-10-04):

| | |
|---|---|
| ready after | ~10 min from a cold page cache (~2 min with `LOAD_STRATEGY=eager`) |
| decode | ~48 tok/s single stream · ~180 tok/s aggregate at 8 streams |
| context | 262k native; **1M verified** (needle found at 978k tokens) |
| KV pool | 22 GiB ≈ 2.31M token slots (vLLM reports 1.83M, see below) |
| memory left to the host | ~12.5 GiB under load |

## Requirements

- DGX Spark / GB10 (121.6 GiB unified memory), Docker with the NVIDIA runtime
- `vllm/vllm-openai:v0.30.0` (arm64; `start.sh` pulls it if missing)
- ~80 GB free disk for the checkpoint, the `hf` CLI for `download.sh`
  (falls back to the one in the image)
- **~112 GiB of free memory at start** — this does not run beside another
  large model

## Why stock vLLM 0.30 works

Kolibri needs Aleph Alpha's vLLM plugin
([`aleph-alpha-inference`](https://github.com/Aleph-Alpha/aleph-alpha-inference) 1.0.0),
whose wheel pins `vllm>=0.29,<0.30`. The pin is the only obstacle:

- Every symbol the plugin imports exists in 0.30.0, and
  `vllm/model_executor/models/qwen3_moe.py`, which holds the base classes the `Kolibri1*`
  classes derive from, is **byte-identical** between v0.29.0 and v0.30.0.
- The other touched files (`fused_moe/layer.py`, `router/gate_linear.py`,
  `attention/attention.py`, `parser/qwen3.py`, `parser/engine/adapters.py`,
  `models/config.py`, `models/utils.py`, `rotary_embedding`, `config/parallel.py`)
  changed only additively; no signature the plugin calls changed.
- The image's transformers 5.17 parses the config through the plugin's
  `Kolibri1Config(Qwen3MoeConfig)` with all extra fields intact
  (`layer_types`, `sliding_window=513`, `head_dim`, `shared_expert_intermediate_size`).

So the plugin is vendored **unmodified** in `files/plugin/` (package + dist-info
from the PyPI wheel, sha256 `5a0ca118…d6f211`) and **bind-mounted** into the
image's `dist-packages`. The dist-info carries the `vllm.general_plugins` entry
point, which is all vLLM needs to load it. Nothing is pip-installed, so no
dependency resolver ever sees the pin.

## Memory

GB10 has one 121.6 GiB pool for CPU and GPU. The budget is computed, not guessed:

| | GiB |
|---|---|
| FP8 weights (+ BF16 embeddings, LM head, norms, routers), measured 73.55 | 73.4 |
| KV cache, pinned via `--kv-cache-memory-bytes` (`KV_CACHE_GIB`) | 22 |
| activations, CUDA graphs, workspaces (`OVERHEAD_GIB`) | 7 |
| **budget → `--gpu-memory-utilization`** | **102.4 → 0.842** |

`start.sh` refuses to launch unless `MemAvailable ≥ budget + HOST_MIN_FREE_GIB`
(10), i.e. ~112 GiB. Stop any other model first (`--force` overrides).

**KV pool.** Only 10 of the 50 layers are full attention (4 KV heads × 128,
FP8, no RoPE); the other 40 keep a 513-token sliding window. vLLM reports the
pool as "how many max-length requests fit × `max_model_len`", reserving window
+ prefill chunk per request for the sliding layers. The physical
full-attention capacity is ~26 % higher (one block = 16 tokens × 10 layer
tensors × 1 KiB = 160 KiB; each 262k request is charged 16,384 full + 4,228
sliding blocks):

| `KV_CACHE_GIB` | reported tokens | physical slots | host MemAvailable |
|---|---|---|---|
| 8 | 666,780 | ~0.84M | ~29 GiB |
| 16 | 1,576,035 (at 1M context) | ~1.68M | ~17 GiB |
| 20 | 1,666,977 | ~2.10M | ~14 GiB |
| **22 (default)** | **1,833,672** | **~2.31M** | **~12.5 GiB under load** |

Don't go above 22. One 1M-token request needs ≥ 12.6 GiB, and `start.sh`
refuses a `MAX_MODEL_LEN` the pool cannot hold once.

Weights, KV pool, CUDA graphs and activation workspace are allocated at start
and do not grow. Under load (8 streams plus a 152k prefill) host memory moved
by < 1 GiB. A watchdog (`scripts/memwatch.sh`) stops the container if
`MemAvailable` stays below 4 GiB for three 2-s samples, before the driver or the
OOM killer pick a victim.

## Kernels on GB10 (sm121)

| part | kernel | note |
|---|---|---|
| attention | FlashInfer | bf16 queries, fp8 KV, xqa decode, per-layer sliding window |
| routed experts | Triton fused MoE, FP8 128×128 blocks | FlashInfer CUTLASS does block FP8 only on SM90; DeepGEMM is off on purpose |
| dense FP8 linear | CUTLASS block-scaled GEMM | `LINEAR_BACKEND` switches it |
| router | Kolibri's sigmoid routing (plugin) | top-k on logits + bias, weights from unbiased sigmoid, fp32 |
| sampling | FlashInfer top-k/top-p | |

- DeepGEMM stays disabled (`VLLM_USE_DEEP_GEMM=0`): Kolibri's block scales are
  fp32 and must not be rounded to UE8M0. The plugin refuses to run otherwise.
- vLLM ships no GB10 Triton table for E=384. `files/moe_configs/` carries
  vLLM's own GB10 E=512 table under the E=384 name, loaded via
  `VLLM_TUNED_CONFIG_FOLDER`. A table tuned on this box is in
  `files/moe_configs/tuned/` (`tools/tune_moe.sh`); it measured within noise.

## Configuration

Precedence: exported environment > `.env` > `.env.sample`. Every knob is
commented in `.env.sample`; the ones that matter most:

| Variable | Default | |
|---|---|---|
| `PORT` | 8895 | `BIND` 0.0.0.0 |
| `MAX_MODEL_LEN` | 262144 | `1048576` works; above 262144 `start.sh` adds `--hf-overrides max_position_embeddings` (model card: extrapolated, ≤ 262k recommended for quality) |
| `KV_CACHE_GIB` | 22 | the memory lever; GMU follows it |
| `MAX_NUM_SEQS` | 8 | concurrent requests |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | chunked-prefill chunk |
| `KV_CACHE_DTYPE` | fp8 | what Aleph Alpha evaluated with |
| `LOAD_STRATEGY` | lazy | weight loading: lazy (mmap) ~500 s, `eager` 80 s, prefetch 458 s. Same weights, identical perplexity; `eager` only starts faster |
| `SPEC` | off | `ngram`: prompt-lookup speculation, for copy-heavy output only (see below) |
| `LINEAR_BACKEND` | auto | dense FP8 GEMM backend (`triton`, …) |
| `REASONING` / `TOOLS` | 1 / 1 | `kolibri1` reasoning parser; `kolibri1` (Hermes) tool parser |
| `API_KEY` | empty | empty + `BIND=0.0.0.0` → `start.sh` warns |
| `MODEL_REVISION` | empty | pin a snapshot sha |

Sampling defaults come from the checkpoint's `generation_config.json`
(temperature 1.0, top_p 0.97, top_k 128, Aleph Alpha's recommendation).

## Reasoning and tools

Thinking is **on by default** (high effort). Per request, through the template:

```json
{"chat_template_kwargs": {"reasoning_effort": "none"}}      // or low | medium | high
```

`none` (or `enable_thinking: false`) answers immediately. The reasoning text is
returned in `message.reasoning`, the answer in `message.content`, streamed or
not. Tool calls use the standard `tools` field and come back as structured
`tool_calls`.

Two log lines look alarming and are harmless:

- `incorrect regex pattern … fix_mistral_regex=True`: transformers assumes a
  Mistral tokenizer for any local `config.json` without `transformers_version`.
  Nothing is changed unless the flag is set. Kolibri is not Mistral: **do not
  set it**.
- `uncalibrated q_scale … fp8 attention`: queries run in bf16 here; only the KV
  cache is fp8.

## Measured (2026-10-04)

| | |
|---|---|
| time to `/health`, cold page cache | ~10 min with the default `lazy` (weights ~500 s); ~2 min with `eager` (80–92 s). torch.compile is cached after the first start |
| weights on the GPU | 73.55 GiB |
| decode | 1 stream 47–49 tok/s · 4: 127–130 · 8: 177–182 tok/s aggregate |
| prefill | ~5.3k tok/s @21k → 1.9k @238k → 578 avg over 978k |
| needle in a haystack | all PASS from 5k to **978k tokens** (start, middle and end positions) |
| smoke test | 5/5 on every configuration tried |

### Why decode does not go faster

One decoded token reads ~3.84 GB of weights: attention projections 1.70 GB
(FP8), six routed experts 1.18, shared expert + router 0.30, BF16 LM head 0.66.
At 48.5 tok/s that is ~186 GB/s, about 85 % of what GB10's LPDDR5X delivers in
practice. Kernel changes therefore measure as noise:

| change | 1 / 4 / 8 streams tok/s |
|---|---|
| baseline | 47.3 / 130.4 / 182.0 |
| E=384 MoE table tuned on the box (1.22× at M=1, ≤ 7 % elsewhere) | 48.1 / 127.2 / 179.3 |
| + `b12x` (SM12x CuTe-DSL) dense FP8 GEMM instead of CUTLASS | 48.5 / 127.5 / 180.8 |

At M ≥ 2 the MoE kernels already stream ~230 GB/s (M=8: 46 experts × 3.9 MB in
0.78 ms). Neither is in the default: the tuned table is in
`files/moe_configs/tuned/` (copy it over the live one to use it), b12x was removed.

**N-gram speculation** (`SPEC=ngram`): 2.26× on output that copies its input
(code rewrite: 49 → 111 tok/s, output identical), but it halves free prose
(48 → 24 tok/s; k=2 / min 4 still 32). vLLM falls back to the V1 runner without
async scheduling, and a rejected draft costs a 5-token MoE verify. Opt-in only.

The remaining single-stream lever would be quantizing the LM head (0.66 GB of
the 3.84 GB per token). This repo deliberately serves the published weights.

## Layout

| Path | |
|---|---|
| `start.sh`, `stop.sh`, `download.sh` | launch, graceful stop (log archived to `logs/archive/`), checkpoint download |
| `.env.sample` | the launch profile |
| `scripts/common.sh` | env loading, snapshot resolution, helpers |
| `scripts/memwatch.sh` | memory watchdog |
| `scripts/drop-model-cache.py` | evict the checkpoint from the page cache (cold-load timing, no root) |
| `files/plugin/` | Aleph Alpha's plugin, unmodified |
| `files/moe_configs/` | live E=384 Triton MoE table (vLLM's GB10 E=512 table); `orig/` = the same, `tuned/` = tuned on this box + tuner CSV |
| `tests/smoke.py` | functional check |
| `tests/bench_decode.py`, `tests/bench_copy.py` | decode benchmarks (prose at 1–8 streams; copy-heavy single stream) |
| `tests/ppl.py` | perplexity of fixed German/English texts inside the chat frame: identical numbers = identical computation |
| `tools/tune_moe.sh`, `tools/tune_moe.py` | re-tune the MoE table on an idle GPU (~25 min) |
| `tools/benchmark_moe.py` | from vLLM v0.30.0, unmodified, used by the tuner |
| `tools/quantize_experts_nvfp4.sh`, `.py` | build the NVFP4-experts checkpoint (experimental) |

## Experimental: NVFP4 experts (branch `nvfp4-experts`)

`tools/quantize_experts_nvfp4.sh` rebuilds the checkpoint with the routed
experts in NVFP4 (weight-only, NVFP4A16) and everything else untouched:

| part | format |
|---|---|
| routed experts (70 of the 73.5 GiB) | FP8 block → fp32 (exact) → NVFP4: FP4 E2M1, group 16, FP8 E4M3 group scales, fp32 global scale; gate/up share it (vLLM fuses them) |
| attention, shared expert | FP8 block, unchanged (`weight_scale_inv` renamed to `weight_scale`, same meaning) |
| router, norms, embeddings, LM head | unchanged |

It runs in the vLLM image with the image's own compressed-tensors 0.17.0, so
the output is exactly what vLLM loads: a mixed `compressed-tensors` config
(FP8_BLOCK group + NVFP4A16 group). No calibration data: NVFP4 weight scales
come from the group max (as in ModelOpt); only W4A4 would need activation
scales. 12 min on the GB10, 42.6 GiB output, mean relative weight error 9.4 %.

```
tools/quantize_experts_nvfp4.sh                    # -> ~/models/Kolibri-1-NVFP4-experts
MODEL_DIR=~/models/Kolibri-1-NVFP4-experts WEIGHTS_GIB=43 ./start.sh
```

vLLM serves the experts with the **Marlin** NvFp4 MoE kernel (FP4 weights,
BF16 math). Measured against FP8 (same KV 22 GiB):

| | FP8 | NVFP4 experts |
|---|---|---|
| weights in memory | 73.55 GiB | **42.77 GiB** |
| host MemAvailable after start | ~13 GiB | **~46 GiB** |
| decode 1 / 2 / 4 / 8 streams tok/s | 48.1 / 80.4 / 127.2 / 179.3 | 52.1 / 93.5 / 162.6 / **265.9** (+8 / +16 / +28 / **+48 %**) |
| prefill @21k / @152k / @238k | ~5.3k / 2.6k / 1.9k | 5.5k / 2.6k / 1.9k tok/s |
| needle 21k–238k | PASS | PASS |
| smoke test | 5/5 | 5/5 |
| perplexity de / en (memorized) | 1.675 / 1.126 | 1.736 / 1.140 (+3.6 / +1.2 %) |
| perplexity fresh de / en | 8.385 / 13.78 | 8.83 / 13.34 (+5.3 / −3.2 %) |

NVFP4 numbers are means of 3 runs: Marlin's reductions are not deterministic
(± ~2 % run to run), FP8 is. German loses a little more than English.

## Open points

1. Re-run `tools/tune_moe.sh` after any image change.
2. `OVERHEAD_GIB=7` held at 262k and 1M; not tested lower.

## Credits

This recipe only glues together other people's work:

- **[Aleph Alpha](https://aleph-alpha.com)**: the
  [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) model (Apache-2.0)
  and the [`aleph-alpha-inference`](https://github.com/Aleph-Alpha/aleph-alpha-inference)
  vLLM plugin (Apache-2.0, © 2026 Aleph Alpha GmbH), vendored unmodified in
  `files/plugin/` with its license.
- **[vLLM](https://github.com/vllm-project/vllm)** and its contributors: the
  serving engine and the `vllm/vllm-openai:v0.30.0` image. `tools/benchmark_moe.py`
  is vLLM's own kernel benchmark (Apache-2.0), vendored unmodified; the
  starting MoE table came from vLLM's GB10 tables.
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)**: attention and sampling kernels.
- **[Triton](https://github.com/triton-lang/triton)**: the fused-MoE kernels.
- **[NVIDIA](https://www.nvidia.com)**: [CUTLASS](https://github.com/NVIDIA/cutlass)
  (block-scaled FP8 GEMM), CUDA and the DGX Spark platform.
- **[PyTorch](https://pytorch.org)**: torch.compile and CUDA graphs underneath vLLM.
- **[Hugging Face](https://huggingface.co)**: model hosting, the `hf` CLI and
  [transformers](https://github.com/huggingface/transformers).
- **[b12x](https://pypi.org/project/b12x/)**: SM12x CuTe-DSL kernels,
  benchmarked here as an alternative dense GEMM.
- **[MiaAI Lab](https://x.com/MiaAI_lab)**: the
  [Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  recipe, whose shape this one follows (single-Spark launcher, `.env` profile,
  derived memory budget, memory watchdog, graceful stop). No code was taken
  from it.

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright 2026
[15ky3](https://github.com/15ky3). Third-party files keep their own licenses
as noted above (both vendored components are Apache-2.0 as well).
