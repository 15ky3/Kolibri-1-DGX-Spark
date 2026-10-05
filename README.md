# Kolibri-1 on ONE DGX Spark (stock vLLM 0.30.0)

Serves [Aleph Alpha's Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1),
a 78 B / 3.46 B-active MoE reasoning model (German + English, tool calling, up
to 1M context), on a single DGX Spark (GB10, sm121) with the **stock
`vllm/vllm-openai:v0.30.0` image**. No image build.

Four checkpoints, two switches (`QUANT`, `ABLIT`):

| switch | checkpoint | on the GPU | decode 1 / 8 streams | KV pool (auto) |
|---|---|---|---|---|
| **`QUANT=nvfp4` (default)** | [`iSkye/Kolibri-1-NVFP4-Experts`](https://huggingface.co/iSkye/Kolibri-1-NVFP4-Experts): routed experts NVFP4, the rest the original FP8 | **42.8 GiB** | **52 / 266 tok/s** | 48 GiB ≈ 4.0M tokens |
| `QUANT=fp8` | [`Aleph-Alpha/Kolibri-1`](https://huggingface.co/Aleph-Alpha/Kolibri-1), the original | 73.6 GiB | 48 / 179 tok/s | 22 GiB ≈ 1.8M tokens |
| `ABLIT=1` | [`iSkye/Kolibri-1-heretic-NVFP4-Experts`](https://huggingface.co/iSkye/Kolibri-1-heretic-NVFP4-Experts): refusals removed with Heretic, experts NVFP4 | 42.8 GiB | 54 / 272 tok/s | 48 GiB ≈ 4.0M tokens |
| `ABLIT=1 QUANT=fp8` | [`iSkye/Kolibri-1-heretic`](https://huggingface.co/iSkye/Kolibri-1-heretic): refusals removed with Heretic, FP8 | 73.6 GiB | as `fp8` | 22 GiB ≈ 1.8M tokens |

```
./download.sh              # the checkpoint QUANT selects (~46 GB nvfp4 / ~79 GB fp8), resumable
./start.sh                 # pre-flight, launch, wait for /health; serves on :8895
tests/smoke.py             # health, chat (reasoning off/on), tool call
./stop.sh                  # graceful, archives the container log

QUANT=fp8 ./download.sh && QUANT=fp8 ./start.sh     # the original FP8 instead
ABLIT=1 ./download.sh && ABLIT=1 ./start.sh         # abliterated, NVFP4 experts (see below)
```

`./start.sh --no-launch` prints the memory budget and the full docker command
without running anything (also written to `.last_launch.sh`).

## Requirements

- DGX Spark / GB10 (121.6 GiB unified memory), Docker with the NVIDIA runtime
- `vllm/vllm-openai:v0.30.0` (arm64; `start.sh` pulls it if missing)
- disk for the checkpoint (~46 GB nvfp4, ~79 GB fp8, ~80 GB heretic fp8); the `hf` CLI for
  `download.sh` (falls back to the one in the image)
- **free memory at start: ~108 GiB (nvfp4) / ~112 GiB (fp8)**: this does not
  run beside another large model

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

## Checkpoints

### NVFP4 experts (default)

[`iSkye/Kolibri-1-NVFP4-Experts`](https://huggingface.co/iSkye/Kolibri-1-NVFP4-Experts)
is built from the original by `tools/quantize_experts_nvfp4.sh`:

| part | format |
|---|---|
| routed experts (70 of the 73.4 GiB) | FP8 block → fp32 (exact) → NVFP4: FP4 E2M1, group 16, FP8 E4M3 group scales, fp32 global scale; gate/up share it (vLLM fuses them) |
| attention, shared expert | FP8 block, unchanged (`weight_scale_inv` renamed to `weight_scale`, same meaning) |
| router, norms, embeddings, LM head | unchanged |

Weight-only (NVFP4A16): the scales come from the group max (the weight recipe
ModelOpt uses), so no calibration data is needed. The script runs in the vLLM
image with the image's own compressed-tensors 0.17.0, so the output is exactly
what vLLM loads: a mixed `compressed-tensors` config (FP8_BLOCK group +
NVFP4A16 group). 12 min on the GB10, 42.6 GiB, mean relative weight error 9.4 %.
vLLM serves the experts with the **Marlin** NVFP4 MoE kernel (FP4 weights,
BF16 math).

### Compared (same box, same image)

| | `fp8` | `nvfp4` |
|---|---|---|
| weights in memory | 73.55 GiB | **42.77 GiB** |
| decode 1 / 2 / 4 / 8 streams tok/s | 48.1 / 80.4 / 127.2 / 179.3 | 52.1 / 93.5 / 162.6 / **265.9** (+8 / +16 / +28 / **+48 %**) |
| prefill @21k / @152k / @238k / 978k avg | ~5.3k / 2.6k / 1.9k / 578 | 5.5k / 2.6k / 1.9k / 591 tok/s |
| needle in a haystack | PASS 5k–978k | PASS 21k–978k |
| smoke test | 5/5 | 5/5 |
| perplexity de / en, memorized | 1.675 / 1.126 | 1.736 / 1.140 (+3.6 / +1.2 %) |
| perplexity de / en, freshly written | 8.385 / 13.78 | 8.83 / 13.34 (+5.3 / −3.2 %) |
| weight loading (`lazy` / `eager`) | ~500 s / 80 s | ~270 s / 32–44 s |

NVFP4 perplexities are means of 3 runs (Marlin's reductions are not
bit-deterministic, ± ~2 %); FP8 is deterministic. German loses a little more
than English. Decode gains most with several streams because each extra
stream pulls in more experts, and those are what got smaller.

### Abliterated: Heretic (`ABLIT=1`)

[`iSkye/Kolibri-1-heretic`](https://huggingface.co/iSkye/Kolibri-1-heretic) is
the original FP8 with its refusal behaviour removed by
[Heretic](https://heretic-project.org) (Arbitrary-Rank Ablation, LoRA rank 50
merged back into FP8). Only `self_attn.o_proj` and `mlp.shared_experts.down_proj`
changed; the routed experts, tensor names, config and tokenizer are the
original's. `ABLIT=1` serves it at either `QUANT`:
[`iSkye/Kolibri-1-heretic-NVFP4-Experts`](https://huggingface.co/iSkye/Kolibri-1-heretic-NVFP4-Experts)
(default) is the same conversion as above applied to the heretic FP8. Its NVFP4
experts are bit-identical to those of `iSkye/Kolibri-1-NVFP4-Experts`; the two
differ only in `o_proj` and the shared `down_proj`.

> **Warning:** this model answers requests the original declines. You are
> responsible for how you use it.

| | `fp8` original | heretic `fp8` | heretic `nvfp4` |
|---|---|---|---|
| refusals, 100 harmful prompts (Heretic's eval, FP8) | 100/100 | 3/100 | follows the heretic FP8 |
| KL divergence on harmless prompts (Heretic's eval) | 0 | 0.071 | |
| perplexity de / en, memorized | 1.675 / 1.126 | 2.327 / 1.221 (+39 / +8 %) | 2.398 / 1.258 |
| perplexity de / en, freshly written | 8.385 / 13.78 | 9.548 / 13.97 (+14 / +1 %) | 9.734 / 14.31 |
| decode 1 / 8 streams | 48 / 179 tok/s | same as `fp8` | 54 / 272 tok/s |
| smoke test | 5/5 | 5/5 | 5/5 |
| weights / KV / host MemAvailable | 73.55 GiB / 22 / ~12.5 GiB | same; ~16 GiB right after a `lazy` start | 42.77 GiB / 48 / ~21 GiB |

The ablation costs noticeably more German quality than English; NVFP4 adds the
usual ~2–3 % on top (heretic NVFP4 perplexities: mean of 3 runs). The heretic FP8 shards are
5 GB (the original's 2.5 GB): with `LOAD_STRATEGY=eager` the engine keeps ~12 GB
of host memory after loading instead of ~2, so stay with the default `lazy`.

`download.sh` and `start.sh` also accept a complete copy in
`~/models/<repo name>` (`LOCAL_MODELS`), the layout `hf download --local-dir`
writes, so a checkpoint that already sits there is not fetched again.

To rebuild an NVFP4 checkpoint yourself: `tools/quantize_experts_nvfp4.sh`
(source = the FP8 that `ABLIT` selects; writes `~/models/Kolibri-1-NVFP4-Experts`
or `~/models/Kolibri-1-heretic-NVFP4-Experts`, ~8–12 min). `start.sh` serves
that directory automatically when the hub cache has no snapshot of the repo.

## Memory

GB10 has one 121.6 GiB pool for CPU and GPU. The budget is computed, not guessed:

| GiB | `nvfp4` | `fp8` |
|---|---|---|
| weights (`WEIGHTS_GIB`, measured 42.77 / 73.55) | 43 | 73.4 |
| KV cache, pinned via `--kv-cache-memory-bytes` (`KV_CACHE_GIB`, auto) | 48 | 22 |
| activations, CUDA graphs, workspaces (`OVERHEAD_GIB`) | 7 | 7 |
| **budget → `--gpu-memory-utilization`** | **98 → 0.806** | **102.4 → 0.842** |
| host MemAvailable, measured under load | ~17.6 GiB | ~12.5 GiB |

`start.sh` refuses to launch unless `MemAvailable ≥ budget + HOST_MIN_FREE_GIB`
(10). Stop any other model first (`--force` overrides).

**KV pool.** Only 10 of the 50 layers are full attention (4 KV heads × 128,
FP8, no RoPE); the other 40 keep a 513-token sliding window. vLLM reports the
pool as "how many max-length requests fit × `max_model_len`", reserving window
+ prefill chunk per request for the sliding layers. The physical
full-attention capacity is ~26 % higher (one block = 16 tokens × 10 layer
tensors × 1 KiB = 160 KiB; each 262k request is charged 16,384 full + 4,228
sliding blocks):

| `KV_CACHE_GIB` | reported tokens (262k) | physical slots | note |
|---|---|---|---|
| 8 | 666,780 | ~0.84M | |
| 16 | 1,576,035 (at 1M context) | ~1.68M | |
| 22 | 1,833,672 | ~2.31M | `fp8` auto; the most `fp8` should get |
| **48** | **4,000,735** (4,728,121 at 1M) | **~5.03M** | **`nvfp4` auto** |

One 1M-token request needs ≥ 12.6 GiB, and `start.sh` refuses a
`MAX_MODEL_LEN` the pool cannot hold once.

Weights, KV pool, CUDA graphs and activation workspace are allocated at start
and do not grow. Under load (8 streams plus a 152k prefill) host memory moved
by < 1 GiB. A watchdog (`scripts/memwatch.sh`) stops the container if
`MemAvailable` stays below 4 GiB for three 2-s samples, before the driver or the
OOM killer pick a victim.

## Kernels on GB10 (sm121)

| part | kernel | note |
|---|---|---|
| attention | FlashInfer | bf16 queries, fp8 KV, xqa decode, per-layer sliding window |
| routed experts, `nvfp4` | Marlin NVFP4 MoE | FP4 weights, BF16 math |
| routed experts, `fp8` | Triton fused MoE, FP8 128×128 blocks | FlashInfer CUTLASS does block FP8 only on SM90; DeepGEMM is off on purpose |
| dense FP8 linear | CUTLASS block-scaled GEMM | `LINEAR_BACKEND` switches it |
| router | Kolibri's sigmoid routing (plugin) | top-k on logits + bias, weights from unbiased sigmoid, fp32 |
| sampling | FlashInfer top-k/top-p | |

- DeepGEMM stays disabled (`VLLM_USE_DEEP_GEMM=0`): Kolibri's block scales are
  fp32 and must not be rounded to UE8M0. The plugin refuses to run otherwise.
- For `fp8`, vLLM ships no GB10 Triton table for E=384. `files/moe_configs/`
  carries vLLM's own GB10 E=512 table under the E=384 name, loaded via
  `VLLM_TUNED_CONFIG_FOLDER`. A table tuned on this box is in
  `files/moe_configs/tuned/` (`tools/tune_moe.sh`); it measured within noise.

## Configuration

Precedence: exported environment > `.env` > `.env.sample`. Every knob is
commented in `.env.sample`; the ones that matter most:

| Variable | Default | |
|---|---|---|
| `QUANT` | nvfp4 | `nvfp4` or `fp8`; picks `MODEL_ID`, `WEIGHTS_GIB` and the auto KV pool |
| `ABLIT` | 0 | `1` = the Heretic-abliterated variant of the same `QUANT` |
| `PORT` | 8895 | `BIND` 0.0.0.0 |
| `MAX_MODEL_LEN` | 262144 | `1048576` works; above 262144 `start.sh` adds `--hf-overrides max_position_embeddings` (model card: extrapolated, ≤ 262k recommended for quality) |
| `KV_CACHE_GIB` | auto | 48 (nvfp4) / 22 (fp8); the memory lever, GMU follows it |
| `MAX_NUM_SEQS` | 8 | concurrent requests |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | chunked-prefill chunk |
| `KV_CACHE_DTYPE` | fp8 | what Aleph Alpha evaluated with |
| `LOAD_STRATEGY` | lazy | weight loading via mmap; `eager` is much faster with identical results (table above) |
| `SPEC` | off | `ngram`: prompt-lookup speculation, for copy-heavy output only (see below) |
| `LINEAR_BACKEND` | auto | dense FP8 GEMM backend (`triton`, …) |
| `REASONING` / `TOOLS` | 1 / 1 | `kolibri1` reasoning parser; `kolibri1` (Hermes) tool parser |
| `API_KEY` | empty | empty + `BIND=0.0.0.0` → `start.sh` warns |
| `MODEL_ID` / `MODEL_REVISION` / `MODEL_DIR` | empty | override the checkpoint: another repo, a pinned snapshot, or a local directory |

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

## Measured with `fp8` (2026-10-04)

| | |
|---|---|
| weights on the GPU | 73.55 GiB |
| decode | 1 stream 47–49 tok/s · 4: 127–130 · 8: 177–182 tok/s aggregate |
| prefill | ~5.3k tok/s @21k → 1.9k @238k → 578 avg over 978k |
| needle in a haystack | all PASS from 5k to **978k tokens** (start, middle and end positions) |
| smoke test | 5/5 on every configuration tried |

### Why FP8 decode does not go faster

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

What does help is moving fewer bytes, which is what the `nvfp4` checkpoint
does: the experts shrink from 1.18 to ~0.66 GB per token. The LM head (0.66 GB,
BF16) is left untouched.

## Layout

| Path | |
|---|---|
| `start.sh`, `stop.sh`, `download.sh` | launch, graceful stop (log archived to `logs/archive/`), checkpoint download |
| `.env.sample` | the launch profile |
| `scripts/common.sh` | env loading, `QUANT` resolution, snapshot resolution, helpers |
| `scripts/memwatch.sh` | memory watchdog |
| `scripts/drop-model-cache.py` | evict the checkpoint from the page cache (cold-load timing, no root) |
| `files/plugin/` | Aleph Alpha's plugin, unmodified |
| `files/moe_configs/` | E=384 Triton MoE table for `fp8` (vLLM's GB10 E=512 table); `orig/` = the same, `tuned/` = tuned on this box + tuner CSV |
| `tests/smoke.py` | functional check |
| `tests/ppl.py` | perplexity of memorized and freshly written German/English texts inside the chat frame |
| `tests/bench_decode.py`, `tests/bench_copy.py` | decode benchmarks (prose at 1–8 streams; copy-heavy single stream) |
| `tools/quantize_experts_nvfp4.sh`, `.py` | build the NVFP4-experts checkpoint from the FP8 original |
| `tools/tune_moe.sh`, `tools/tune_moe.py` | re-tune the FP8 MoE table on an idle GPU (~25 min) |
| `tools/benchmark_moe.py` | from vLLM v0.30.0, unmodified, used by the tuner |

## Open points

1. W4A4 NVFP4 (FP4 activations too) would need activation calibration; there
   is no transformers implementation of Kolibri, so it would have to run inside
   vLLM. Mainly a prefill gain.
2. `OVERHEAD_GIB=7` held for both checkpoints at 262k and 1M; not tested lower.

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
- **[compressed-tensors](https://github.com/vllm-project/compressed-tensors)**
  (vLLM project): the quantization and packing code behind the NVFP4 checkpoint,
  and the Marlin NVFP4 MoE kernel in vLLM that serves it.
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)**: attention and sampling kernels.
- **[Triton](https://github.com/triton-lang/triton)**: the fused-MoE kernels.
- **[NVIDIA](https://www.nvidia.com)**: [CUTLASS](https://github.com/NVIDIA/cutlass)
  (block-scaled FP8 GEMM), CUDA and the DGX Spark platform.
- **[PyTorch](https://pytorch.org)**: torch.compile and CUDA graphs underneath vLLM.
- **[Hugging Face](https://huggingface.co)**: model hosting, the `hf` CLI and
  [transformers](https://github.com/huggingface/transformers).
- **[b12x](https://pypi.org/project/b12x/)**: SM12x CuTe-DSL kernels,
  benchmarked here as an alternative dense GEMM.
- **[Heretic](https://heretic-project.org)**: the abliteration (ARA) behind the
  `ABLIT=1` checkpoint.
- **[MiaAI Lab](https://x.com/MiaAI_lab)**: the
  [Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  recipe, whose shape this one follows (single-Spark launcher, `.env` profile,
  derived memory budget, memory watchdog, graceful stop). No code was taken
  from it.

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright 2026
[15ky3](https://github.com/15ky3). Third-party files keep their own licenses
as noted above (both vendored components are Apache-2.0 as well).
