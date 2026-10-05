#!/usr/bin/env python3
"""Re-quantize a Kolibri-1 checkpoint's routed experts from FP8 (128x128 blocks) to NVFP4.

Runs inside the vLLM image (see tools/quantize_experts_nvfp4.sh), using the
image's own compressed-tensors, so the output is exactly the format vLLM loads.

  routed experts      FP8 block -> fp32 (exact) -> NVFP4 (group 16, FP8 E4M3
                      group scales, fp32 global scale), weight-only (NVFP4A16).
                      gate_proj and up_proj of one expert share their global
                      scale: vLLM fuses them into w13.
  attention, shared   kept FP8 block; `weight_scale_inv` renamed to
  expert              `weight_scale` (same meaning: dequant multiplier) —
                      bit-exact.
  everything else     copied unchanged (router gate + bias, norms,
                      embeddings, LM head).

Weights are RTN from the per-group max, which is also what ModelOpt does for
NVFP4 weights; there are no activation scales to calibrate in NVFP4A16.

Output: one safetensors shard per decoder layer plus one for the rest, an
index, config.json with a mixed compressed-tensors quantization_config, and
the tokenizer / generation files. Prints the reconstruction error of sampled
experts as it goes.
"""
import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict

import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, NVFP4A16
from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj)\.weight$")
LAYER = re.compile(r"^model\.layers\.(\d+)\.")
BLOCK = 128
GROUP = 16

TARGET_FP8 = [r"re:.*self_attn\.(q|k|v|o)_proj$", r"re:.*shared_experts\.(gate|up|down)_proj$"]
TARGET_FP4 = [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"]
IGNORE = ["lm_head", r"re:.*mlp\.gate$"]


def dequant_fp8_block(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 weight [N, K] x fp32 block scale [ceil(N/128), ceil(K/128)] -> fp32."""
    n, k = w.shape
    s_full = s.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return w.to(torch.float32) * s_full


def nvfp4(w: torch.Tensor, gscale: torch.Tensor, scheme: QuantizationScheme) -> dict:
    """fp32 [N, K] -> {weight_packed, weight_scale, weight_global_scale}."""
    n, k = w.shape
    g = w.view(n, k // GROUP, GROUP)
    scale, _ = calculate_qparams(g.amin(-1), g.amax(-1), scheme.weights, global_scale=gscale)
    return NVFP4PackedCompressor.compress(
        {"weight": w, "weight_scale": scale, "weight_global_scale": gscale}, scheme)


def nvfp4_error(w: torch.Tensor, sd: dict, scheme: QuantizationScheme) -> float:
    back = NVFP4PackedCompressor.decompress(dict(sd), scheme)["weight"].float()
    return ((back - w).norm() / w.norm()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="FP8 snapshot directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--layers", type=int, nargs="*", help="only these layers (testing)")
    ap.add_argument("--sample-every", type=int, default=97, help="error check every Nth expert")
    a = ap.parse_args()

    dev = torch.device("cuda")
    os.makedirs(a.dst, exist_ok=True)
    scheme4 = QuantizationScheme(targets=TARGET_FP4, **NVFP4A16)
    wmap = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]

    by_layer: dict = defaultdict(list)
    for name in wmap:
        m = LAYER.match(name)
        by_layer[int(m.group(1)) if m else "rest"].append(name)
    layers = sorted(k for k in by_layer if k != "rest")
    if a.layers:
        layers = [l for l in layers if l in a.layers]

    handles: dict = {}

    def get(name: str) -> torch.Tensor:
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(a.src, f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    new_map: dict = {}
    total = 0
    errs: list = []
    t_start = time.time()
    for key in layers + ["rest"]:
        t0 = time.time()
        names = by_layer[key]
        out: dict = {}
        experts: dict = defaultdict(dict)
        for name in names:
            m = EXPERT.match(name)
            if m:
                experts[int(m.group(3))][m.group(4)] = m.group(1)
            elif name.endswith(".weight_scale_inv"):
                base = name[: -len(".weight_scale_inv")]
                if not EXPERT.match(base + ".weight"):
                    out[base + ".weight_scale"] = get(name).contiguous()
            else:
                out[name] = get(name).contiguous()

        for e in sorted(experts):
            p = experts[e]
            w = {proj: dequant_fp8_block(get(f"{p[proj]}.weight").to(dev),
                                         get(f"{p[proj]}.weight_scale_inv").to(dev))
                 for proj in ("gate", "up", "down")}
            gu = torch.stack([w["gate"].abs().amax(), w["up"].abs().amax()]).amax()
            g_gu = generate_gparam(-gu, gu)
            g_d = generate_gparam(w["down"].amin(), w["down"].amax())
            for proj, gs in (("gate", g_gu), ("up", g_gu), ("down", g_d)):
                sd = nvfp4(w[proj], gs, scheme4)
                if e % a.sample_every == 0:
                    errs.append(nvfp4_error(w[proj], sd, scheme4))
                for k, v in sd.items():
                    out[f"{p[proj]}.{k}"] = v.to("cpu").contiguous()

        fname = f"model-{key:02d}-of-{len(layers):02d}.safetensors" if key != "rest" \
            else "model-rest.safetensors"
        save_file(out, os.path.join(a.dst, fname), metadata={"format": "pt"})
        size = sum(v.nbytes for v in out.values())
        total += size
        new_map.update({k: fname for k in out})
        recent = errs[-9:]
        print(f"layer {key}: {len(experts)} experts, {size / 2**30:.2f} GiB, "
              f"{time.time() - t0:.1f}s, rel. error (sampled) "
              f"{(sum(recent) / len(recent)) if recent else float('nan'):.4f}", flush=True)
        handles.clear()

    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    qcfg = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(targets=TARGET_FP8, format="float-quantized", **FP8_BLOCK),
            "group_1": QuantizationScheme(targets=TARGET_FP4, format="nvfp4-pack-quantized", **NVFP4A16),
        },
        ignore=IGNORE,
        format="mixed-precision",
        quantization_status="compressed",
    )
    q = qcfg.model_dump(exclude_none=True)
    q["quant_method"] = "compressed-tensors"
    cfg["quantization_config"] = json.loads(json.dumps(q, default=str))
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)

    # every companion file of the source (tokenizer, chat_template.jinja,
    # generation config, remote-code files, license) — not the weights, the
    # index or config.json (rewritten above), and not the model card
    skip = {"config.json", "model.safetensors.index.json", "README.md", ".gitattributes"}
    for f in sorted(os.listdir(a.src)):
        src = os.path.join(a.src, f)
        if f in skip or f.endswith(".safetensors") or not os.path.isfile(src):
            continue
        shutil.copy(src, os.path.join(a.dst, f))
        print("copied", f)

    print(f"done: {total / 2**30:.2f} GiB in {time.time() - t_start:.0f}s, "
          f"mean sampled rel. error {sum(errs) / max(len(errs), 1):.4f} over {len(errs)} matrices")


if __name__ == "__main__":
    main()
