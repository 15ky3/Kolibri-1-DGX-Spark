#!/usr/bin/env python3
"""Tune the Triton fused-MoE tiles for Kolibri-1 on this GPU.

Runs inside the vLLM image (see tools/tune_moe.sh). Uses benchmark_config()
from vLLM v0.30.0's benchmarks/kernels/benchmark_moe.py (vendored next to this
file) but drives it itself: that script's model lookup does not know
Kolibri1ForCausalLM, and its full search space (1920 tiles x 18 batch sizes)
takes many hours on a GB10.

For each batch size it times the CURRENT table entry first (the baseline), then
every tile in a search space sized for that M, and keeps the faster. Batch
sizes not tuned keep the current table's entry, so the output is always a
complete table. Writes the new table and a CSV of baseline vs. best.
"""
import argparse
import itertools
import json
import os
import sys
import time
import types

try:
    import ray  # noqa: F401  (benchmark_moe imports it at module level)
except ImportError:
    stub = types.ModuleType("ray")
    stub.remote = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda c: c))
    sys.modules["ray"] = stub
    tq = types.ModuleType("ray.experimental.tqdm_ray")
    tq.tqdm = lambda x, **k: x
    sys.modules["ray.experimental"] = types.ModuleType("ray.experimental")
    sys.modules["ray.experimental.tqdm_ray"] = tq

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_moe as bm  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402

# Kolibri-1: 384 routed experts, top-6, expert hidden 512 (w13 = 2 x 512), d=2560
E, TOPK, N2, HIDDEN, BLOCK = 384, 6, 1024, 2560, [128, 128]


def space(m: int) -> list[dict]:
    """Tiles worth trying at M tokens. Small M is memory-bound: tiny BLOCK_M,
    every N/K/warps/stages combination. Large M gets bigger BLOCK_M."""
    bms = [16, 32] if m <= 32 else [16, 32, 64] if m <= 256 else [32, 64, 128, 256]
    out = []
    gms = [1, 16] if m <= 64 else [1, 16, 32, 64]
    for bm_, bn, bk, gm, w, s in itertools.product(
        bms, [32, 64, 128, 256], [64, 128, 256], gms, [4, 8], [2, 3, 4, 5]
    ):
        if not (bn % BLOCK[0] == 0 or BLOCK[0] % bn == 0):
            continue
        if not (bk % BLOCK[1] == 0 or BLOCK[1] % bk == 0):
            continue
        out.append(dict(BLOCK_SIZE_M=bm_, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                        GROUP_SIZE_M=gm, num_warps=w, num_stages=s))
    return out


def timed(cfg: dict, m: int, iters: int) -> float:
    try:
        return bm.benchmark_config(
            cfg, m, E, N2, HIDDEN, TOPK, torch.bfloat16,
            use_fp8_w8a8=True, use_int8_w8a16=False,
            num_iters=iters, block_quant_shape=BLOCK,
        )
    except (triton.runtime.autotuner.OutOfResources, RuntimeError):
        return float("inf")


def nearest(table: dict, m: int) -> dict:
    keys = [k for k in table if k.isdigit()]  # skip metadata such as triton_version
    return table[min(keys, key=lambda k: abs(int(k) - m))]


def ordered(table: dict) -> dict:
    meta = {k: v for k, v in table.items() if not k.isdigit()}
    return {**meta, **dict(sorted(((k, v) for k, v in table.items() if k.isdigit()),
                                  key=lambda kv: int(kv[0])))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="current table (json)")
    ap.add_argument("--out", required=True, help="tuned table (json)")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32, 48, 64])
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    torch.set_default_device("cuda")
    base = json.load(open(a.base))
    new = dict(base)
    report = []
    for m in a.batch:
        ref = {k: v for k, v in nearest(base, m).items()}
        t_ref = timed(ref, m, a.iters)
        best, t_best = ref, t_ref
        cands = space(m)
        t0 = time.time()
        for i, cfg in enumerate(cands):
            t = timed(cfg, m, a.iters)
            if t < t_best:
                best, t_best = cfg, t
            if i % 100 == 0:
                print(f"  M={m}: {i}/{len(cands)} best {t_best:.1f} us (base {t_ref:.1f})", flush=True)
        # confirm the winner with more iterations before trusting it
        t_best = timed(best, m, a.iters * 5)
        t_ref = timed(ref, m, a.iters * 5)
        if t_best >= t_ref:
            best, t_best = ref, t_ref
        new[str(m)] = best
        report.append((m, t_ref, t_best))
        print(f"M={m:5d}: base {t_ref:8.1f} us -> best {t_best:8.1f} us "
              f"({t_ref / t_best:4.2f}x) {best}  [{time.time() - t0:.0f}s]", flush=True)
        json.dump(ordered(new), open(a.out, "w"), indent=4)

    with open(os.path.splitext(a.out)[0] + ".csv", "w") as f:
        f.write("M,base_us,tuned_us,speedup\n")
        for m, r, b in report:
            f.write(f"{m},{r:.2f},{b:.2f},{r / b:.3f}\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
