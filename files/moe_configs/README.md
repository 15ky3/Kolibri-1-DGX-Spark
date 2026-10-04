# Triton fused-MoE tile configs

Mounted read-only at `/root/moe_configs` and selected with
`VLLM_TUNED_CONFIG_FOLDER`. vLLM looks up
`E=<experts>,N=<intermediate per shard>,device_name=<GPU>,dtype=…,block_shape=…`.

`E=384,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json`
is the live table: vLLM 0.30.0's E=512 GB10 table (identical to its E=256 one)
under the E=384 name. `orig/` holds the same file.

`tuned/` holds a table tuned on this GB10 on 2026-10-04 with `tools/tune_moe.sh`
for M = 1, 2, 4, 8, 16, 24, 32, 48, 64, plus the tuner's CSV: 1.22x at M=1,
1.00-1.07x elsewhere in the kernel, end-to-end decode within noise (decode is
bandwidth-bound, see the top-level README). Copy it over the live table to use it.
