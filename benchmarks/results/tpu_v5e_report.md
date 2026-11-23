# Thalassa experimental TPU v5e report

Status: **not run**

## Hardware and software

- Intended hardware: Google Cloud TPU v5e-8, using one XLA device.
- Python 3.12.
- PyTorch 2.8.0 and PyTorch/XLA 2.8.0.
- Command: `PJRT_DEVICE=TPU python benchmarks/tpu_v5e.py --workload all --dtype all --baselines`.

## Reference matrix

| workload | cells | batch | steps | path exercised |
|---|---:|---:|---:|---|
| explicit heat | 65,536 | 32 | 8 | explicit/direct |
| implicit heat | 4,096 | 16 | 1 | CG |
| implicit advection | 4,096 | 16 | 1 | BiCGStab |
| variable pointwise | 16,384 | 16 | 1 | unrolled 4x4 solve |
| Taylor–Green | 128x128 | 1 | 1 | 2D stencils and projected CG |

Each workload is parameterized over FP32, FP16, and BF16 and compared with a CPU FP64 result. The harness records first-call cost, median/p95 synchronized time, throughput, XLA compile/execute metrics, fallback operations, convergence, true residual, relative-L2 error, and post-warm-up compilation counts.

## Correctness and performance

No hardware measurements are checked in yet. Acceptance limits are relative L2 `2e-4` (FP32), `1e-2` (FP16), and `5e-2` (BF16), together with finite converged results, no CPU fallback operations, and no post-warm-up recompilation.

## CPU/CUDA comparison

Not measured in this workspace. The report imposes no TPU speedup requirement.

## Conclusion

The implementation is not accepted or rejected on TPU performance until the raw JSON is replaced by a complete v5e-8 run. This placeholder deliberately makes no performance claim.
