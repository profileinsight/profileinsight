"""Rule 5: signs of common misconfigurations, recognised from the trace alone.

Some causes of a slow step are not in the program's own code but in the settings of the process: a flag, an
environment variable or a hook set by a library, a launcher or left over from debugging. The trace cannot show the
setting, but it shows its effect in a recognisable form. This rule looks for these forms and names, for each one
found, the evidence and the setting to check:
- math attention: operations launched inside aten::_scaled_dot_product_attention_math take at least MIN_SHARE of the
  kernel time: attention runs the unfused fallback. Check whether the flash, memory-efficient and cuDNN backends are
  enabled (torch.backends.cuda.*_sdp_enabled()) and whether the inputs (dtype, head size, mask) suit them.
- anomaly detection: at least MIN_CALLS kernels of aten::isnan inside autograd's evaluate_function: the backward
  checks every result for NaN, which costs kernels and a host synchronization per operation. Check
  torch.is_anomaly_enabled().
- driver allocations: at least MIN_CALLS host calls to the driver's allocator (ALLOC_CALLS), taking at least
  MIN_SHARE of the span: memory goes to and from the driver every step instead of being cached. Check
  PYTORCH_NO_CUDA_MEMORY_CACHING and PYTORCH_CUDA_ALLOC_CONF in the environment and torch.cuda.empty_cache() in the
  loop.
- slow collectives: collectives of at least MIN_MESSAGE bytes reach at most SLOW_GBPS of bus bandwidth even in their
  fastest call (the fastest call waits least for other ranks), and collectives take at least MIN_SHARE of the span:
  the transport is slow. The bus bandwidth is the message size times bus_factor (all-reduce 2(n-1)/n, all-gather and
  reduce-scatter (n-1)/n, others 1) over the kernel's time. Check
  the NCCL environment variables (NCCL_P2P_DISABLE, NCCL_SHM_DISABLE, NCCL_IB_DISABLE, NCCL_NET, NCCL_PROTO,
  NCCL_ALGO, NCCL_MAX_NCHANNELS); NCCL_DEBUG=INFO prints the transport chosen.
- module hooks: at least MIN_CALLS operations run inside BackwardHookFunctionBackward: full backward hooks are
  registered on modules, and together with host syncs (the number of SYNC_CALLS is given) they serialize the step.
  Check the hooks registered globally (torch.nn.modules.module._global_forward_hooks, _global_backward_hooks) and on
  the modules.

Public components
    MIN_SHARE, MIN_CALLS, MIN_MESSAGE, SLOW_GBPS, ALLOC_CALLS
        The thresholds above and the driver allocator calls counted.
    bus_factor(name, ranks) -> float
        The bus bandwidth factor of a collective kernel from its name and group size.
        Example: `bus_factor("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 4)` is 1.5.
    Sign(name, evidence, check)
        One sign found: its name, the evidence (one sentence with the numbers) and the setting to check.
    find(timeline) -> list[Sign]
        The signs found in a timeline, in the order above.
        Example: `[s.name for s in find(load("trace_rank0.json.gz"))]`.
    report(signs) -> str
        The text report: one line per sign with its evidence and check, or a line saying none was found.
"""
from dataclasses import dataclass

import numpy as np

from profileinsight.timeline import SYNC_CALLS, Timeline, short

MIN_SHARE = 0.02
MIN_CALLS = 50
MIN_MESSAGE = 1 << 20
SLOW_GBPS = 10.0
ALLOC_CALLS = ("cudaMalloc", "cudaFree", "cuMemAlloc_v2", "cuMemFree_v2", "cudaMallocAsync", "cudaFreeAsync")


@dataclass
class Sign:
    name: str
    evidence: str
    check: str


def bus_factor(name: str, ranks: int) -> float:
    n = max(ranks, 1)
    lower = name.lower()
    if "allreduce" in lower:
        return 2.0 * (n - 1) / n
    if "allgather" in lower or "reducescatter" in lower:
        return (n - 1) / n
    return 1.0


def find(tl: Timeline) -> list[Sign]:
    dur = tl.end - tl.start
    span = float(tl.end.max() - tl.start.min())
    kernel_time = float(dur[tl.kind == 0].sum())
    out = []
    math = np.array(["aten::_scaled_dot_product_attention_math" in s for s in tl.site])
    if dur[math].sum() >= MIN_SHARE * kernel_time:
        out.append(Sign("math attention",
                        f"{dur[math].sum() / 1e3:,.1f} ms of kernels ({100 * dur[math].sum() / kernel_time:.0f}% of "
                        f"kernel time, forward only) run inside aten::_scaled_dot_product_attention_math",
                        "torch.backends.cuda.flash_sdp_enabled(), mem_efficient_sdp_enabled(), cudnn_sdp_enabled(), "
                        "and whether dtype, head size and mask suit the fused kernels"))
    nan = np.array([any("aten::isnan" in c for c in s) and any("evaluate_function" in c for c in s)
                    for s in tl.site])
    if nan.sum() >= MIN_CALLS:
        syncs = sum(1 for s in tl.syncs if any("evaluate_function" in c for c in s.site))
        out.append(Sign("anomaly detection",
                        f"{int(nan.sum()):,} isnan kernels inside the backward's evaluate_function, "
                        f"{syncs:,} host synchronizations from the backward",
                        "torch.is_anomaly_enabled() (torch.autograd.set_detect_anomaly)"))
    allocs = [tl.host_calls[n] for n in ALLOC_CALLS if n in tl.host_calls]
    count, time = sum(c for c, _ in allocs), sum(t for _, t in allocs)
    if count >= MIN_CALLS and time >= MIN_SHARE * span:
        out.append(Sign("driver allocations every step",
                        f"{count:,} driver allocation calls ({', '.join(n for n in ALLOC_CALLS if n in tl.host_calls)}"
                        f") taking {time / 1e3:,.1f} ms of host time ({100 * time / span:.0f}% of the span)",
                        "PYTORCH_NO_CUDA_MEMORY_CACHING and PYTORCH_CUDA_ALLOC_CONF in the environment, "
                        "torch.cuda.empty_cache() in the loop"))
    comm = np.nonzero((tl.comm_bytes >= MIN_MESSAGE) & (tl.kind == 0))[0]
    if len(comm) and dur[comm].sum() >= MIN_SHARE * span:
        rates = [bus_factor(tl.names[tl.op[i]], int(tl.comm_ranks[i])) * tl.comm_bytes[i] / dur[i] / 1e3 for i in comm]
        best = int(np.argmax(rates))
        if rates[best] <= SLOW_GBPS:
            k = comm[best]
            out.append(Sign("slow collectives",
                            f"the fastest of {len(comm):,} collectives of at least {MIN_MESSAGE >> 20} MB reaches "
                            f"{rates[best]:,.1f} GB/s of bus bandwidth ({short(tl.names[tl.op[k]])}, "
                            f"{tl.comm_bytes[k] / 1e6:,.1f} MB in {dur[k] / 1e3:,.2f} ms)",
                            "the NCCL environment (NCCL_P2P_DISABLE, NCCL_SHM_DISABLE, NCCL_IB_DISABLE, NCCL_NET, "
                            "NCCL_PROTO, NCCL_ALGO, NCCL_MAX_NCHANNELS); NCCL_DEBUG=INFO prints the transport"))
    hooks = np.array([any("BackwardHookFunctionBackward" in c for c in s) for s in tl.site])
    if hooks.sum() >= MIN_CALLS:
        syncs = sum(1 for s in tl.syncs if s.name in SYNC_CALLS)
        out.append(Sign("module hooks",
                        f"{int(hooks.sum()):,} GPU operations run inside BackwardHookFunctionBackward (full backward "
                        f"hooks on modules); the trace has {syncs:,} host synchronizations",
                        "hooks registered globally (torch.nn.modules.module._global_forward_hooks, "
                        "_global_backward_hooks) and on the modules (_forward_hooks, _backward_hooks)"))
    return out


def report(signs: list[Sign]) -> str:
    if not signs:
        return "Signs of misconfiguration: none found (math attention, anomaly detection, driver allocations every " \
               "step, slow collectives, module hooks)."
    lines = [f"Signs of misconfiguration: {len(signs)}"]
    for s in signs:
        lines.append(f"  {s.name}: {s.evidence}")
        lines.append(f"    check: {s.check}")
    return "\n".join(lines)

