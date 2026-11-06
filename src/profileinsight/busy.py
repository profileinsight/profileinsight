"""Rule 3: where the GPU's busy time goes, by kind of work, with what each kind says about its own efficiency.

When the GPU is busy most of the time (few bubbles, rule 2), the step can only get faster by making the work itself
cheaper. This rule sums the time of every GPU operation by family (timeline.py: the name with digits replaced by "#",
so recompiled or retuned variants of one kernel count together) and lists the families that take the most time. Times
are sums of the operations' own durations: operations that overlap on concurrent streams both count, so the shares
are of the summed time, not of the span.

For each family the report gives its calls, total and mean time, the call site most of its time comes from (the
innermost CPU calls around the launches), and what the trace knows about how well it used the GPU:
- kernels: the median blocks per SM of their grids and registers per thread, as the profiler recorded them (its
  estimated occupancy is kept but not shown: it reads 0-20% for well-performing tensor-core kernels); and, when the
  profile was recorded with shapes (record_shapes=True) and the innermost CPU call is a matrix multiplication
  (FLOP_CALLS), the achieved TFLOPS;
- copies and memory sets: the bytes moved and the rate in GB/s, and whether a copy went through pageable host memory.
From these it marks a family with the flags below, each a common reason for slow work that the profile itself shows:
- "small": mean duration under SMALL_US and at least SMALL_CALLS calls: the time goes to many short launches (launch
  latency and tails); fuse them, batch them, or capture them in a CUDA graph;
- "underfilled": median blocks per SM under UNDERFILLED (communication kernels excepted: they use few blocks by
  design): the grid leaves SMs without a block, so part of the GPU idles while the kernel runs; give each launch more
  work (larger batch, split-K, smaller tiles);
- "pageable": copies to or from pageable host memory, which are synchronous and slow; use pinned memory.

Public components
    TOP, SMALL_US, SMALL_CALLS, UNDERFILLED, FLOP_CALLS
        The number of families listed, the thresholds of the flags, and the CPU calls whose shapes give FLOPs.
    Family(name, kind, calls, total, mean, site, blocks_per_sm, occupancy, registers, bytes, flops, flags)
        One family: name (the family's text), kind ("kernel", "copy" or "set"), calls, total and mean time in
        microseconds, site (the call site of most of its time), median blocks per SM, occupancy and registers (NaN or
        -1 when unknown), bytes moved, FLOPs (0 when unknown), flags (list of the flag names above).
    find(timeline) -> list[Family]
        All families, the most time first.
        Example: `fams = find(load("trace.json.gz")); fams[0].name, fams[0].total`.
    report(timeline, families) -> str
        The text report: totals by kind, then the TOP families, one line each and a line of flags where any.
        Example: `print(report(tl, find(tl)))`.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from profileinsight.timeline import Timeline, short

TOP = 15
SMALL_US = 10.0
SMALL_CALLS = 100
UNDERFILLED = 0.5
FLOP_CALLS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::_scaled_mm", "aten::matmul")
KINDS = ("kernel", "copy", "set")


@dataclass
class Family:
    name: str
    kind: str
    calls: int
    total: float
    mean: float
    site: str
    blocks_per_sm: float
    occupancy: float
    registers: int
    bytes: int
    flops: float
    flags: list[str] = field(default_factory=list)


def find(tl: Timeline) -> list[Family]:
    dur = tl.end - tl.start
    out = []
    for f in np.unique(tl.family):
        rows = np.nonzero(tl.family == f)[0]
        kind = KINDS[int(tl.kind[rows[0]])]
        sites = Counter()
        for r in rows:
            sites[" < ".join(tl.site[r]) or "no CPU call"] += dur[r]
        total = float(dur[rows].sum())
        flops = sum(_flops(tl.site[r], tl.dims[r]) for r in rows)
        fam = Family(tl.families[f], kind, len(rows), total, total / len(rows), sites.most_common(1)[0][0],
                     _median(tl.blocks_per_sm[rows]), _median(tl.occupancy[rows]),
                     int(np.median(tl.registers[rows])), int(tl.bytes[rows].sum()), flops)
        fam.flags = _flags(fam)
        out.append(fam)
    return sorted(out, key=lambda x: -x.total)


def report(tl: Timeline, families: list[Family]) -> str:
    span = tl.end.max() - tl.start.min()
    busy = {k: sum(f.total for f in families if f.kind == k) for k in KINDS}
    lines = ["Busy time (operation time summed over streams): " +
             ", ".join(f"{label} {busy[k] / 1e3:,.3f} ms" for k, label in zip(KINDS, ("kernels", "copies",
                                                                                     "memory sets")))
             + f", in a span of {span / 1e3:,.3f} ms",
             f"  the {min(TOP, len(families))} families with the most time (of {len(families)}):"]
    all_time = sum(busy.values())
    for i, f in enumerate(families[:TOP]):
        lines.append(f"  {i + 1}. {f.total / 1e3:,.3f} ms ({100 * f.total / all_time:.1f}%), {f.calls:,} calls, mean "
                     f"{_us(f.mean)}: {short(f.name)} [{f.kind}] in {f.site}")
        lines.append("     " + _facts(f))
        if f.flags:
            lines.append("     flags: " + ", ".join(f.flags))
    return "\n".join(lines)


def _facts(f: Family) -> str:
    if f.kind != "kernel":
        rate = f.bytes / f.total / 1e3 if f.total else 0.0
        return f"{f.bytes / 1e6:,.1f} MB at {rate:,.1f} GB/s"
    parts = []
    if not np.isnan(f.blocks_per_sm):
        parts.append(f"{f.blocks_per_sm:,.2f} blocks per SM")
    if f.registers >= 0:
        parts.append(f"{f.registers} registers per thread")
    if f.flops:
        parts.append(f"{f.flops / f.total / 1e6:,.1f} TFLOPS")
    return ", ".join(parts) or "no launch facts in the trace"


def _flags(f: Family) -> list[str]:
    flags = []
    if f.mean < SMALL_US and f.calls >= SMALL_CALLS:
        flags.append(f"small (mean under {SMALL_US:g} us over {f.calls:,} calls: fuse, batch or graph them)")
    if f.kind == "kernel" and f.blocks_per_sm < UNDERFILLED and "nccl" not in f.name.lower():
        flags.append(f"underfilled ({f.blocks_per_sm:.2f} blocks per SM: part of the GPU idles while it runs)")
    if f.kind == "copy" and "Pageable" in f.name:
        flags.append("pageable (synchronous copy through pageable host memory: use pinned memory)")
    return flags


def _flops(site: tuple[str, ...], dims) -> float:
    if not dims or not site or site[0] not in FLOP_CALLS:
        return 0.0
    shapes = [d for d in dims if isinstance(d, list) and len(d) >= 2]
    if site[0] == "aten::addmm" or site[0] == "aten::baddbmm":
        shapes = shapes[1:]
    if len(shapes) < 2:
        return 0.0
    a, b = shapes[0], shapes[1]
    batch = int(np.prod(a[:-2])) if len(a) > 2 else 1
    return 2.0 * batch * a[-2] * a[-1] * b[-1]


def _median(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _us(t: float) -> str:
    return f"{t / 1e3:,.3f} ms" if t >= 1e3 else f"{t:,.1f} us"
