"""Rule 4: across the ranks of a distributed job, which rank the others wait for at the collectives, and why.

A collective (an all-reduce, all-gather, ...) finishes on every rank at about the same time, when the last rank has
joined it. Its kernel starts on each rank when that rank reaches it. So the time a collective kernel takes on one
rank is mostly the time that rank waited for the latest one, and a single rank's profile cannot tell slow
communication from a late peer. Given the profiles of all ranks, this rule:
- finds the collectives: kernels whose name contains one of COLLECTIVE_NAMES, in launch order. The ranks' lists are
  matched from their ends (all ranks record the same steps; at the start of a window a rank may leave out operations
  launched before it, timeline.py), and only as far back as their families agree;
- puts the ranks on one clock: the offset of rank r is the median over the collectives of (end on r - end on the
  first rank), so the ranks may be on different machines;
- per collective, takes the arrival of each rank (its kernel's start on the common clock), the last rank, and the time
  the others waited for it alone (last arrival - second last);
- per rank: how often it arrived last, the time the others waited for it alone in total, and the mean time it waited
  itself;
- for the rank the others waited for most (if they waited at least LATE_SHARE of the span for it): what it did before
  the collectives it was late for. For each such collective, the GPU operations launched between the previous
  collective and it (program order, so that every rank's window holds the same part of the program) are summed by
  family and call site (digits replaced by "#") on every rank; the late rank's sums minus the median of
  the other ranks' are its extra work. The extra work, and the time its GPU idled between the two collectives
  beyond the other ranks', tell GPU work (extra or slower kernels) from a host that was late to launch.

Public components
    COLLECTIVE_NAMES, LATE_SHARE, TOP, MINOR
        The name parts of collective kernels (matched in lower case), the share of the span from which a rank is
        reported as late, the most extra-work lines listed, and the share of the extra work below which a line is
        left out.
    Ranks(names, collectives, offset, arrival, end, last, held, span)
        names        list[str], the ranks' labels (file names).
        collectives  list[str], the matched collectives' families, in order.
        offset       float64 array (ranks), per rank its clock offset to the first rank, in microseconds.
        arrival, end float64 arrays (ranks x collectives), per rank and collective the start and end of its kernel on
                     the common clock.
        last         int64 array (collectives), per collective the rank that arrived last.
        held         float64 array (collectives), per collective the time the other ranks waited for the last one alone.
        span         float, the common span of the matched collectives (first arrival to last end), microseconds.
    collectives(timeline) -> int64 array
        The indices of the timeline's collective kernels, in launch order.
    compare(timelines, names) -> Ranks | None
        The collectives of the ranks matched and put on one clock; None if fewer than two ranks have a collective in
        common.
        Example: `r = compare([load(p) for p in paths], paths); r.names[np.bincount(r.last).argmax()]`.
    report(timelines, ranks) -> str
        The text report: one line per rank, then, for the rank the others waited for most, its extra work.
        Example: `print(report(tls, compare(tls, paths)))`.
    held_share(ranks) -> float
        The largest time the other ranks wait for one rank alone, as a share of the span.
    headline(timelines, ranks) -> str
        One sentence for the report's verdict: the rank the others wait for, how long, and whether its host launches
        late or its GPU runs extra work (with the largest extra kind); empty if no rank holds the others up.
"""
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from profileinsight.timeline import Timeline, short

COLLECTIVE_NAMES = ("nccl", "rccl")
LATE_SHARE = 0.02
TOP = 8
MINOR = 0.01


@dataclass
class Ranks:
    names: list[str]
    collectives: list[str]
    offset: np.ndarray
    arrival: np.ndarray
    end: np.ndarray
    last: np.ndarray
    held: np.ndarray
    span: float


def compare(tls: list[Timeline], names: list[str]) -> Ranks | None:
    rows = [collectives(tl) for tl in tls]
    fams = [[tl.families[tl.family[k]] for k in r] for tl, r in zip(tls, rows)]
    n = min(len(r) for r in rows)
    common = 0
    while common < n and all(f[-1 - common] == fams[0][-1 - common] for f in fams):
        common += 1
    if len(tls) < 2 or common == 0:
        return None
    rows = [r[len(r) - common:] for r in rows]
    start = np.array([tl.start[r] for tl, r in zip(tls, rows)])
    end = np.array([tl.end[r] for tl, r in zip(tls, rows)])
    offset = np.median(end - end[0], axis=1)
    arrival, end = start - offset[:, None], end - offset[:, None]
    order = np.sort(arrival, axis=0)
    return Ranks(names, fams[0][len(fams[0]) - common:], offset, arrival, end, np.argmax(arrival, axis=0),
                 order[-1] - order[-2], float(end.max() - arrival.min()))


def report(tls: list[Timeline], r: Ranks) -> str:
    n = len(r.collectives)
    kinds = ", ".join(f"{c} x{k}" for c, k in _counts([short(c) for c in r.collectives]))
    lines = [f"Ranks: {len(r.names)} profiles, {n} collectives in common ({kinds}), over {r.span / 1e3:,.3f} ms",
             "  per rank: collectives it arrived last at; time the others waited for it alone; mean time it waited "
             "itself (clock offset to the first rank)"]
    waited = (r.arrival.max(axis=0) - r.arrival)
    for k, name in enumerate(r.names):
        mine = r.last == k
        lines.append(f"  {name}: last at {int(mine.sum())} of {n}, held the others {r.held[mine].sum() / 1e3:,.3f} ms, "
                     f"waited {waited[k].mean() / 1e3:,.3f} ms per collective (offset {r.offset[k] / 1e3:+,.3f} ms)")
    held = np.array([r.held[r.last == k].sum() for k in range(len(r.names))])
    late = int(held.argmax())
    if held[late] < LATE_SHARE * r.span:
        lines.append(f"  No rank holds the others up: the most, {r.names[late]}, for {held[late] / 1e3:,.3f} ms "
                     f"({100 * held[late] / r.span:.1f}% of the span). The collectives' time is their own.")
        return "\n".join(lines)
    lines.append(f"  {r.names[late]} holds the others up: they wait {held[late] / 1e3:,.3f} ms for it alone "
                 f"({100 * held[late] / r.span:.1f}% of the span). Collective kernels on the other ranks are slow "
                 f"because they wait for it, not because communication is slow.")
    lines.extend(_extra(tls, r, late))
    return "\n".join(lines)


def held_share(r: Ranks) -> float:
    return max(r.held[r.last == k].sum() for k in range(len(r.names))) / r.span


def headline(tls: list[Timeline], r: Ranks) -> str:
    held = np.array([r.held[r.last == k].sum() for k in range(len(r.names))])
    late = int(held.argmax())
    if held[late] < LATE_SHARE * r.span:
        return ""
    late_at, extra, total, more_idle = _extra_work(tls, r, late)
    why = (f"its host launches late (its GPU idles {more_idle / 1e3:,.1f} ms more than the others' before the "
           f"collectives it is late for)" if more_idle > total else
           f"it runs {total / 1e3:,.1f} ms more GPU work before them" +
           (f", mostly {short(extra[0][2][0])} in {extra[0][2][1]}" if extra else ""))
    return (f"{r.names[late]} holds the other ranks up at the collectives: they wait {held[late] / 1e3:,.1f} ms for "
            f"it alone ({100 * held[late] / r.span:.0f}% of the span); {why}. The collectives are slow because they "
            f"wait for it")


def _extra_work(tls: list[Timeline], r: Ranks, late: int) -> tuple[list[int], list, float, float]:
    """The collectives the late rank was late for (after the first), its extra work before them beyond the median of
    the other ranks (time, calls, (family, site)), largest first, its total, and how much more its GPU idled."""
    rows = [collectives(tl)[-len(r.collectives):] for tl in tls]
    sums = [defaultdict(lambda: np.zeros(2)) for _ in tls]
    idle = np.zeros(len(tls))
    late_at = [c for c in np.nonzero(r.last == late)[0] if c > 0]
    for c in late_at:
        for k, tl in enumerate(tls):
            for i in range(rows[k][c - 1] + 1, rows[k][c]):
                site = re.sub(r"#?[0-9]+", "#", " < ".join(tl.site[i])) or "no CPU call"
                sums[k][(tl.families[tl.family[i]], site)] += (tl.end[i] - tl.start[i], 1)
            idle[k] += _uncovered(tl, tl.end[rows[k][c - 1]], tl.start[rows[k][c]])
    others = [k for k in range(len(tls)) if k != late]
    extra = []
    for key, (t, calls) in sums[late].items():
        rest = np.median([sums[k][key] if key in sums[k] else np.zeros(2) for k in others], axis=0)
        if t - rest[0] > 0:
            extra.append((t - rest[0], calls - rest[1], key))
    extra.sort(reverse=True)
    return late_at, extra, sum(e[0] for e in extra), float(idle[late] - np.median(idle[others]))


def _extra(tls: list[Timeline], r: Ranks, late: int) -> list[str]:
    """What the late rank did before the collectives it was late for, beyond the median of the other ranks."""
    late_at, extra, total, more_idle = _extra_work(tls, r, late)
    out = [f"  Between the collectives it was late for and the ones before (in program order; {len(late_at)} after the "
           f"first in common), {r.names[late]} ran {total / 1e3:,.3f} ms more GPU work than the median other rank, "
           f"and its GPU idled {more_idle / 1e3:+,.3f} ms more" +
           (" (so its host launched late: read its own report's bubbles)" if more_idle > total else "") + ".",
           "  its extra work (time and calls more than the median other rank, by kind and call site):"]
    for t, calls, (family, site) in [e for e in extra if e[0] >= MINOR * total][:TOP]:
        out.append(f"    {t / 1e3:,.3f} ms, {calls:+,.0f} calls: {short(family)} in {site}")
    return out


def _uncovered(tl: Timeline, lo: float, hi: float) -> float:
    """The time in [lo, hi] when no GPU operation of the timeline runs."""
    inside = (tl.end > lo) & (tl.start < hi)
    covered, reach = 0.0, lo
    for a, b in sorted(zip(np.maximum(tl.start[inside], lo), np.minimum(tl.end[inside], hi))):
        if b > reach:
            covered += b - max(a, reach)
            reach = b
    return max(hi - lo, 0.0) - covered


def collectives(tl: Timeline) -> np.ndarray:
    is_collective = np.array([any(c in n.lower() for c in COLLECTIVE_NAMES) for n in tl.names])
    return np.nonzero(is_collective[tl.op] & (tl.kind == 0))[0]


def _counts(items: list[str]) -> list[tuple[str, int]]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return list(out.items())
