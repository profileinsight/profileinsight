"""Rule 2: find the bubbles of a timeline, the times when the GPU computes nothing, say why, and sum them per cycle.

A bubble is a stretch of at least MIN_BUBBLE_US in which no compute kernel runs on any stream of the GPU. A compute
kernel is a kernel whose name does not match COMMUNICATION; communication kernels, copies and memory sets may run
during a bubble, but they do not advance the model's computation. Every bubble ends when a compute kernel starts; its
reason is found from that kernel's launch and from what else ran, by the first of these rules that applies:
  1. communication: the bubble lasts at least LATENCY_US and communication kernels ran for more than half of it: the
     GPU was busy communicating (if the host was waiting in a sync meanwhile, it was waiting for this communication).
     The detail names the communication kernel that overlaps the bubble longest, and the host's sync if rule 3's
     test for one also holds (e.g. the allocator waiting for a copy before it can free memory).
  2. copy: the same with copies and memory sets.
  3. sync: the kernel was launched late, after the GPU had gone idle, by at least half the bubble, and the host had
     been stopped by a synchronizing call (timeline.SYNC_CALLS) on the launching thread: either the GPU went idle while
     the thread was inside it, or it covered more than half of the late time, or it happened earlier, fewer than
     REFILL_OPS launches before this one, and since its end the host never got ahead again (no operation launched in
     between started more than AHEAD_US after its launch call returned). The sync emptied the queue of work and the
     CPU could not run ahead. After REFILL_OPS launches without getting ahead the host itself is the bottleneck.
  4. host: launched late as in 3 without such a sync: the host was too slow to keep the GPU fed.
  5. communication or copy: launched in time, and a communication kernel (or a copy or memory set) ended within
     WAIT_MATCH_US before the kernel started: the kernel waited for it.
  6. blocking: the kernel's launch call itself lasted more than half the bubble (an allocation, a synchronous copy, a
     full launch queue).
  7. latency: the bubble is shorter than LATENCY_US, the GPU's own delay between dependent kernels.
  8. unknown.
Detail and site name the cause: for sync the synchronizing call and the innermost CPU calls around it, for host the
CPU calls around the late launch, for communication and copy the operation waited for.

Bubbles are then summed per cycle (cycles.py): every bubble belongs to the repetition of the kernel that ended it, the
work it delayed, and within a
segment the bubbles with the same reason, detail and site form one group; names are compared with digits replaced by
N and hashes (runs of HASH or more letters and digits with at least one of each, such as torch.compile graph names)
replaced by H. A group seen in every
repetition is systematic; one seen in a single repetition is an outlier. For a group of communication or copy
bubbles the report shows one example, the bubble whose length is closest to the group's median, as a timeline: the
compute kernel that ended before it, the communication and copy operations that overlap it (the EXAMPLE_OPS that overlap
it longest), the bubble itself and the compute kernel that ended it, each split into its beginning and its ending, in
time order. Every event takes a lane while it lasts, drawn as "|", so that what happens together is seen side by side.
The report gives per segment the bubble time,
its share of the segment's time (from the end of the segment before it, or its earliest bubble if that started
before, to its own end) and the largest groups, and over the whole timeline the bubble time per reason.

Public components
    COMMUNICATION, MIN_BUBBLE_US, WAIT_MATCH_US, LATENCY_US, AHEAD_US, REFILL_OPS, SHOWN, HASH, EXAMPLE_OPS
        The name pattern of communication kernels; the thresholds of the rules above; the number of groups shown per
        segment; the shortest run of letters and digits taken as a hash; the most overlapping operations shown in an
        example.
    Bubble(start, end, reason, detail, site, kernel)
        One bubble: start and end in microseconds (trace time), reason (one of the rules), detail, site (innermost
        first) and the index of the compute kernel that ended it. Property dur.
        Example: `b.dur / 1e3` is its length in ms.
    find(timeline) -> list[Bubble]
        The bubbles of a Timeline in time order.
        Example: `sum(b.dur for b in find(tl) if b.reason == "sync")`.
    Group(segment, reason, detail, site, repetitions, count, total, median, offset, earliest, example)
        Bubbles of one segment with the same reason, detail and site: the segment label, the indices of the
        repetitions with at least one of them, their number, total and median duration (us), the median time from
        the start of their repetition to their end (us), the earliest start (us, trace time) and the bubble whose
        length is closest to the median.
    group(timeline, structure, bubbles) -> dict[int, list[Group]]
        Per segment (its index in structure.segments), its groups, largest total first.
    view(timeline, bubble) -> list[str]
        The example timeline of a bubble described above, one line per beginning or ending, e.g.
            "||     1,234.567 ms  begins  ncclDevKernel_AllReduce_Sum_f32_RING_LL [stream 23]".
    report(timeline, structure, bubbles) -> str
        The text report described above.
        Example: `print(report(tl, detect(tl), find(tl)))`.
"""
import bisect
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from profileinsight.cycles import Structure
from profileinsight.timeline import Timeline

COMMUNICATION = re.compile(r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|sendrecv|"
                           r"cross_device_reduce|deep_?ep", re.IGNORECASE)
MIN_BUBBLE_US = 5.0
WAIT_MATCH_US = 20.0
LATENCY_US = 20.0
AHEAD_US = 50.0
REFILL_OPS = 10
SHOWN = 5
HASH = 20
EXAMPLE_OPS = 6


@dataclass
class Bubble:
    start: float
    end: float
    reason: str
    detail: str
    site: tuple[str, ...]
    kernel: int

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Group:
    segment: str
    reason: str
    detail: str
    site: tuple[str, ...]
    repetitions: list[int]
    count: int
    total: float
    median: float
    offset: float
    earliest: float
    example: Bubble


def find(tl: Timeline) -> list[Bubble]:
    compute, comm = _compute(tl)
    copy = tl.kind != 0
    order = np.nonzero(compute)[0][np.argsort(tl.start[compute], kind="stable")]
    reach = np.maximum.accumulate(tl.end[order])
    other = {"communication": _Busy(tl, comm), "copy": _Busy(tl, copy)}
    launch_order = np.argsort(tl.launch_start, kind="stable")
    launched = tl.launch_start[launch_order]
    ahead = np.append(np.nonzero((tl.start - tl.launch_end)[launch_order] > AHEAD_US)[0], len(launched))
    next_ahead = ahead[np.searchsorted(ahead, np.arange(len(launched) + 1))]
    syncs = defaultdict(list)
    for s in tl.syncs:
        syncs[s.thread].append(s)
    out = []
    for i in np.nonzero(tl.start[order[1:]] - reach[:-1] >= MIN_BUBBLE_US)[0]:
        k = int(order[i + 1])
        g0, g1 = float(reach[i]), float(tl.start[k])
        out.append(Bubble(g0, g1, *_reason(tl, k, g0, g1, other, syncs[int(tl.launch_thread[k])], launched,
                                           next_ahead), k))
    return out


def _compute(tl: Timeline) -> tuple[np.ndarray, np.ndarray]:
    """Masks of the compute kernels and of the communication kernels."""
    comm_name = np.array([bool(COMMUNICATION.search(n)) for n in tl.names])
    return (tl.kind == 0) & ~comm_name[tl.op], (tl.kind == 0) & comm_name[tl.op]


def _reason(tl: Timeline, k: int, g0: float, g1: float, other: dict, syncs: list, launched: np.ndarray,
            next_ahead: np.ndarray) -> tuple[str, str, tuple[str, ...]]:
    late = max(0.0, min(tl.launch_start[k], g1) - g0)
    sync = _stopper(syncs, g0, tl.launch_start[k], late, launched, next_ahead) if late >= 0.5 * (g1 - g0) else None
    if g1 - g0 >= LATENCY_US:
        for reason, busy in other.items():
            if busy.covered(g0, g1) > 0.5 * (g1 - g0):
                over = busy.ops[(tl.start[busy.ops] < g1) & (tl.end[busy.ops] > g0)]
                main = int(over[np.argmax(np.minimum(tl.end[over], g1) - np.maximum(tl.start[over], g0))])
                waited = "" if sync is None else f"; host waited in {sync.name} in {' < '.join(sync.site)}"
                return reason, tl.names[tl.op[main]] + waited, tl.site[main]
    if late >= 0.5 * (g1 - g0):
        if sync is not None:
            return "sync", sync.name, sync.site
        return "host", "launched late", tl.site[k]
    for reason, busy in other.items():
        waited = busy.ended(g0, g1)
        if waited is not None:
            return reason, tl.names[tl.op[waited]], tl.site[waited]
    if min(tl.launch_end[k], g1) - max(tl.launch_start[k], g0) > 0.5 * (g1 - g0):
        return "blocking", tl.launch_call[k], tl.site[k]
    if g1 - g0 < LATENCY_US:
        return "latency", "", ()
    return "unknown", "", ()


def _stopper(syncs: list, g0: float, launch: float, late: float, launched: np.ndarray, next_ahead: np.ndarray):
    """The synchronizing call that stopped the host before a late launch at `launch`, or None. next_ahead[p] is the
    first position at or after p, in launch order, of an operation that started more than AHEAD_US after its launch
    call returned."""
    starts = [s.start for s in syncs]
    j = bisect.bisect_right(starts, launch) - 1
    best, covered = None, 0.0
    while j >= 0:
        s = syncs[j]
        if s.start <= g0 < s.end:
            return s
        overlap = min(s.end, launch) - max(s.start, g0)
        if overlap > 0:
            covered += overlap
            best = best or s
        elif s.end <= g0:
            if covered > 0.5 * late:
                return best
            a, b = np.searchsorted(launched, [s.end, launch])
            return s if b - a < REFILL_OPS and next_ahead[a] >= b else None
        j -= 1
    return best if covered > 0.5 * late else None


class _Busy:
    """The operations selected by a mask: which one ended just before a time, and how much of an interval they cover."""

    def __init__(self, tl: Timeline, mask: np.ndarray):
        idx = np.nonzero(mask)[0]
        self.ops = idx
        self.by_end = idx[np.argsort(tl.end[idx], kind="stable")]
        self.ends = tl.end[self.by_end]
        spans = sorted(zip(tl.start[idx].tolist(), tl.end[idx].tolist()))
        merged: list[list[float]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.lo = np.array([a for a, _ in merged])
        self.hi = np.array([b for _, b in merged])
        self.before = np.concatenate([[0.0], np.cumsum(self.hi - self.lo)])

    def ended(self, g0: float, g1: float) -> int | None:
        j = int(np.searchsorted(self.ends, g1 + 2.0, side="right")) - 1
        if j >= 0 and g0 < self.ends[j] and g1 - self.ends[j] <= WAIT_MATCH_US:
            return int(self.by_end[j])
        return None

    def covered(self, g0: float, g1: float) -> float:
        return self._upto(g1) - self._upto(g0)

    def _upto(self, t: float) -> float:
        j = int(np.searchsorted(self.lo, t, side="right"))
        if j == 0:
            return 0.0
        return float(self.before[j - 1] + min(self.hi[j - 1], t) - self.lo[j - 1])


def group(tl: Timeline, structure: Structure, bubbles: list[Bubble]) -> dict[int, list[Group]]:
    firsts, owners = [], []
    for n, s in enumerate(structure.segments):
        for r, f in enumerate(s.first):
            firsts.append(f)
            owners.append((n, r, s.begin[r] * 1e3 + tl.origin))
    found = defaultdict(list)
    for b in bubbles:
        segment, rep, begin = owners[bisect.bisect_right(firsts, b.kernel) - 1]
        key = (segment, b.reason, _plain(b.detail), tuple(_plain(x) for x in b.site))
        found[key].append((rep, b, b.end - begin))
    groups = defaultdict(list)
    for (segment, reason, detail, site), members in found.items():
        durs = [b.dur for _, b, _ in members]
        median = float(np.median(durs))
        example = min(members, key=lambda m: abs(m[1].dur - median))[1]
        groups[segment].append(Group(structure.segments[segment].label, reason, detail, site,
                                     sorted({r for r, _, _ in members}), len(members), float(sum(durs)), median,
                                     float(np.median([o for _, _, o in members])),
                                     min(b.start for _, b, _ in members), example))
    return {segment: sorted(gs, key=lambda g: -g.total) for segment, gs in groups.items()}


def _plain(name: str) -> str:
    """A name with hashes replaced by H and digits by N."""
    return re.sub(r"[0-9]+", "N", re.sub(rf"\b(?=[A-Za-z]*[0-9])(?=[0-9]*[A-Za-z])[A-Za-z0-9]{{{HASH},}}\b", "H",
                                         name))


def view(tl: Timeline, bubble: Bubble) -> list[str]:
    return _view(tl, bubble, _compute(tl)[0])


def _view(tl: Timeline, bubble: Bubble, compute: np.ndarray) -> list[str]:
    g0, g1 = bubble.start, bubble.end
    done = np.nonzero(compute & (tl.end <= g0 + 1e-3))[0]
    before = [int(done[np.argmax(tl.end[done])])] if len(done) else []
    over = np.nonzero(~compute & (tl.start < g1) & (tl.end > g0))[0]
    overlap = np.minimum(tl.end[over], g1) - np.maximum(tl.start[over], g0)
    during = sorted(over[np.argsort(-overlap, kind="stable")[:EXAMPLE_OPS]].tolist())
    events = [(g0, 1, -1), (g1, 0, -1)]
    for k in before + during + [bubble.kernel]:
        events += [(float(tl.start[k]), 1, k), (float(tl.end[k]), 0, k)]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    lane: dict[int, int] = {}
    width = 0
    rows = []
    for t, begins, k in events:
        if begins:
            lane[k] = min(set(range(len(lane) + 1)) - set(lane.values()))
            width = max(width, lane[k] + 1)
        what = "bubble: no compute kernel" if k < 0 else f"{_short(tl.names[tl.op[k]])} [stream {tl.stream[k]}]"
        rows.append((set(lane.values()), t, "begins" if begins else "ends  ", what))
        if not begins:
            del lane[k]
    return ["".join("|" if i in used else " " for i in range(width)) + f"  {(t - tl.origin) / 1e3:>12,.3f} ms  "
            f"{word}  {what}" for used, t, word, what in rows]


def report(tl: Timeline, structure: Structure, bubbles: list[Bubble]) -> str:
    total = sum(b.dur for b in bubbles)
    span = structure.span * 1e3
    reasons = defaultdict(float)
    for b in bubbles:
        reasons[b.reason] += b.dur
    lines = [f"Bubbles (no compute kernel running): {total / 1e3:,.3f} ms of {span / 1e3:,.3f} ms "
             f"({100 * total / span:.1f}%), {len(bubbles):,} bubbles",
             "  by reason: " + ", ".join(f"{r} {t / 1e3:,.3f} ms" for r, t in sorted(reasons.items(),
                                                                                    key=lambda x: -x[1])), ""]
    groups = group(tl, structure, bubbles)
    compute = _compute(tl)[0]
    ends = [(tl.start.min() - tl.origin) / 1e3] + [max(s.end) for s in structure.segments]
    for n, s in enumerate(structure.segments):
        gs = groups.get(n, [])
        lost = sum(g.total for g in gs)
        if not gs:
            continue
        name = s.label if s.count == 1 else f"{s.label}^{s.count}"
        since = min(max(ends[:n + 1]), (min(g.earliest for g in gs) - tl.origin) / 1e3)
        own = max(ends[n + 1] - since, 1e-3) * 1e3
        lines.append(f"{name}: {lost / 1e3:,.3f} ms of bubbles in {own / 1e3:,.3f} ms ({100 * lost / own:.1f}%)")
        for i, g in enumerate(gs[:SHOWN]):
            each = "" if s.count == 1 else f" ({g.count / len(g.repetitions):,.0f} per {s.label})"
            lines.append(f"  {i + 1}. {g.reason}, {_where(s.label, s.count, g)}: {g.count:,} bubbles{each}, "
                         f"{g.median / 1e3:,.3f} ms median, {g.total / 1e3:,.3f} ms total, "
                         f"ending typically {g.offset / 1e3:,.3f} ms into the repetition")
            name, _, waited = g.detail.partition("; ")
            cause = _short(name) + (" in " + " < ".join(g.site) if g.site else "") + (f"; {waited}" if waited else "")
            if cause:
                lines.append(f"     {cause[:200]}")
            if g.reason in ("communication", "copy"):
                lines.append("     example:")
                lines.extend("       " + row for row in _view(tl, g.example, compute))
        rest = gs[SHOWN:]
        if rest:
            lines.append(f"  and {len(rest):,} more groups, {sum(g.total for g in rest) / 1e3:,.3f} ms")
        lines.append("")
    return "\n".join(lines).rstrip()


def _short(name: str) -> str:
    """A kernel name without "void", its parameter list and, when long, its template arguments."""
    name = re.sub(r"^void ", "", name.replace("(anonymous namespace)", "anonymous").split("(")[0].strip())
    shorter = re.sub(r"<[^<>]*>", "", name)
    while len(name) > 80 and shorter != name:
        name, shorter = shorter, re.sub(r"<[^<>]*>", "", shorter)
    return name


def _where(label: str, count: int, g: Group) -> str:
    if count == 1:
        return f"in {label}"
    if len(g.repetitions) == 1:
        return f"once, in {label}{g.repetitions[0] + 1}"
    return f"in {len(g.repetitions)} of {count} {label}"
