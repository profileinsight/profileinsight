"""Rule 2: find the bubbles of a timeline, the times when the GPU computes nothing, say why, and sum them per cycle.

A bubble is a stretch of at least MIN_BUBBLE_US in which no compute kernel runs on any stream of the GPU. A compute
kernel is a kernel whose name does not match COMMUNICATION; communication kernels, copies and memory sets may run
during a bubble, but they do not advance the model's computation. Every bubble ends when a compute kernel starts. Its
reason is what that kernel actually waited for, by the first of these rules that applies; an operation that merely
runs during the bubble, without the kernel waiting for it, is not a reason:
  1. late launch: the kernel was launched late, after the GPU had gone idle, by at least half the bubble.
     a. If the host had been stopped by a synchronizing call (timeline.SYNC_CALLS) on the launching thread (the GPU
        went idle while the thread was inside it, or it covered more than half of the late time, or it happened
        earlier, fewer than REFILL_OPS launches before this one, and since its end the host never got ahead again: no
        operation launched in between started more than AHEAD_US after its launch call returned), the sync emptied
        the queue of work and the CPU could not run ahead. The sync waited for the stream of the GPU operation that
        ended last before it returned; what it waited for is the work of that stream while the host was blocked: if
        communication kernels ran on it for more than half of the sync, the reason is communication; if copies and
        memory sets did, copy (e.g. a pageable copy, or the allocator waiting for a prefetch copy to finish);
        otherwise sync (the host waited for compute, or for nothing). Work on other streams during the sync does not
        count: the host did not wait for it. For communication and copy the awaited operation is the one of that kind
        on that stream that ended last before the sync returned; the detail names it and the sync.
     b. Otherwise host: the host was too slow to keep the GPU fed; after REFILL_OPS launches without getting ahead a
        sync long before is not the cause either.
  2. communication or copy: launched in time, and a communication kernel (or a copy or memory set) ended within
     WAIT_MATCH_US before the kernel started: the kernel waited for it.
  3. blocking: the kernel's launch call itself lasted more than half the bubble (an allocation, a synchronous copy, a
     full launch queue).
  4. latency: the bubble is shorter than LATENCY_US, the GPU's own delay between dependent kernels.
  5. unknown.
Detail and site name the cause: the awaited operation and its launch site, or the sync and its site, or for host the
CPU calls around the late launch.

Bubbles are then summed per cycle (cycles.py): every bubble belongs to the repetition of the kernel that ended it, the
work it delayed, and within a segment the bubbles with the same reason, detail and site form one group; names are
compared with digits replaced by N and hashes (runs of HASH or more letters and digits with at least one of each, such
as torch.compile graph names) replaced by H. A group seen in every repetition is systematic; one seen in a single
repetition is an outlier. Groups whose total is below MIN_SHOWN of the timeline's span are not listed, only counted,
and so are segments without a listed group, to keep the report short.

For a communication, copy or sync group the report shows one example, the bubble whose length is closest to the
group's median, as a timeline: the compute kernel that ended before it, the awaited operation, the communication and
copy operations that overlap the bubble (the EXAMPLE_OPS that overlap it longest), the host's sync call, the bubble
itself and the compute kernel that ended it, each split into its beginning and its ending, in time order. Every event
takes a lane while it lasts, drawn as "|", so that what happens together is seen side by side. For a sync it says
how long the host was blocked and how much work was queued ahead of the operation it waited for. For an awaited
communication or copy operation it says what is needed to hide it: its size and rate, when it was launched (and from
where) against when it started, what was queued before it on its stream back to the last time that stream was idle
(the QUEUED operations just before it are in the timeline too), and that idle stretch, where it could have run
instead.

The report gives per segment the bubble time, its share of the segment's time (from the end of the segment before it,
or its earliest bubble if that started before, to its own end) and the largest groups, and over the whole timeline
the bubble time per reason.

Public components
    COMMUNICATION, MIN_BUBBLE_US, WAIT_MATCH_US, LATENCY_US, AHEAD_US, REFILL_OPS, SHOWN, MIN_SHOWN, HASH,
    EXAMPLE_OPS, QUEUED
        The name pattern of communication kernels; the thresholds of the rules above; the most groups listed per
        segment and the smallest listed share of the span; the shortest run of letters and digits taken as a hash;
        the most overlapping operations and queued operations shown in an example.
    Bubble(start, end, reason, detail, site, kernel, awaited, sync)
        One bubble: start and end in microseconds (trace time), reason (one of the rules), detail, site (innermost
        first), the index of the compute kernel that ended it, the index of the operation it waited for (-1 if none)
        and the host's sync call (a timeline.Sync, or None). Property dur.
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
        The example of a bubble described above: the lines about the awaited operation, then the timeline, one line
        per beginning or ending, e.g.
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
from profileinsight.timeline import Sync, Timeline, short

COMMUNICATION = re.compile(r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|sendrecv|"
                           r"cross_device_reduce|deep_?ep", re.IGNORECASE)
MIN_BUBBLE_US = 5.0
WAIT_MATCH_US = 20.0
LATENCY_US = 20.0
AHEAD_US = 50.0
REFILL_OPS = 10
SHOWN = 5
MIN_SHOWN = 0.002
HASH = 20
EXAMPLE_OPS = 6
QUEUED = 3


@dataclass
class Bubble:
    start: float
    end: float
    reason: str
    detail: str
    site: tuple[str, ...]
    kernel: int
    awaited: int
    sync: Sync | None

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
    order = np.nonzero(compute)[0][np.argsort(tl.start[compute], kind="stable")]
    reach = np.maximum.accumulate(tl.end[order])
    kinds = {"communication": comm, "copy": tl.kind != 0}
    other = {r: _Ended(tl, m) for r, m in kinds.items()}
    every = _Ended(tl, np.ones(len(tl.op), dtype=bool))
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
        late = max(0.0, min(tl.launch_start[k], g1) - g0)
        if late >= 0.5 * (g1 - g0):
            sync = _stopper(syncs[int(tl.launch_thread[k])], g0, tl.launch_start[k], late, launched, next_ahead)
            if sync is None:
                out.append(Bubble(g0, g1, "host", "launched late", tl.site[k], k, -1, None))
                continue
            w = _waited_for(tl, kinds, every.last(sync.start, sync.end), sync)
            if w < 0:
                out.append(Bubble(g0, g1, "sync", sync.name, sync.site, k, -1, sync))
            else:
                waited = f"; host waited in {sync.name} in {' < '.join(sync.site)}"
                reason = "communication" if comm[w] else "copy"
                out.append(Bubble(g0, g1, reason, tl.names[tl.op[w]] + waited, tl.site[w], k, w, sync))
            continue
        for reason, ended in other.items():
            w = ended.last(g0, g1)
            if w >= 0 and g1 - tl.end[w] <= WAIT_MATCH_US:
                out.append(Bubble(g0, g1, reason, tl.names[tl.op[w]], tl.site[w], k, w, None))
                break
        else:
            if min(tl.launch_end[k], g1) - max(tl.launch_start[k], g0) > 0.5 * (g1 - g0):
                out.append(Bubble(g0, g1, "blocking", tl.launch_call[k], tl.site[k], k, -1, None))
            else:
                out.append(Bubble(g0, g1, "latency" if g1 - g0 < LATENCY_US else "unknown", "", (), k, -1, None))
    return out


def _compute(tl: Timeline) -> tuple[np.ndarray, np.ndarray]:
    """Masks of the compute kernels and of the communication kernels."""
    comm_name = np.array([bool(COMMUNICATION.search(n)) for n in tl.names])
    return (tl.kind == 0) & ~comm_name[tl.op], (tl.kind == 0) & comm_name[tl.op]


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


class _Ended:
    """The operations selected by a mask, by end time: the one that ended last within a stretch of time."""

    def __init__(self, tl: Timeline, mask: np.ndarray):
        idx = np.nonzero(mask)[0]
        self.by_end = idx[np.argsort(tl.end[idx], kind="stable")]
        self.ends = tl.end[self.by_end]

    def last(self, t0: float, t1: float) -> int:
        """The operation that ended last in (t0, t1 + 2 us], or -1."""
        j = int(np.searchsorted(self.ends, t1 + 2.0, side="right")) - 1
        return int(self.by_end[j]) if j >= 0 and self.ends[j] > t0 else -1


def _waited_for(tl: Timeline, kinds: dict[str, np.ndarray], last: int, sync: Sync) -> int:
    """The communication or copy operation a sync waited for (rule 1a), or -1: on the stream of `last`, the
    operation that ended last before the sync returned, the kind of work that covered more than half of the sync."""
    if last < 0:
        return -1
    on = (tl.stream == tl.stream[last]) & (tl.start < sync.end) & (tl.end > sync.start)
    for mask in kinds.values():
        mine = np.nonzero(on & mask)[0]
        covered = (np.minimum(tl.end[mine], sync.end) - np.maximum(tl.start[mine], sync.start)).sum()
        if covered > 0.5 * (sync.end - sync.start):
            return int(mine[np.argmax(tl.end[mine])])
    return -1


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
    g0, g1, w = bubble.start, bubble.end, bubble.awaited
    lines, queued = (_awaited(tl, w) if w >= 0 else ([], []))
    if bubble.reason == "sync":
        s = bubble.sync
        lines = [f"host blocked {(s.end - s.start) / 1e3:,.3f} ms in {s.name}"] + lines[1:2]
    done = np.nonzero(compute & (tl.end <= g0 + 1e-3))[0]
    before = [int(done[np.argmax(tl.end[done])])] if len(done) else []
    over = np.nonzero(~compute & (tl.start < g1) & (tl.end > g0))[0]
    overlap = np.minimum(tl.end[over], g1) - np.maximum(tl.start[over], g0)
    during = over[np.argsort(-overlap, kind="stable")[:EXAMPLE_OPS]].tolist()
    shown = sorted(set(before + queued + during + ([w] if w >= 0 else []) + [bubble.kernel]))
    events = [(g0, 1, -1, "bubble: no compute kernel"), (g1, 0, -1, "bubble: no compute kernel")]
    if bubble.sync is not None:
        s = bubble.sync
        what = f"host blocked in {s.name}" + (f" in {' < '.join(s.site)}" if s.site else "")
        events += [(s.start, 1, -2, what), (s.end, 0, -2, what)]
    for k in shown:
        what = f"{short(tl.names[tl.op[k]])} [stream {tl.stream[k]}]" + (" (awaited)" if k == w else "")
        events += [(float(tl.start[k]), 1, k, what), (float(tl.end[k]), 0, k, what)]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    lane: dict[int, int] = {}
    width = 0
    rows = []
    for t, begins, k, what in events:
        if begins:
            lane[k] = min(set(range(len(lane) + 1)) - set(lane.values()))
            width = max(width, lane[k] + 1)
        rows.append((set(lane.values()), t, "begins" if begins else "ends  ", what))
        if not begins:
            del lane[k]
    return lines + ["".join("|" if i in used else " " for i in range(width)) +
                    f"  {(t - tl.origin) / 1e3:>12,.3f} ms  {word}  {what}" for used, t, word, what in rows]


def _awaited(tl: Timeline, w: int) -> tuple[list[str], list[int]]:
    """Lines on what it takes to hide the awaited operation w, and the operations queued just before it."""
    ms = lambda t: f"{(t - tl.origin) / 1e3:,.3f} ms"
    dur = tl.end[w] - tl.start[w]
    size = f", {tl.bytes[w] / 1e6:,.1f} MB at {tl.bytes[w] / dur / 1e3:,.1f} GB/s" if tl.bytes[w] else ""
    site = " < ".join(tl.site[w]) or "no CPU call"
    delay = (tl.start[w] - tl.launch_start[w]) / 1e3
    lines = [f"awaited: {short(tl.names[tl.op[w]])} [stream {tl.stream[w]}], {dur / 1e3:,.3f} ms{size}; "
             f"launched at {ms(tl.launch_start[w])} from {site}, started {delay:,.3f} ms after its launch"]
    same = np.nonzero(tl.stream == tl.stream[w])[0]
    same = same[np.argsort(tl.start[same], kind="stable")]
    i = int(np.nonzero(same == w)[0][0])
    j = i
    while j > 0 and tl.start[same[j]] - tl.end[same[j - 1]] < MIN_BUBBLE_US:
        j -= 1
    queue = same[j:i]
    if len(queue):
        moved = int(tl.bytes[queue].sum())
        lines.append(f"stream {tl.stream[w]} before it: busy without a gap since {ms(tl.start[queue[0]])}, "
                     f"{len(queue):,} operations ({moved / 1e6:,.1f} MB) ahead of it")
    if j > 0:
        idle = tl.start[same[j]] - tl.end[same[j - 1]]
        lines.append(f"stream {tl.stream[w]} was last idle from {ms(tl.end[same[j - 1]])} to "
                     f"{ms(tl.start[same[j]])} ({idle / 1e3:,.3f} ms): the earliest it could have started there")
    return lines, same[max(j, i - QUEUED):i].tolist()


def report(tl: Timeline, structure: Structure, bubbles: list[Bubble]) -> str:
    total = sum(b.dur for b in bubbles)
    span = structure.span * 1e3
    reasons = defaultdict(float)
    for b in bubbles:
        reasons[b.reason] += b.dur
    lines = [f"Bubbles (no compute kernel running): {total / 1e3:,.3f} ms of {span / 1e3:,.3f} ms "
             f"({100 * total / span:.1f}%), {len(bubbles):,} bubbles",
             "  by reason: " + ", ".join(f"{r} {t / 1e3:,.3f} ms" for r, t in sorted(reasons.items(),
                                                                                    key=lambda x: -x[1])),
             f"  groups below {100 * MIN_SHOWN:g}% of the span ({MIN_SHOWN * span / 1e3:,.3f} ms) are not listed", ""]
    groups = group(tl, structure, bubbles)
    compute = _compute(tl)[0]
    ends = [(tl.start.min() - tl.origin) / 1e3] + [max(s.end) for s in structure.segments]
    hidden = [0, 0.0]
    for n, s in enumerate(structure.segments):
        gs = groups.get(n, [])
        listed = [g for g in gs if g.total >= MIN_SHOWN * span][:SHOWN]
        rest = gs[len(listed):]
        if not listed:
            hidden[0] += len(rest)
            hidden[1] += sum(g.total for g in rest)
            continue
        lost = sum(g.total for g in gs)
        name = s.label if s.count == 1 else f"{s.label}^{s.count}"
        since = min(max(ends[:n + 1]), (min(g.earliest for g in gs) - tl.origin) / 1e3)
        own = max(ends[n + 1] - since, 1e-3) * 1e3
        lines.append(f"{name}: {lost / 1e3:,.3f} ms of bubbles in {own / 1e3:,.3f} ms ({100 * lost / own:.1f}%)")
        for i, g in enumerate(listed):
            each = "" if s.count == 1 else f" ({g.count / len(g.repetitions):,.0f} per {s.label})"
            lines.append(f"  {i + 1}. {g.reason}, {_where(s.label, s.count, g)}: {g.count:,} bubbles{each}, "
                         f"{g.median / 1e3:,.3f} ms median, {g.total / 1e3:,.3f} ms total, "
                         f"ending typically {g.offset / 1e3:,.3f} ms into the repetition")
            name, _, waited = g.detail.partition("; ")
            cause = short(name) + (" in " + " < ".join(g.site) if g.site else "") + (f"; {waited}" if waited else "")
            if cause:
                lines.append(f"     {cause[:200]}")
            if g.reason in ("communication", "copy", "sync"):
                lines.append("     example:")
                lines.extend("       " + row for row in _view(tl, g.example, compute))
        if rest:
            lines.append(f"  and {len(rest):,} smaller groups, {sum(g.total for g in rest) / 1e3:,.3f} ms")
        lines.append("")
    if hidden[0]:
        lines.append(f"{hidden[0]:,} groups in other segments, {hidden[1] / 1e3:,.3f} ms in all, are below the "
                     f"listing threshold")
    return "\n".join(lines).rstrip()





def _where(label: str, count: int, g: Group) -> str:
    if count == 1:
        return f"in {label}"
    if len(g.repetitions) == 1:
        return f"once, in {label}{g.repetitions[0] + 1}"
    return f"in {len(g.repetitions)} of {count} {label}"
