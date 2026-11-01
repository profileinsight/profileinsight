"""Rule 1: find the cycles of a timeline, the way a person sees them, as a pattern such as A B^20 C^10 D.

A person reading a profile first sees its repetitions (training steps, decode steps, layers) and ignores the small
differences between them. This module does the same on the GPU timeline read as a sequence of symbols (timeline.py),
in four steps.

a. Exact loops, innermost first (loops.py). Every exact tandem repeat becomes one item whose symbol ignores the number
   of iterations, until none is left. The outermost loops are then unfolded one level into the items of their
   copies, so that a copy can be compared with a slightly different repetition next to the loop (a first training
   step that differs from the others is not part of the exact loop of the others).
b. Tolerant runs on that sequence of items. A run B^n with differences between its repetitions is recognized by its
   markers: symbols that occur exactly n times in the searched stretch, once in each repetition, always in the same
   order. For every count n that some symbol has, a few anchors among those symbols are tried (evenly spread ones
   first); the symbols that keep a fixed order with the anchor in every repetition are the markers, the anchor moves
   to the earliest of them. A marker may also occur outside the run (the kernel of every fourth layer occurs in the
   forward and again in the backward pass): the occurrences of every symbol are split into groups of regular spacing
   in operations, a new group starting where the distance to the next occurrence leaves half to twice the group's
   first distance, and the GROUPS groups spanning the most operations are searched again as stretches of their own,
   each from just after the occurrence before it to the occurrence after it. Distances and the spread of anchors are
   measured in operations, not items, because folding makes items unequal.
   Neighbouring repetitions are cut between the last marker of one and the first marker of the next, where the
   program moves on to a new unit of work: at a position where the enclosing CPU calls change, at the outermost call
   level that every such stretch offers (e.g. between two layer calls). Every stretch takes the k-th of these
   positions after its last marker, with the same k everywhere so that all repetitions start at the same phase. Of
   the PHASES values of k with the largest total GPU idle gap, and the choice of the largest gap in each stretch, the
   one whose repetitions save the most bits (step c) wins: the phase where the repetitions are most alike and leave
   the least work outside the run. The first repetition starts at the position of that level before its first marker
   whose distance in operations comes closest to the median distance from cut to first marker; the last one ends
   likewise after its last marker.
c. Choice by minimum description length (MDL, as in SQS by Tatti and Vreeken, KDD 2012), priced on the operations
   the repetitions contain, not on items, because one extra operation can keep an inner loop from folding and make
   two nearly equal repetitions look very different as items. Operations are compared by name family (timeline.py),
   so a recompiled kernel that differs only in a number is the same operation. Writing operations one by one costs
   log2(V) bits each, V being the number of distinct families plus one. Writing a run costs its template (per
   family, the median over the repetitions of how often it occurs) at the same rate, log2(n + 1) bits for n, and, for
   every repetition, each operation more or fewer than the template at log2(V) + log2(length + 1) bits. The
   repetitions of a candidate are grouped into blocks of identical content; then, again and again, the two
   neighbouring blocks whose merging saves the most bits (one template for both instead of two) are merged, until no
   merge saves bits. So a stray odd repetition joins its neighbours while a lasting change of content (a new phase)
   stays a separate block. The block that saves the most bits over writing its operations one by one is the
   candidate's run, and the candidate that saves the most wins. The parts before and after it are
   searched the same way; a part where no run saves bits is one segment.
d. Differences. The most common content of a run's repetitions (as a multiset of name families) is its reference;
   every repetition is compared with it as the operations it has more and fewer, so that e.g. prefill steps among
   decode steps or a slower first step stand out. The report counts the repetitions identical to the reference and
   names the most different ones.

Times are taken on the main stream, the one with the most GPU busy time; operations on other streams stay in the
repetition that launched them but do not stretch its time span (a prefetch copy launched by the backward pass may run
during the forward pass). A loop item that spans at least MIN_SHARE of the timeline on the main stream is shown as a
run of its own copies; smaller loops and runs stay inside the parts around them, because a person does not see them
at this zoom level either. Runs with the same loop symbol or the same template get the same letter.

Public components
    Segment(label, begin, end, ops, more, fewer, shape)
        One letter of the pattern. begin and end are lists of times in ms from the start of the trace, one entry per
        repetition; ops lists the GPU operations per repetition; more and fewer list per repetition how many
        operations it has more and fewer than the reference (step d); shape describes the content of a reference
        repetition, e.g. "12 ops, (35 ops)^61, 40 ops". Property count is the number of repetitions.
        Example: `seg.count`, `max(seg.end) - min(seg.begin)` (time spanned by the whole run, in ms).
    Structure(segments, ops, span)
        The segmentation of a whole timeline: segments in time order, the number of GPU operations and the time span
        of the GPU timeline in ms. Property pattern is the pattern string, e.g. "A B^20 C^10 D". str(structure) is
        the report shown to the user: the pattern, then every segment with its content, its differences and its first
        two and last repetitions.
        Example: `print(detect(load("trace.json.gz")))`.
    detect(timeline) -> Structure
        Segments a Timeline as described above.
        Example: `s = detect(tl); [(x.label, x.count) for x in s.segments]`.
"""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from profileinsight.loops import Item, reduce
from profileinsight.timeline import CALL_DEPTH, Timeline

MIN_SHARE = 0.01
ANCHORS = 4
GROUPS = 16
PHASES = 3


@dataclass
class Segment:
    label: str
    begin: list[float]
    end: list[float]
    ops: list[int]
    more: list[int]
    fewer: list[int]
    shape: str

    @property
    def count(self) -> int:
        return len(self.begin)


@dataclass
class Structure:
    segments: list[Segment]
    ops: int
    span: float

    @property
    def pattern(self) -> str:
        return " ".join(s.label if s.count == 1 else f"{s.label}^{s.count}" for s in self.segments)

    def __str__(self) -> str:
        lines = [f"Timeline structure: {self.pattern}",
                 f"{self.ops:,} GPU operations over {self.span:,.3f} ms. Times are ms from the start of the trace.", ""]
        for s in self.segments:
            lines.extend(_describe(s))
            lines.append("")
        return "\n".join(lines).rstrip()


@dataclass
class _Items:
    """The sequence searched in steps b and c: item symbols, their operation ranges, and the operations themselves."""
    sym: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    gap: np.ndarray
    level: np.ndarray
    op: np.ndarray
    names: int
    bits: float


def detect(tl: Timeline) -> Structure:
    items = [x for it in reduce(tl.op, len(tl.names)) for x in (sum(it.copies, []) if it.copies else [it])]
    idle = np.zeros(len(tl.op))
    idle[1:] = tl.start[1:] - np.maximum.accumulate(tl.end)[:-1]
    lo = np.array([it.lo for it in items])
    hi = np.array([it.hi for it in items])
    changed = np.ones((len(items), CALL_DEPTH + 1), dtype=bool)
    changed[1:, :CALL_DEPTH] = tl.call[hi[:-1] - 1] != tl.call[lo[1:]]
    seq = _Items(np.array([it.sym for it in items], dtype=np.int64), lo, hi, idle[lo], changed.argmax(axis=1),
                 tl.family, len(tl.families), math.log2(len(tl.families) + 1))
    busy = np.bincount(np.unique(tl.stream, return_inverse=True)[1], weights=tl.end - tl.start)
    main = tl.stream == np.unique(tl.stream)[busy.argmax()]
    total = tl.end[main].max() - tl.start[main].min()
    groups = []
    for reps in _split(seq, 0, len(items)):
        if len(reps) > 1 and _span(tl, main, items[reps[0][0]:reps[-1][1]]) >= MIN_SHARE * total:
            template = _median(*_contents(seq, reps)[1:])
            groups.append(([items[a:b] for a, b in reps], ("run", tuple(sorted(template.items())))))
            continue
        for it in items[reps[0][0]:reps[-1][1]]:
            if it.copies and _span(tl, main, [it]) >= MIN_SHARE * total:
                groups.append((it.copies, ("loop", it.sym)))
            elif groups and groups[-1][1] is None:
                groups[-1][0][0].append(it)
            else:
                groups.append(([[it]], None))
    labels: dict = {}
    segments = []
    for reps, key in groups:
        key = key or len(segments)
        labels.setdefault(key, _letter(len(labels)))
        segments.append(_segment(tl, main, labels[key], reps))
    return Structure(segments, len(tl.op), total / 1e3)


def _letter(k: int) -> str:
    return chr(65 + k) if k < 26 else chr(64 + k // 26) + chr(65 + k % 26)


def _span(tl: Timeline, main: np.ndarray, items: list[Item]) -> float:
    """Time from the first start to the last end of the items' operations on the main stream, 0 if they have none."""
    begin, end = _window(tl, main, items[0].lo, items[-1].hi)
    return end - begin


def _window(tl: Timeline, main: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """(first start, last end) of operations [lo, hi) on the main stream, or of all of them if none is on it."""
    on = main[lo:hi]
    if not on.any():
        return tl.start[lo], tl.start[lo]
    return tl.start[lo:hi][on].min(), tl.end[lo:hi][on].max()


def _segment(tl: Timeline, main: np.ndarray, label: str, reps: list[list[Item]]) -> Segment:
    content = [Counter(tl.family[r[0].lo:r[-1].hi].tolist()) for r in reps]
    keys = [tuple(sorted(c.items())) for c in content]
    reference = content[keys.index(Counter(keys).most_common(1)[0][0])]
    more = [sum((c - reference).values()) for c in content]
    fewer = [sum((reference - c).values()) for c in content]
    times = [_window(tl, main, r[0].lo, r[-1].hi) for r in reps]
    return Segment(label, [(b - tl.origin) / 1e3 for b, _ in times], [(e - tl.origin) / 1e3 for _, e in times],
                   [r[-1].hi - r[0].lo for r in reps], more, fewer, _shape(reps[content.index(reference)]))


def _shape(items: list[Item]) -> str:
    """Content of one repetition: plain operations as "n ops", loops as "(ops per copy)^count", at most 8 terms."""
    terms: list = []
    for it in items:
        if it.copies:
            terms.append(f"({it.copies[0][-1].hi - it.copies[0][0].lo:,} ops)^{it.count}")
        elif terms and isinstance(terms[-1], int):
            terms[-1] += 1
        else:
            terms.append(1)
    words = [t if isinstance(t, str) else f"{t:,} ops" for t in terms]
    return ", ".join(words[:8]) + (f", ... ({len(words) - 8} more terms)" if len(words) > 8 else "")


def _describe(s: Segment) -> list[str]:
    dur = np.array(s.end) - np.array(s.begin)
    if s.count == 1:
        lines = [f"{s.label}: one part, {dur[0]:,.3f} ms, {s.ops[0]:,} ops", f"  content: {s.shape}"]
    else:
        lines = [f"{s.label}^{s.count}: {s.count} repetitions over {max(s.end) - min(s.begin):,.3f} ms; each "
                 f"{np.median(dur):,.3f} ms median ({dur.min():,.3f} to {dur.max():,.3f}), "
                 f"{int(np.median(s.ops)):,} ops median ({min(s.ops):,} to {max(s.ops):,})",
                 f"  content of each {s.label}: {s.shape}", "  " + _differences(s, dur)]
    shown = [0, 1, s.count - 1] if s.count > 3 else range(s.count)
    for i in shown:
        if s.count > 3 and i == s.count - 1:
            lines.append(f"    ... {s.count - 3} more")
        name = s.label if s.count == 1 else f"{s.label}{i + 1}"
        lines.append(f"  {name:<8} begins {s.begin[i]:>12,.3f}   ends {s.end[i]:>12,.3f}")
    return lines


def _differences(s: Segment, dur: np.ndarray) -> str:
    same = sum(1 for m, f in zip(s.more, s.fewer) if m == f == 0)
    if same == s.count:
        return f"all {s.count} repetitions have the same operations"
    odd = sorted((i for i in range(s.count) if s.more[i] or s.fewer[i]), key=lambda i: -(s.more[i] + s.fewer[i]))
    words = [f"{s.label}{i + 1} +{s.more[i]:,}/-{s.fewer[i]:,} ops ({dur[i]:,.3f} ms)" for i in odd[:4]]
    return (f"{same:,} of {s.count:,} repetitions have the same operations; most different: " + ", ".join(words)
            + (f", and {len(odd) - 4:,} more" if len(odd) > 4 else ""))


def _split(seq: _Items, lo: int, hi: int) -> list[list[tuple[int, int]]]:
    """Item ranges of the runs and parts of [lo, hi), in order; a part is a run of one repetition."""
    if hi - lo < 4:
        return [[(lo, hi)]] if hi > lo else []
    reps = _best_run(seq, lo, hi)
    if not reps:
        return [[(lo, hi)]]
    return _split(seq, lo, reps[0][0]) + [reps] + _split(seq, reps[-1][1], hi)


def _best_run(seq: _Items, lo: int, hi: int) -> list[tuple[int, int]]:
    best = (0.0, [])
    tried = set()
    for reps in _candidates(seq, lo, hi, None):
        if tuple(reps) not in tried:
            tried.add(tuple(reps))
            found = _best_block(seq, reps)
            if found[0] > best[0]:
                best = found
    return best[1]


def _candidates(seq: _Items, lo: int, hi: int, only: int | None):
    """Repetition grids of items [lo, hi), from markers of every count (or of count `only`) and, at the outer level,
    from the regularly spaced groups of each anchor's occurrences."""
    _, inverse, counts = np.unique(seq.sym[lo:hi], return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable") + lo
    offsets = np.concatenate([[0], np.cumsum(counts)])
    stretches = set()
    for c in np.unique(counts):
        if c < 2 or (only is not None and c != only):
            continue
        syms = np.nonzero(counts == c)[0]
        positions = order[offsets[syms][:, None] + np.arange(c)]
        spread = np.diff(seq.lo[positions], axis=1).min(axis=1)
        for anchor in np.argsort(-spread, kind="stable")[:ANCHORS]:
            yield _repetitions(positions[_markers(positions, int(anchor))], seq, lo, hi)
        if only is None:
            for p in positions:
                for i, j in _regular(seq.lo[p]):
                    if j - i < c:
                        a, b = int(p[i - 1]) + 1 if i else lo, int(p[j]) if j < c else hi
                        stretches.add((int(seq.lo[p[j - 1]] - seq.lo[p[i]]), a, b, j - i))
    for _, a, b, n in sorted(stretches, reverse=True)[:GROUPS]:
        yield from _candidates(seq, a, b, n)


def _regular(p: np.ndarray) -> list[tuple[int, int]]:
    """[i, j) index ranges of at least two consecutive positions whose distances stay within half to twice the first."""
    groups = [[0, 1]]
    ref = None
    for k, d in enumerate(np.diff(p)):
        if ref is None or ref / 2 <= d <= 2 * ref:
            groups[-1][1] = k + 2
            ref = d if ref is None else ref
        else:
            groups.append([k + 1, k + 2])
            ref = None
    return [(i, j) for i, j in groups if j - i >= 2]


def _markers(positions: np.ndarray, anchor: int) -> np.ndarray:
    """Rows of positions that keep a fixed order with the anchor in every repetition, starting from the earliest."""
    for _ in range(len(positions)):
        a = positions[anchor]
        nxt = np.append(a[1:], np.iinfo(np.int64).max)
        prv = np.concatenate([[-1], a[:-1]])
        before = ((positions < a) & (positions > prv)).all(axis=1)
        if not before.any():
            break
        rows = np.nonzero(before)[0]
        anchor = int(rows[np.argmin(positions[rows, 0])])
    return np.nonzero(((positions >= a) & (positions < nxt)).all(axis=1))[0]


def _repetitions(markers: np.ndarray, seq: _Items, lo: int, hi: int) -> list[tuple[int, int]]:
    """[begin, end) of every repetition, cut where the CPU calls change between the markers of neighbouring ones."""
    first, last = markers.min(axis=0), markers.max(axis=0)
    stretches = [np.arange(last[i - 1] + 1, first[i] + 1) for i in range(1, len(first))]
    level = max(int(seq.level[s].min()) for s in stretches)
    options = [s[seq.level[s] <= level] for s in stretches]
    total = [sum(seq.gap[o[j]] for o in options) for j in range(min(len(o) for o in options))]
    choices = [[int(o[k]) for o in options] for k in np.argsort(total)[::-1][:PHASES]]
    choices.append([int(o[np.argmax(seq.gap[o])]) for o in options])
    runs = []
    for cuts in choices:
        lead = np.median([seq.lo[f] - seq.lo[c] for f, c in zip(first[1:], cuts)])
        tail = np.median([seq.lo[c] - seq.lo[t] for t, c in zip(last[:-1], cuts)])
        begin = _closest(seq, np.arange(lo, first[0] + 1), level, seq.lo[first[0]] - lead)
        end = _closest(seq, np.arange(last[-1] + 1, hi + 1), level, seq.lo[last[-1]] + tail)
        bounds = [begin] + cuts + [end]
        runs.append(list(zip(bounds[:-1], bounds[1:])))
    return max(runs, key=lambda run: _saved(seq, run))


def _closest(seq: _Items, positions: np.ndarray, level: int, target: float) -> int:
    """The position (a cut before that item; len(items) is the end) at most `level` deep whose first operation is
    closest to `target`; any position if none is that shallow."""
    at = np.append(seq.lo, seq.hi[-1])[positions]
    shallow = np.append(seq.level, 0)[positions] <= level
    pool = np.nonzero(shallow)[0] if shallow.any() else np.arange(len(positions))
    return int(positions[pool[np.argmin(np.abs(at[pool] - target))]])


def _contents(seq: _Items, reps: list[tuple[int, int]]) -> tuple[list[tuple], dict[tuple, Counter], Counter]:
    """Per repetition the key of its operation content; per key the operation counts; how many repetitions per key."""
    begin = seq.lo[[a for a, _ in reps]]
    end = seq.hi[[b - 1 for _, b in reps]]
    rep = np.repeat(np.arange(len(reps)), end - begin)
    pairs, counts = np.unique(rep * seq.names + seq.op[begin[0]:end[-1]], return_counts=True)
    owner = pairs // seq.names
    split = np.searchsorted(owner, np.arange(len(reps) + 1))
    keys = [tuple(zip((pairs[i:j] % seq.names).tolist(), counts[i:j].tolist())) for i, j in zip(split[:-1], split[1:])]
    return keys, {k: Counter(dict(k)) for k in set(keys)}, Counter(keys)


def _best_block(seq: _Items, reps: list[tuple[int, int]]) -> tuple[float, list[tuple[int, int]]]:
    """The contiguous block of repetitions that saves the most bits, and the bits it saves."""
    keys, content, _ = _contents(seq, reps)
    blocks: list[list] = []
    for i, k in enumerate(keys):
        if blocks and blocks[-1][3] == k:
            blocks[-1][1] = i + 1
            blocks[-1][2][k] += 1
        else:
            blocks.append([i, i + 1, Counter({k: 1}), k])
    merged = [b[:3] for b in blocks]
    costs = [_cost(b[2], content, seq.bits) for b in merged]
    saving = [costs[m] + costs[m + 1] - _cost(merged[m][2] + merged[m + 1][2], content, seq.bits)
              for m in range(len(merged) - 1)]
    while saving and max(saving) > 0:
        m = int(np.argmax(saving))
        merged[m] = [merged[m][0], merged[m + 1][1], merged[m][2] + merged[m + 1][2]]
        costs[m] = _cost(merged[m][2], content, seq.bits)
        del merged[m + 1], costs[m + 1], saving[m]
        for n in (m - 1, m):
            if 0 <= n < len(merged) - 1:
                saving[n] = costs[n] + costs[n + 1] - _cost(merged[n][2] + merged[n + 1][2], content, seq.bits)
    best = (0.0, [])
    for i, j, kinds in merged:
        ops = sum(sum(content[k].values()) * n for k, n in kinds.items())
        gain = ops * seq.bits - _cost(kinds, content, seq.bits)
        if j - i >= 2 and gain > best[0]:
            best = (gain, reps[i:j])
    return best


def _saved(seq: _Items, reps: list[tuple[int, int]]) -> float:
    """Bits saved by writing the repetitions as one run instead of operation by operation."""
    _, content, kinds = _contents(seq, reps)
    return sum(sum(content[k].values()) * n for k, n in kinds.items()) * seq.bits - _cost(kinds, content, seq.bits)


def _median(content: dict[tuple, Counter], kinds: Counter) -> Counter:
    """Per symbol, the median over repetitions of how often it occurs (zero where it is absent)."""
    total = sum(kinds.values())
    values: dict[int, list[tuple[int, int]]] = {}
    for k, n in kinds.items():
        for s, c in content[k].items():
            values.setdefault(s, []).append((c, n))
    template = Counter()
    for s, vals in values.items():
        seen = total - sum(n for _, n in vals)
        if 2 * seen >= total:
            continue
        for c, n in sorted(vals):
            seen += n
            if 2 * seen >= total:
                template[s] = c
                break
    return template


def _cost(kinds: Counter, content: dict[tuple, Counter], bits: float) -> float:
    """Bits to write the repetitions of a block as one template plus each repetition's differences from it."""
    template = _median(content, kinds)
    cost = sum(template.values()) * bits + math.log2(sum(kinds.values()) + 1)
    for k, n in kinds.items():
        c = content[k]
        diff = sum(abs(c[s] - template[s]) for s in c.keys() | template.keys())
        cost += n * diff * (bits + math.log2(sum(c.values()) + 1))
    return cost
