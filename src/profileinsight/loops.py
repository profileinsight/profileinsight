"""Exact loops of a symbol sequence, innermost first, each replaced by one loop symbol that ignores its count.

This is the greedy loop-nest discovery of Xu and Subhlok ("Efficient Discovery of Loop Nests in Communication Traces
of Parallel Programs", University of Houston, 2008), with one change: the loop symbol does not depend on the number of
iterations. A tandem repeat (a body repeated two or more times back to back) is a loop. In each round, all maximal
tandem repeats with a period up to the current limit are found; the ones with the shortest period are taken first and
overlapping longer ones wait for a later round. Each taken repeat becomes one item whose symbol is the body read from
its least rotation, so the same loop seen at another phase gets the same symbol, and whose count is the number of
whole copies (a partial last copy stays outside). The rounds go on until no repeat is left. Because the symbol ignores
the count, a step with 61 layers and a step with 60 layers become the same sequence one level up, which is how small
differences between repetitions vanish before outer loops are compared. The period limit starts at START_PERIOD and
doubles while nothing is found, up to MAX_PERIOD, which bounds the cost (each round is O(n * limit)).

Public components
    Item(sym, lo, hi, count=1, copies=None)
        One element of a reduced sequence. sym is an operation name index, or a loop symbol (at least the number of
        operation names). lo and hi are the range of operations it covers, [lo, hi), in timeline order. count is the
        number of loop copies, 1 for an operation; copies lists, for a loop, the items of every copy.
        Example: `item.hi - item.lo` is the number of operations covered; `len(item.copies) == item.count`.
    reduce(op, names) -> list[Item]
        Reduces the operation sequence `op` (int array of name indices, `names` distinct names) to a fixed point.
        Example: `items = reduce(tl.op, len(tl.names)); print(len(tl.op), "ops ->", len(items), "items")`.
"""
from dataclasses import dataclass

import numpy as np

START_PERIOD = 32
MAX_PERIOD = 4096


@dataclass
class Item:
    sym: int
    lo: int
    hi: int
    count: int = 1
    copies: list[list["Item"]] | None = None


def reduce(op: np.ndarray, names: int) -> list[Item]:
    items = [Item(int(s), k, k + 1) for k, s in enumerate(op)]
    loops: dict[tuple, int] = {}
    limit = START_PERIOD
    while True:
        sym = np.array([it.sym for it in items], dtype=np.int64)
        runs = _runs(sym, min(limit, len(sym) // 2))
        if runs:
            items = _fold(items, sym, runs, loops, names)
            limit = START_PERIOD
        elif limit < min(MAX_PERIOD, len(sym) // 2):
            limit *= 2
        else:
            return items


def _runs(sym: np.ndarray, limit: int) -> list[tuple[int, int, int]]:
    """Non-overlapping tandem repeats (period, start, copies), shortest period first."""
    taken = np.zeros(len(sym), dtype=bool)
    found = []
    for p in range(1, limit + 1):
        same = np.concatenate([[0], (sym[:-p] == sym[p:]).astype(np.int8), [0]])
        edges = np.diff(same)
        starts, stops = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        long = stops - starts >= p
        for s, t in zip(starts[long], stops[long]):
            copies = (t + p - s) // p
            if not taken[s:s + copies * p].any():
                taken[s:s + copies * p] = True
                found.append((p, int(s), int(copies)))
    return found


def _fold(items: list[Item], sym: np.ndarray, runs: list[tuple[int, int, int]], loops: dict[tuple, int],
          names: int) -> list[Item]:
    out = []
    k = 0
    for p, s, copies in sorted(runs, key=lambda r: r[1]):
        out.extend(items[k:s])
        body = tuple(sym[s:s + p].tolist())
        r = _least_rotation(body)
        key = body[r:] + body[:r]
        out.append(Item(loops.setdefault(key, names + len(loops)), items[s].lo, items[s + copies * p - 1].hi, copies,
                        [items[s + i * p:s + (i + 1) * p] for i in range(copies)]))
        k = s + copies * p
    out.extend(items[k:])
    return out


def _least_rotation(t: tuple) -> int:
    """Start index of the lexicographically least rotation of t (Booth's algorithm)."""
    s = t + t
    f = [-1] * len(s)
    k = 0
    for j in range(1, len(s)):
        i = f[j - k - 1]
        while i != -1 and s[j] != s[k + i + 1]:
            if s[j] < s[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if s[j] != s[k + i + 1]:
            if s[j] < s[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % len(t)
