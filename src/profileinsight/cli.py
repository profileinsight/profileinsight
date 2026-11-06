"""Command line interface: `profileinsight PROFILE [PROFILE ...]` prints the analysis report of a profile.

The report is plain text, so that a person or an agent can read it directly. It opens with a verdict: the few
findings that matter most, one line each with its evidence (and for a sign of misconfiguration the setting to check),
in this order: the rank the others wait for (rule 4) if they wait for it at least HELD_SHARE of the span, the signs
of misconfiguration (rule 5), the main reason of the bubbles (rule 2) if the GPU computes nothing for at least
IDLE_SHARE of the span, and the rank the others wait for if they wait less than HELD_SHARE. Readers act on the first
lines, so the verdict comes before the details. The parts follow: the cycles of the GPU timeline (cycles.py,
rule 1), its bubbles with their reasons, summed per cycle (bubbles.py, rule 2), where the busy time goes, by family of
operations, with their efficiency flags (busy.py, rule 3), given the profiles of several ranks of one job, which rank
the others wait for at the collectives and what that rank does beyond them (ranks.py, rule 4), and the signs of
misconfiguration (signs.py, rule 5). Parts 1-3 and 5 are of the first profile. Given one profile that has
collectives, it says how to compare the ranks.

Public components
    IDLE_SHARE, HELD_SHARE
        The share of the span without compute from which the bubbles' main reason enters the verdict, and the share
        of the span the others wait for one rank from which that rank comes first in it.
    main(argv=None)
        Reads the profiles named on the command line (PyTorch profiler traces, .json or .json.gz) and prints the
        report. Example: `profileinsight trace_rank0.json.gz`, `profileinsight trace_rank*.json.gz`, or
        `main(["trace_rank0.json.gz"])`.
"""
import argparse
import os
from collections import Counter

from profileinsight import busy, ranks, signs
from profileinsight.bubbles import find, report
from profileinsight.cycles import detect
from profileinsight.timeline import load

IDLE_SHARE = 0.1
HELD_SHARE = 0.1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="profileinsight", description="Find why a GPU job is slow from its profile.")
    parser.add_argument("profiles", nargs="+", help="PyTorch profiler traces, .json or .json.gz; several for the "
                                                    "ranks of one job (the first is reported on in full)")
    paths = parser.parse_args(argv).profiles
    tls = [load(p) for p in paths]
    tl = tls[0]
    structure = detect(tl)
    bubbles = find(tl)
    found = signs.find(tl)
    compared = ranks.compare(tls, [os.path.basename(p) for p in paths])
    late = ranks.headline(tls, compared) if compared and len(paths) > 1 else ""
    strong = late and ranks.held_share(compared) >= HELD_SHARE
    verdict = [late] if strong else []
    verdict += [f"{s.name}: {s.evidence}; check {s.check}" for s in found]
    span = float(tl.end.max() - tl.start.min())
    idle = sum(b.dur for b in bubbles)
    if idle >= IDLE_SHARE * span:
        reason, time = Counter({r: sum(b.dur for b in bubbles if b.reason == r) for r in
                                {b.reason for b in bubbles}}).most_common(1)[0]
        verdict.append(f"the GPU computes nothing {100 * idle / span:.0f}% of the span, mostly for reason "
                       f"\"{reason}\" ({time / 1e3:,.1f} ms; part 2 says where)")
    verdict = [v for v in verdict + ([] if strong else [late]) if v]
    print("Verdict (most important first; the parts below give the details):")
    for k, v in enumerate(verdict or ["nothing stands out: no rank holds the others up, no sign of misconfiguration, "
                                      "and the GPU computes most of the time (part 3 says where the time goes)"]):
        print(f"  {k + 1}. {v}")
    print()
    if len(paths) > 1:
        print(f"Profiles of {len(paths)} ranks; parts 1-3 and 5 are of {paths[0]}, part 4 compares the ranks.")
    if tl.unlaunched:
        print(f"{tl.unlaunched:,} GPU operations at the start of the trace were launched before the recorded window "
              "and are left out.")
    print(structure)
    print()
    print(report(tl, structure, bubbles))
    print()
    print(busy.report(tl, busy.find(tl)))
    print()
    if len(paths) > 1:
        print(ranks.report(tls, compared) if compared else "Ranks: the profiles have no collective in common.")
    elif len(ranks.collectives(tl)):
        print("Ranks: this profile has collectives, whose time includes waiting for the other ranks; one rank's profile"
              " cannot tell slow communication from a late rank. Pass the profiles of all ranks to compare them: "
              "profileinsight trace_rank0.json.gz trace_rank1.json.gz ...")
    print()
    print(signs.report(found))
