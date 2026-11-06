"""Command line interface: `profileinsight PROFILE [PROFILE ...]` prints the analysis report of a profile.

The report is plain text, so that a person or an agent can read it directly. It has three parts: the cycles of the
GPU timeline (cycles.py, rule 1), its bubbles with their reasons, summed per cycle (bubbles.py, rule 2), and where the
busy time goes, by family of operations, with their efficiency flags (busy.py, rule 3). Given the profiles of several
ranks of one job, it reports on the first one and adds a fourth part: which rank the others wait for at the
collectives, and what that rank does beyond them (ranks.py, rule 4). Given one profile that has collectives, it says
how to compare the ranks.

Public components
    main(argv=None)
        Reads the profiles named on the command line (PyTorch profiler traces, .json or .json.gz) and prints the
        report. Example: `profileinsight trace_rank0.json.gz`, `profileinsight trace_rank*.json.gz`, or
        `main(["trace_rank0.json.gz"])`.
"""
import argparse
import os

from profileinsight import busy, ranks
from profileinsight.bubbles import find, report
from profileinsight.cycles import detect
from profileinsight.timeline import load


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="profileinsight", description="Find why a GPU job is slow from its profile.")
    parser.add_argument("profiles", nargs="+", help="PyTorch profiler traces, .json or .json.gz; several for the "
                                                    "ranks of one job (the first is reported on in full)")
    paths = parser.parse_args(argv).profiles
    tls = [load(p) for p in paths]
    tl = tls[0]
    structure = detect(tl)
    if len(paths) > 1:
        print(f"Profiles of {len(paths)} ranks; parts 1-3 are of {paths[0]}, part 4 compares the ranks.")
    if tl.unlaunched:
        print(f"{tl.unlaunched:,} GPU operations at the start of the trace were launched before the recorded window "
              "and are left out.")
    print(structure)
    print()
    print(report(tl, structure, find(tl)))
    print()
    print(busy.report(tl, busy.find(tl)))
    print()
    compared = ranks.compare(tls, [os.path.basename(p) for p in paths])
    if len(paths) > 1:
        print(ranks.report(tls, compared) if compared else "Ranks: the profiles have no collective in common.")
    elif len(ranks.collectives(tl)):
        print("Ranks: this profile has collectives, whose time includes waiting for the other ranks; one rank's profile"
              " cannot tell slow communication from a late rank. Pass the profiles of all ranks to compare them: "
              "profileinsight trace_rank0.json.gz trace_rank1.json.gz ...")
