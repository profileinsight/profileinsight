"""Command line interface: `profileinsight PROFILE` prints the analysis report of a profile.

The report is plain text, so that a person or an agent can read it directly. It has three parts: the cycles of the
GPU timeline (cycles.py, rule 1), its bubbles with their reasons, summed per cycle (bubbles.py, rule 2), and where the
busy time goes, by family of operations, with their efficiency flags (busy.py, rule 3).

Public components
    main(argv=None)
        Reads the profile named on the command line (a PyTorch profiler trace, .json or .json.gz) and prints its report.
        Example: `profileinsight trace_rank0.json.gz`, or `main(["trace_rank0.json.gz"])`.
"""
import argparse

from profileinsight import busy
from profileinsight.bubbles import find, report
from profileinsight.cycles import detect
from profileinsight.timeline import load


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="profileinsight", description="Find why a GPU job is slow from its profile.")
    parser.add_argument("profile", help="PyTorch profiler trace, .json or .json.gz")
    tl = load(parser.parse_args(argv).profile)
    structure = detect(tl)
    if tl.unlaunched:
        print(f"{tl.unlaunched:,} GPU operations at the start of the trace were launched before the recorded window "
              "and are left out.")
    print(structure)
    print()
    print(report(tl, structure, find(tl)))
    print()
    print(busy.report(tl, busy.find(tl)))
