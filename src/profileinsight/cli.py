"""Command line interface: `profileinsight PROFILE` prints the analysis report of a profile.

The report is plain text, so that a person or an agent can read it directly. It has two parts: the cycles of the GPU
timeline (cycles.py, rule 1) and its bubbles with their reasons, summed per cycle (bubbles.py, rule 2).

Public components
    main(argv=None)
        Reads the profile named on the command line (a PyTorch profiler trace, .json or .json.gz) and prints its report.
        Example: `profileinsight trace_rank0.json.gz`, or `main(["trace_rank0.json.gz"])`.
"""
import argparse

from profileinsight.bubbles import find, report
from profileinsight.cycles import detect
from profileinsight.timeline import load


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="profileinsight", description="Find why a GPU job is slow from its profile.")
    parser.add_argument("profile", help="PyTorch profiler trace, .json or .json.gz")
    tl = load(parser.parse_args(argv).profile)
    structure = detect(tl)
    print(structure)
    print()
    print(report(tl, structure, find(tl)))
