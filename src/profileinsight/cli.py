"""Command line interface: `profileinsight <command> TRACE`.

Each command loads the GPU timeline of a trace (timeline.py), runs one analysis and prints its report as plain text,
so that a person or an agent can read it directly.

Public components
    main(argv=None)
        Parses the command line and runs the command. Commands:
            structure TRACE   the cycles of the timeline (cycles.py), e.g. "Timeline structure: A B^20 C^10 D".
        Example: `profileinsight structure trace_rank0.json.gz`, or `main(["structure", "trace_rank0.json.gz"])`.
"""
import argparse

from profileinsight.cycles import detect
from profileinsight.timeline import load


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="profileinsight", description="Find why a GPU job is slow from its profile.")
    commands = parser.add_subparsers(dest="command", required=True)
    structure = commands.add_parser("structure", help="the cycles of the GPU timeline, e.g. A B^20 C^10 D")
    structure.add_argument("trace", help="PyTorch profiler trace, .json or .json.gz")
    args = parser.parse_args(argv)
    print(detect(load(args.trace)))
