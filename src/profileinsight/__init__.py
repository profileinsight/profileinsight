"""ProfileInsight: find why a GPU job is slow from its profile.

The package reads PyTorch profiler traces and reports what a person would see first when reading them. Modules:
timeline.py (the GPU timeline in program order), loops.py (exact loops), cycles.py (rule 1: the cycles of the
timeline, e.g. A B^20 C^10 D), bubbles.py (rule 2: the times without compute kernels, their reasons, summed per cycle)
and cli.py (the `profileinsight` command).

Public components
    __version__   str, the package version, kept equal to `version` in pyproject.toml.
                  Example: `import profileinsight; print(profileinsight.__version__)` prints `0.0.3`.
"""

__version__ = "0.0.3"
