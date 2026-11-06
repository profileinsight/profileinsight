"""ProfileInsight: find why a GPU job is slow from its profile.

The package reads PyTorch profiler traces and reports what a person would see first when reading them. Modules:
timeline.py (the GPU timeline in program order), loops.py (exact loops), cycles.py (rule 1: the cycles of the
timeline, e.g. A B^20 C^10 D), bubbles.py (rule 2: the times without compute kernels, their reasons, summed per cycle),
busy.py (rule 3: where the busy time goes, by family, with efficiency flags), ranks.py (rule 4: across the ranks of
a job, which rank the others wait for at the collectives, and why), signs.py (rule 5: signs of common
misconfigurations) and cli.py (the `profileinsight` command, which opens with a verdict).

Public components
    __version__   str, the package version, kept equal to `version` in pyproject.toml.
                  Example: `import profileinsight; print(profileinsight.__version__)` prints `0.0.5`.
"""

__version__ = "0.0.5"
