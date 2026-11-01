# ProfileInsight

Find why your GPU job is slow from its profile.

ProfileInsight reads a PyTorch profiler trace (Chrome trace JSON, optionally gzip-compressed) and reports what a person
would see first when reading it. It is in an early stage.

## Install

```
pip install profileinsight
```

## Use

```
profileinsight trace_rank0.json.gz
```

prints the analysis report of the profile:

1. the cycles of the GPU timeline, such as `Timeline structure: A B^8 C` for a setup part, eight repetitions and a
   tail, with the time of every repetition and how the repetitions differ from each other
   (`src/profileinsight/cycles.py`);
2. the bubbles, the times when no compute kernel runs, with their reasons (communication, copies, a host
   synchronization, a slow host, ...), summed per repetition, and an example timeline for communication and copy
   bubbles (`src/profileinsight/bubbles.py`).

The rules behind each part are described at the top of its file.

## Release

Pushing a tag `v<version>` that matches `version` in `pyproject.toml` builds the package and publishes it to PyPI through trusted publishing (`.github/workflows/build.yml`).
