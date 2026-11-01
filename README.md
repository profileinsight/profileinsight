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
profileinsight structure trace_rank0.json.gz
```

prints the cycles of the GPU timeline, such as `Timeline structure: A B^8 C` for a setup part, eight repetitions and a
tail, with the time of every repetition and how the repetitions differ from each other. The method is described at the
top of `src/profileinsight/cycles.py`.

## Release

Pushing a tag `v<version>` that matches `version` in `pyproject.toml` builds the package and publishes it to PyPI through trusted publishing (`.github/workflows/build.yml`).
