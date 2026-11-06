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
profileinsight trace_rank0.json.gz trace_rank1.json.gz trace_rank2.json.gz trace_rank3.json.gz
```

prints the analysis report of the profile (the first one, when several are given):

1. the cycles of the GPU timeline, such as `Timeline structure: A B^8 C` for a setup part, eight repetitions and a
   tail, with the time of every repetition and how the repetitions differ from each other
   (`src/profileinsight/cycles.py`);
2. the bubbles, the times when no compute kernel runs, with their reasons (communication, copies, a host
   synchronization, a slow host, ...) taken from what the GPU actually waited for, summed per repetition, and an
   example timeline for each listed group: for a copy or communication the awaited operation's size, rate, launch
   time and the queue ahead of it, for a sync how long the host was blocked (`src/profileinsight/bubbles.py`);
3. where the busy time goes, by family of operations, with the call site, launch facts and flags for small,
   underfilled or pageable work (`src/profileinsight/busy.py`);
4. given the profiles of several ranks of one job: which rank the others wait for at the collectives, how long, and
   the extra work that rank did before them (`src/profileinsight/ranks.py`). A single rank's profile cannot tell slow
   communication from a late peer.

The rules behind each part are described at the top of its file.

## Release

Pushing a tag `v<version>` that matches `version` in `pyproject.toml` builds the package and publishes it to PyPI through trusted publishing (`.github/workflows/build.yml`).
