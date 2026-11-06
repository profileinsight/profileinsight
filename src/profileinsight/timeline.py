"""The GPU timeline of a profile: every kernel, memory copy and memory set as one row, in program order.

A PyTorch profiler (Kineto) trace is a Chrome trace JSON file, optionally gzip-compressed. Its GPU operations are the
complete events ("ph": "X") of the categories in GPU_CATEGORIES. Each one carries a correlation id that links it to the
host call that launched it (a runtime or driver event with the same id). Operations are ordered by the start time of
that launch call, then by GPU start time (all kernels of one CUDA graph launch share the launch call). Launch order is
the order of the program, so it repeats exactly from iteration to iteration, while GPU start order varies with the
timing of concurrent streams. For every operation the timeline also keeps the CPU calls (CALL_CATEGORIES events on the
launching thread) that enclosed its launch, outermost first, up to CALL_DEPTH levels: where these change, the program
moves from one unit of work to the next; it also keeps the launch call itself and the innermost CPU calls around it (the
call site). The host's synchronizing calls (SYNC_CALLS: stream, device and event synchronize, synchronous copies) are
kept with their call sites, because a synchronize empties the GPU's queue of work. All analyses in this package read
the timeline as a sequence of symbols: each operation is replaced by the index of its name, so that repeated work
becomes repeated symbols. Each operation also has a family: its name with every run of digits replaced by "#", so that
kernels that differ only in a number (a recompiled triton_poi_fused__to_copy_21 and _3, tile sizes in template
arguments) count as the same kind of work when repetitions are compared.

Public components
    GPU_CATEGORIES, LAUNCH_CATEGORIES, CALL_CATEGORIES, CALL_DEPTH, SITE_DEPTH, SYNC_CALLS
        The trace event categories of GPU operations, of the host calls that launch them and of the CPU calls that
        enclose launches; the number of outermost enclosing call levels kept as ids, and of innermost ones kept as
        names (the call site); the names of synchronizing host calls.
    Sync(thread, start, end, name, site)
        One synchronizing host call: its thread, start and end in microseconds, name and call site (innermost CPU call
        names first). Example: `f"{s.name} in {' < '.join(s.site)}"`.
    DTYPE_BYTES
        The size in bytes of the element types that collectives report.
    Timeline(names, op, families, family, kind, start, end, stream, call, launch_start, launch_end, launch_thread,
             launch_call, site, syncs, bytes, blocks_per_sm, occupancy, registers, dims, comm_bytes, comm_ranks,
             host_calls, unlaunched, origin)
        names   list[str], the distinct operation names.
        op      int64 array, per operation the index of its name in `names`, in launch order.
        families list[str], the distinct name families.
        family  int64 array, per operation the index of its family in `families`.
        start   float64 array, per operation the GPU start time in microseconds.
        end     float64 array, per operation the GPU end time in microseconds.
        stream  int64 array, per operation the GPU stream.
        kind    int64 array, per operation the index of its category in GPU_CATEGORIES (0 kernel, 1 copy, 2 set).
        call    int64 array (operations x CALL_DEPTH), per operation the ids of the enclosing CPU calls, outermost
                first, -1 below the innermost one; two operations launched inside the same call share its id.
        launch_start, launch_end, launch_thread
                arrays, per operation the start and end (microseconds) and the thread of its launch call.
        launch_call  list[str], per operation the name of its launch call.
        site    list[tuple[str, ...]], per operation the names of the innermost SITE_DEPTH CPU calls around its launch,
                innermost first.
        syncs   list[Sync], the synchronizing host calls, ordered by thread and start.
        bytes   int64 array, per operation the bytes a copy or memory set moves (its "bytes" argument), 0 for a kernel.
        blocks_per_sm, occupancy, registers
                arrays, per kernel the thread blocks of its grid per SM, its estimated achieved occupancy in percent and
                its registers per thread (the kernel event's "blocks per SM", "est. achieved occupancy %" and
                "registers per thread"); NaN (-1 for registers) for copies, sets and traces without them.
        comm_bytes, comm_ranks
                int64 arrays, per operation launched inside a collective (an enclosing "record_param_comms" CPU event)
                the bytes of its input message ("In msg nelems" times the size of its "dtype", 0 if the type is not in
                DTYPE_BYTES) and the size of its process group; 0 for all other operations.
        host_calls  dict[str, tuple[int, float]], per name of a host runtime or driver call (LAUNCH_CATEGORIES) its
                count and total duration in microseconds over the trace, e.g. host_calls["cudaFree"].
        unlaunched  int, the GPU operations left out because their launch call is not in the trace.
        dims    list, per operation the "Input Dims" of the innermost CPU call around its launch (present when the
                profile was recorded with record_shapes=True), else None.
        origin  float, the earliest timestamp of any event in the trace, in microseconds; reported times are relative
                to it.
        Example: `tl.names[tl.op[0]]` is the name of the first GPU operation launched; `tl.call[k, 0] ==
        tl.call[k + 1, 0]` tells whether operations k and k + 1 were launched inside the same outermost CPU call.
    short(name) -> str
        A kernel name without "void", its parameter list and, when long, its template arguments.
        Example: `short("void cutlass::Kernel<...>(Params)")` is "cutlass::Kernel".
    load(path) -> Timeline
        Reads a trace file (.json or .json.gz) and returns its GPU timeline. GPU operations whose launch call is not in
        the trace are left out: when the host runs ahead of the GPU, the work at the start of the recorded window was
        launched before it (their number is in `unlaunched`). Fails if no GPU operation is left.
        Example: `tl = load("trace_rank0.json.gz"); print(len(tl.op), "GPU operations")`.
"""
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
CALL_CATEGORIES = ("cpu_op", "user_annotation")
CALL_DEPTH = 4
SITE_DEPTH = 3
DTYPE_BYTES = {"BFloat16": 2, "Half": 2, "Float": 4, "Double": 8, "Int": 4, "Long": 8, "Short": 2, "Byte": 1,
               "Char": 1, "Bool": 1, "Float8_e4m3fn": 1, "Float8_e5m2": 1}
SYNC_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy",
              "cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize", "hipStreamSynchronize",
              "hipDeviceSynchronize", "hipEventSynchronize", "hipMemcpy")


@dataclass
class Sync:
    thread: int
    start: float
    end: float
    name: str
    site: tuple[str, ...]


@dataclass
class Timeline:
    names: list[str]
    op: np.ndarray
    families: list[str]
    family: np.ndarray
    kind: np.ndarray
    start: np.ndarray
    end: np.ndarray
    stream: np.ndarray
    call: np.ndarray
    launch_start: np.ndarray
    launch_end: np.ndarray
    launch_thread: np.ndarray
    launch_call: list[str]
    site: list[tuple[str, ...]]
    syncs: list[Sync]
    bytes: np.ndarray
    blocks_per_sm: np.ndarray
    occupancy: np.ndarray
    registers: np.ndarray
    dims: list
    comm_bytes: np.ndarray
    comm_ranks: np.ndarray
    host_calls: dict
    unlaunched: int
    origin: float


def load(path: str) -> Timeline:
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    launch = {e["args"]["correlation"]: e for e in events
              if e.get("cat") in LAUNCH_CATEGORIES and "correlation" in e.get("args", {})}
    every = [e for e in events if e.get("cat") in GPU_CATEGORIES]
    gpu = [e for e in every if e["args"]["correlation"] in launch]
    assert gpu, f"{path} has no GPU operations with a launch call (categories {GPU_CATEGORIES})"
    gpu.sort(key=lambda e: (launch[e["args"]["correlation"]]["ts"], e["ts"]))
    index: dict[str, int] = {}
    op = np.array([index.setdefault(e["name"], len(index)) for e in gpu], dtype=np.int64)
    start = np.array([e["ts"] for e in gpu], dtype=np.float64)
    end = start + np.array([e["dur"] for e in gpu], dtype=np.float64)
    stream = np.array([e["tid"] for e in gpu], dtype=np.int64)
    kind = np.array([GPU_CATEGORIES.index(e["cat"]) for e in gpu], dtype=np.int64)
    calls = [launch[e["args"]["correlation"]] for e in gpu]
    call, site, inner, comm = _calls(events, calls)
    sync = sorted((e for e in events if e.get("cat") in LAUNCH_CATEGORIES and e["name"] in SYNC_CALLS),
                  key=lambda e: (e["tid"], e["ts"]))
    syncs = [Sync(e["tid"], e["ts"], e["ts"] + e["dur"], e["name"], s) for e, s in zip(sync, _calls(events, sync)[1])]
    comm_args = [events[k]["args"] if k >= 0 else {} for k in comm]
    host_calls: dict[str, tuple[int, float]] = {}
    for e in events:
        if e.get("cat") in LAUNCH_CATEGORIES:
            n, t = host_calls.get(e["name"], (0, 0.0))
            host_calls[e["name"]] = (n + 1, t + e["dur"])
    arg = lambda key, empty: np.array([e["args"].get(key, empty) for e in gpu], dtype=np.float64)
    groups: dict[str, int] = {}
    of = np.array([groups.setdefault(re.sub(r"[0-9]+", "#", n), len(groups)) for n in index], dtype=np.int64)
    return Timeline(list(index), op, list(groups), of[op], kind, start, end, stream, call,
                    np.array([c["ts"] for c in calls], dtype=np.float64),
                    np.array([c["ts"] + c["dur"] for c in calls], dtype=np.float64),
                    np.array([c["tid"] for c in calls], dtype=np.int64), [c["name"] for c in calls], site, syncs,
                    np.array([e["args"]["bytes"] if k else 0 for e, k in zip(gpu, kind)], dtype=np.int64),
                    arg("blocks per SM", np.nan), arg("est. achieved occupancy %", np.nan),
                    arg("registers per thread", -1).astype(np.int64),
                    [events[k]["args"].get("Input Dims") if k >= 0 else None for k in inner],
                    np.array([a.get("In msg nelems", 0) * DTYPE_BYTES.get(a.get("dtype"), 0) for a in comm_args],
                             dtype=np.int64),
                    np.array([a.get("Group size", 0) for a in comm_args], dtype=np.int64), host_calls,
                    len(every) - len(gpu),
                    float(min(e["ts"] for e in events)))


def _calls(events: list[dict], launches: list[dict]) -> tuple[np.ndarray, list[tuple[str, ...]], list[int], list[int]]:
    """Per launch, the ids of the outermost CPU calls enclosing it on its thread, outermost first, the names of the
    innermost ones, innermost first, the event index of the innermost one and that of the innermost enclosing
    "record_param_comms" (-1 if none)."""
    calls = defaultdict(list)
    for k, e in enumerate(events):
        if e.get("cat") in CALL_CATEGORIES:
            calls[e["tid"]].append((e["ts"], -e["dur"], k))
    order = defaultdict(list)
    for i, e in enumerate(launches):
        order[e["tid"]].append((e["ts"], i))
    out = np.full((len(launches), CALL_DEPTH), -1, dtype=np.int64)
    sites: list[tuple[str, ...]] = [()] * len(launches)
    inner = [-1] * len(launches)
    comm = [-1] * len(launches)
    for tid, todo in order.items():
        cpu = sorted(calls[tid])
        stack: list[tuple[float, int]] = []
        j = 0
        for ts, i in sorted(todo):
            while j < len(cpu) and cpu[j][0] <= ts:
                while stack and stack[-1][0] < cpu[j][0]:
                    stack.pop()
                stack.append((cpu[j][0] - cpu[j][1], cpu[j][2]))
                j += 1
            while stack and stack[-1][0] < ts:
                stack.pop()
            ids = [k for _, k in stack[:CALL_DEPTH]]
            out[i, :len(ids)] = ids
            sites[i] = tuple(events[k]["name"] for _, k in reversed(stack[-SITE_DEPTH:]))
            inner[i] = stack[-1][1] if stack else -1
            comm[i] = next((k for _, k in reversed(stack) if events[k]["name"] == "record_param_comms"), -1)
    return out, sites, inner, comm


def short(name: str) -> str:
    """A kernel name without "void", its parameter list and, when long, its template arguments."""
    name = re.sub(r"^void ", "", name.replace("(anonymous namespace)", "anonymous").split("(")[0].strip())
    shorter = re.sub(r"<[^<>]*>", "", name)
    while len(name) > 80 and shorter != name:
        name, shorter = shorter, re.sub(r"<[^<>]*>", "", shorter)
    return name
