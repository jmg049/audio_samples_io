#!/usr/bin/env python3
"""Audio file loading and saving in Python: audio_samples against the other libraries.

Runs on Linux, macOS and Windows with one command:

    python python_io_bench.py --dir SOME_DIRECTORY_ON_THE_DISK_TO_TEST

Results go to one file, printed at the end, holding the OS, file system, library versions and one
JSON line per row. A row is one combination of operation, format, sample type, channels and length;
for it, every library takes turns in one process, so that drift affects them equally, and each writes
its own output file. Finished rows are skipped when the same command runs again.

Sections, each measured for every library that can do it:

    read      warm   read a whole file already in the page cache
    read      cold   the same, after dropping that file from the cache before each read
    read-f32  warm/cold   read into float32, which is what librosa.load does by default
    stream    warm/cold   read a file in blocks of 4096 frames
    write     new    write to a path that does not exist, with fsync time as a separate figure
    write     over   write over an existing file, likewise

Each entry records wall time (median and quartiles), CPU time across all threads, and for writes the
time of an fsync after the call. Cold rows include "direct", a plain unbuffered read of the same file
where the OS has one, as the device's speed to compare with.

How the cache is dropped, and what that means, is stated in the results file:

    Linux    fsync, then posix_fadvise(POSIX_FADV_DONTNEED). Advisory, but honoured for clean pages.
    macOS    fsync, then mmap and msync(MS_INVALIDATE), which evicts the file's pages from the unified
             buffer cache. Not documented as a guarantee.
    Windows  Opening the file with FILE_FLAG_NO_BUFFERING makes the cache manager flush and purge its
             cached data for the file when no other cached handle is open. Pages can stay on the
             standby list; a full purge needs administrator rights (RAMMap).

The results file records, for each cold row, how much slower it was than the warm one. A ratio near 1
means the drop did not work on that system.
"""

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

SAMPLE_RATE = 44100
SUBTYPE = {"i16": "PCM_16", "i24": "PCM_24", "i32": "PCM_32", "f32": "FLOAT", "f64": "DOUBLE"}
DTYPE = {"i16": np.int16, "i24": np.int32, "i32": np.int32, "f32": np.float32, "f64": np.float64}
SF_DTYPE = {"i16": "int16", "i24": "int32", "i32": "int32", "f32": "float32", "f64": "float64"}
WIDTH = {"i16": 2, "i24": 3, "i32": 4, "f32": 4, "f64": 8}
SCALE = {"i16": 32767.0, "i24": 8388607.0, "i32": 2147483647.0}
BLOCK_FRAMES = 4096


def optional(name):
    try:
        return __import__(name)
    except Exception:
        return None


aus = optional("audio_samples")
sf = optional("soundfile")
librosa = optional("librosa")
scipy_wav = None
try:
    import scipy.io.wavfile as scipy_wav
except Exception:
    pass


# --- Dropping one file from the cache ----------------------------------------------------------


def fsync_path(path):
    """Make a file durable, as far as the OS can be asked to."""
    if os.name == "nt":
        fd = os.open(path, os.O_RDWR | os.O_BINARY)
    else:
        fd = os.open(path, os.O_RDONLY if sys.platform != "darwin" else os.O_RDWR)
    try:
        if sys.platform == "darwin":
            import fcntl

            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        else:
            os.fsync(fd)
    finally:
        os.close(fd)


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
    ]
    _INVALID = wintypes.HANDLE(-1).value
    _GENERIC_READ, _SHARE_ALL, _OPEN_EXISTING, _NO_BUFFERING = 0x80000000, 7, 3, 0x20000000

    def _open_unbuffered(path):
        handle = _kernel32.CreateFileW(path, _GENERIC_READ, _SHARE_ALL, None, _OPEN_EXISTING, _NO_BUFFERING, None)
        if handle == _INVALID:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed", path)
        return handle

    def drop_cache(path):
        _kernel32.CloseHandle(_open_unbuffered(path))

    def direct_read(path, buffer):
        """Read the file with the cache bypassed, in the 1 MiB requests an application would use."""
        handle = _open_unbuffered(path)
        try:
            got = wintypes.DWORD(0)
            base = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            size = len(buffer)
            step = 1 << 20
            done = 0
            while done < size:
                ok = _kernel32.ReadFile(handle, base + done, min(step, size - done), ctypes.byref(got), None)
                if not ok or got.value == 0:
                    break
                done += got.value
            return done
        finally:
            _kernel32.CloseHandle(handle)

    DROP_METHOD = "CreateFile with FILE_FLAG_NO_BUFFERING before each read (purges the cache manager's data)"
elif sys.platform == "darwin":
    import fcntl

    _libc = None

    def drop_cache(path):
        import ctypes

        global _libc
        if _libc is None:
            _libc = ctypes.CDLL(None, use_errno=True)
            _libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(fd)
            _libc.mmap.restype = ctypes.c_void_p
            _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
            _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            mapped = _libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
            _libc.msync(mapped, size, 2)  # MS_INVALIDATE
            _libc.munmap(mapped, size)
        finally:
            os.close(fd)

    def direct_read(path, buffer):
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            return os.readv(fd, [buffer])
        finally:
            os.close(fd)

    DROP_METHOD = "fsync, then msync(MS_INVALIDATE) on a mapping of the file; direct read with F_NOCACHE"
else:

    def drop_cache(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def direct_read(path, buffer):
        flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
        try:
            fd = os.open(path, flags)
        except OSError:  # tmpfs and some others have no direct I/O
            fd = os.open(path, os.O_RDONLY)
        try:
            return os.preadv(fd, [buffer], 0)
        finally:
            os.close(fd)

    DROP_METHOD = "fsync, then posix_fadvise(POSIX_FADV_DONTNEED); direct read with O_DIRECT"


# --- What the results are from -----------------------------------------------------------------


def file_system_of(path):
    path = os.path.abspath(path)
    try:
        if sys.platform == "win32":
            import ctypes

            root = os.path.splitdrive(path)[0] + "\\"
            name = ctypes.create_unicode_buffer(64)
            ctypes.windll.kernel32.GetVolumeInformationW(root, None, 0, None, None, None, name, 64)
            return {"type": name.value, "mount": root}
        if sys.platform == "darwin":
            out = subprocess.run(["df", path], capture_output=True, text=True).stdout.splitlines()[-1].split()
            info = subprocess.run(["diskutil", "info", out[0]], capture_output=True, text=True).stdout
            kept = [line.strip() for line in info.splitlines() if any(k in line for k in ("File System Personality", "Solid State", "Protocol", "Device / Media Name"))]
            return {"type": kept, "mount": out[-1]}
        out = subprocess.run(["findmnt", "-n", "-o", "FSTYPE,SOURCE,OPTIONS", "--target", path], capture_output=True, text=True).stdout.strip()
        return {"type": out}
    except Exception as e:
        return {"type": f"unknown: {e}"}


def versions():
    found = {"python": platform.python_version(), "numpy": np.__version__}
    for name, module in (("soundfile", sf), ("librosa", librosa)):
        if module:
            found[name] = getattr(module, "__version__", "?")
    if sf:
        found["libsndfile"] = sf.__libsndfile_version__
    if scipy_wav:
        found["scipy"] = __import__("scipy").__version__
    try:
        from importlib.metadata import version

        found["audio_samples"] = version("audio_samples")
    except Exception:
        found["audio_samples"] = "?"
    return found


# --- Signals and files -------------------------------------------------------------------------


def signal(sample_type, channels, seconds):
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(7 + channels)
    x = np.stack(
        [0.5 * np.sin(2 * np.pi * (220 + 55 * c) * t) * np.exp(-((t % 0.7) * 2.0)) + 0.02 * rng.standard_normal(n).clip(-3, 3) for c in range(channels)],
        axis=1,
    )
    if sample_type in SCALE:
        return (x * SCALE[sample_type]).astype(DTYPE[sample_type])
    return x.astype(DTYPE[sample_type])


def input_file(directory, fmt, sample_type, channels, seconds, instance=0):
    """The input file for a row; instances past the first are separate copies, written separately so
    that each has its own place on the disk."""
    suffix = f"_{instance}" if instance else ""
    path = os.path.join(directory, f"in_{sample_type}_{channels}_{seconds}{suffix}.{fmt}")
    if not os.path.exists(path):
        x = signal(sample_type, channels, seconds)
        if sample_type == "i24":
            x = x.astype(np.int32) << 8
        sf.write(path, x, SAMPLE_RATE, subtype=SUBTYPE[sample_type], format=fmt.upper())
        fsync_path(path)
    return path


def i24_bytes(x):
    return x.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


# --- The libraries -----------------------------------------------------------------------------


def readers(fmt, sample_type):
    r = {}
    if aus:
        r["ours"] = lambda p: aus.io.read(p)
    if sf:
        r["soundfile"] = lambda p: sf.read(p, dtype=SF_DTYPE[sample_type], always_2d=True)
    if librosa:
        r["librosa"] = lambda p: librosa.load(p, sr=None, mono=False, dtype=np.float64 if sample_type == "f64" else np.float32)
    if fmt == "wav":
        if scipy_wav:
            r["scipy"] = lambda p: scipy_wav.read(p)
        if sample_type in ("i16", "i32"):
            def read_wave(p):
                with wave.open(p, "rb") as f:
                    return np.frombuffer(f.readframes(f.getnframes()), dtype=DTYPE[sample_type]).reshape(-1, f.getnchannels())
            r["wave"] = read_wave
        elif sample_type == "i24":
            def read_wave_24(p):
                with wave.open(p, "rb") as f:
                    a = np.frombuffer(f.readframes(f.getnframes()), dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                    return ((a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)) << 8) >> 8
            r["wave"] = read_wave_24
    return r


def float_readers(fmt, sample_type):
    r = {}
    if aus:
        r["ours"] = lambda p: aus.io.read(p, as_type=aus.SampleType.F32)
    if sf:
        r["soundfile"] = lambda p: sf.read(p, dtype="float32", always_2d=True)
    if librosa:
        r["librosa"] = lambda p: librosa.load(p, sr=None, mono=False)
    if fmt == "wav" and sample_type in ("i16", "i32"):
        scale = 32768.0 if sample_type == "i16" else 2147483648.0
        if scipy_wav:
            r["scipy"] = lambda p: scipy_wav.read(p)[1].astype(np.float32) / scale
        def read_wave(p):
            with wave.open(p, "rb") as f:
                return np.frombuffer(f.readframes(f.getnframes()), dtype=DTYPE[sample_type]).astype(np.float32) / scale
        r["wave"] = read_wave
    return r


def stream_readers(fmt, sample_type):
    r = {}
    if aus:
        native = getattr(aus.SampleType, sample_type.upper())

        def stream_ours(p):
            total = 0
            with aus.io.StreamedAudioReader(p) as s:
                for block in s.frames(BLOCK_FRAMES, native):
                    total += block.samples_per_channel()
            return total
        r["ours"] = stream_ours
    if sf:
        def stream_sf(p):
            total = 0
            for block in sf.blocks(p, blocksize=BLOCK_FRAMES, dtype=SF_DTYPE[sample_type], always_2d=True):
                total += len(block)
            return total
        r["soundfile"] = stream_sf
    if fmt == "wav" and sample_type in ("i16", "i32"):
        def stream_wave(p):
            total = 0
            with wave.open(p, "rb") as f:
                channels = f.getnchannels()
                while True:
                    raw = f.readframes(BLOCK_FRAMES)
                    if not raw:
                        return total
                    total += len(np.frombuffer(raw, dtype=DTYPE[sample_type]).reshape(-1, channels))
        r["wave"] = stream_wave
    return r


def writers(directory, fmt, sample_type, channels, seconds):
    x = signal(sample_type, channels, seconds)
    sf_x = x.astype(np.int32) << 8 if sample_type == "i24" else x
    w = {}
    if aus:
        # Over the same memory as the other libraries where the sample type allows, so that none
        # has its source warmer in the cache than another.
        if sample_type == "i24":
            a = aus.io.read(input_file(directory, "wav", sample_type, channels, seconds))
        elif channels == 1:
            a = aus.AudioSamples.new_mono(x[:, 0], sample_rate=SAMPLE_RATE)
        else:
            a = aus.AudioSamples.new_multi(x.T, sample_rate=SAMPLE_RATE)
        w["ours"] = lambda p: aus.io.save(p, a)
        # The usual pipeline: audio that was read with this library and is saved again.
        loaded = aus.io.read(input_file(directory, "wav", sample_type, channels, seconds))
        w["ours, array from read"] = lambda p: aus.io.save(p, loaded)
    if sf:
        w["soundfile"] = lambda p: sf.write(p, sf_x, SAMPLE_RATE, subtype=SUBTYPE[sample_type], format=fmt.upper())
    if fmt == "wav":
        if scipy_wav and sample_type != "i24":
            w["scipy"] = lambda p: scipy_wav.write(p, SAMPLE_RATE, x)
        if sample_type in ("i16", "i32"):
            def write_wave(p):
                with wave.open(p, "wb") as f:
                    f.setnchannels(channels)
                    f.setsampwidth(WIDTH[sample_type])
                    f.setframerate(SAMPLE_RATE)
                    f.writeframes(x.tobytes())
            w["wave"] = write_wave
        elif sample_type == "i24":
            def write_wave_24(p):
                data = i24_bytes(x)  # wave takes packed bytes, so packing is part of the call
                with wave.open(p, "wb") as f:
                    f.setnchannels(channels)
                    f.setsampwidth(3)
                    f.setframerate(SAMPLE_RATE)
                    f.writeframes(data)
            w["wave"] = write_wave_24
    return w


# --- Timing --------------------------------------------------------------------------------------


def percentiles(values):
    a = np.asarray(values)
    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    return {"p25": float(p25), "p50": float(p50), "p75": float(p75), "min": float(a.min())}


def run_row(args, row, funcs):
    """Every library in turn for as many rounds as the budget allows."""
    op, cache, target = row["op"], row["cache"], row["target"]
    reading = op != "write"
    fmt = row["fmt"]
    pid = os.getpid()
    path_of = {name: os.path.join(args.dir, f"out_{pid}_{i}.{fmt}") for i, name in enumerate(funcs)}
    sources = []
    if reading:
        # Every library reads the same files; with several instances, each round takes the next one,
        # so that one file's place on the disk does not decide the row.
        sources = [input_file(args.dir, fmt, row["type"], row["ch"], row["dur"], k) for k in range(args.instances)]
        row["bytes"] = os.path.getsize(sources[0])
        row["instances"] = len(sources)
    source = sources[0] if sources else None

    def call(name):
        if reading:
            if cache == "cold":
                drop_cache(source)
            started_cpu = time.process_time_ns()
            started = time.perf_counter_ns()
            result = funcs[name](source)
            ended = time.perf_counter_ns()
            ended_cpu = time.process_time_ns()
            del result
            return (ended - started) / 1e6, (ended_cpu - started_cpu) / 1e6, None
        path = path_of[name]
        if target == "new":
            try:
                os.remove(path)
            except OSError:
                pass
        started_cpu = time.process_time_ns()
        started = time.perf_counter_ns()
        funcs[name](path)
        ended = time.perf_counter_ns()
        ended_cpu = time.process_time_ns()
        fsync_path(path)
        return (ended - started) / 1e6, (ended_cpu - started_cpu) / 1e6, (time.perf_counter_ns() - ended) / 1e6

    names = list(funcs)
    first = {n: call(n)[0] for n in names}
    slowest = max(first.values()) / 1e3
    rounds = int(max(args.min_rounds, min(args.max_rounds, args.budget / max(slowest, 1e-6))))
    wall = {n: [] for n in names}
    cpu = {n: [] for n in names}
    sync = {n: [] for n in names}
    direct = []
    buffer = None
    if cache == "cold":
        # A page-aligned buffer, which unbuffered reads need.
        import mmap

        buffer = mmap.mmap(-1, (row["bytes"] // 4096) * 4096)
    was_enabled = gc.isenabled()
    gc.disable()
    order = random.Random(0)
    for i in range(rounds):
        if sources:
            source = sources[i % len(sources)]
        turn = names[:]
        order.shuffle(turn)
        for name in turn:
            a, b, c = call(name)
            wall[name].append(a)
            cpu[name].append(b)
            if c is not None:
                sync[name].append(c)
        if buffer is not None:
            drop_cache(source)
            started = time.perf_counter_ns()
            direct_read(source, buffer)
            direct.append((time.perf_counter_ns() - started) / 1e6)
    if was_enabled:
        gc.enable()
    if buffer is not None:
        buffer.close()
    for path in path_of.values():
        try:
            os.remove(path)
        except OSError:
            pass

    rng = np.random.default_rng(1)
    row["rounds"] = rounds
    row["lib"] = {}
    ours = np.asarray(wall["ours"]) if "ours" in wall else None
    for name in names:
        entry = percentiles(wall[name])
        entry["cpu_p50"] = float(np.median(cpu[name]))
        if sync[name]:
            entry["fsync_p50"] = float(np.median(sync[name]))
        if ours is not None and name != "ours":
            ratio = ours / np.asarray(wall[name])
            boot = np.median(rng.choice(ratio, size=(2000, len(ratio))), axis=1)
            low, high = np.percentile(boot, [2.5, 97.5])
            entry.update(ratio=float(np.median(ratio)), lo=float(low), hi=float(high))
        row["lib"][name] = entry
    if direct:
        row["direct_p50"] = float(np.median(direct))
    return row


def describe(row):
    parts = []
    for name, e in row["lib"].items():
        text = f"{name} {e['p50']:.3f}"
        if "fsync_p50" in e:
            text += f"+{e['fsync_p50']:.1f}"
        if "ratio" in e:
            text += f" [{e['ratio']:.2f}]"
        parts.append(text)
    if "direct_p50" in row:
        parts.append(f"direct {row['direct_p50']:.3f}")
    return (
        f"{row['fmt']} {row['op']:<8} {row['cache'] or row['target']:<5} {row['type']} {row['ch']}ch {row['dur']:>4}s "
        + (f"block {row['block']} " if "block" in row else "")
        + ""
        f"x{row['rounds']:<4} " + "  ".join(parts)
    )


def main():
    global BLOCK_FRAMES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=None, help="directory on the disk to test (default: a new temporary one)")
    ap.add_argument("--out", default=None, help="results file (default: python_io_results_<host>_<fs>.jsonl here)")
    ap.add_argument("--ops", default="read,read-f32,stream,write")
    ap.add_argument("--fmt", default="wav,flac")
    ap.add_argument("--types", default="i16,i24,i32,f32,f64")
    ap.add_argument("--ch", default="1,2,6")
    ap.add_argument("--dur", default="1,10,60,600")
    ap.add_argument("--cold-dur", default="1,10,60", help="lengths for cold reads, which are bound by the disk")
    ap.add_argument("--caches", default="warm,cold")
    ap.add_argument("--targets", default="new,over")
    ap.add_argument("--budget", type=float, default=0.3, help="seconds per row per library, roughly")
    ap.add_argument("--min-rounds", type=int, default=11)
    ap.add_argument("--max-rounds", type=int, default=400)
    ap.add_argument("--instances", type=int, default=1, help="copies of each input file, read in turn by every library")
    ap.add_argument("--stream-blocks", default=str(BLOCK_FRAMES), help="frames per block for streamed reads, comma-separated")
    ap.add_argument("--quick", action="store_true", help="a small subset, to check that everything works")
    args = ap.parse_args()
    if args.quick:
        args.types, args.ch, args.dur, args.cold_dur, args.min_rounds, args.budget = "i16", "2", "1,10", "1", 3, 0.02

    missing = [n for n, m in (("audio_samples", aus), ("soundfile", sf)) if not m]
    if missing:
        sys.exit("needs " + " and ".join(missing) + ": see the instructions that came with this script")
    if not args.dir:
        args.dir = tempfile.mkdtemp(prefix="aus_io_bench_", dir=os.getcwd())
    os.makedirs(args.dir, exist_ok=True)

    fs = file_system_of(args.dir)
    if not args.out:
        kind = str(fs.get("type", "fs")).split()[0].strip("[]',:") or "fs"
        args.out = f"python_io_results_{platform.node()}_{kind}.jsonl"
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "op" in r:
                    done.add((r["fmt"], r["op"], r["cache"], r["target"], r["type"], r["ch"], r["dur"], r.get("threads", "default"), r.get("block", BLOCK_FRAMES)))
    threads = os.environ.get("AUDIO_SAMPLES_IO_THREADS", "default")
    with open(args.out, "a") as out:
        if not done:
            out.write(json.dumps({
                "meta": True, "date": time.strftime("%Y-%m-%d %H:%M"), "platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor(), "cpus": os.cpu_count(), "dir": args.dir, "file_system": fs, "versions": versions(), "cold_method": DROP_METHOD,
            }) + "\n")
        for op in args.ops.split(","):
            for fmt in args.fmt.split(","):
                for sample_type in args.types.split(","):
                    if fmt == "flac" and sample_type not in ("i16", "i24"):
                        continue
                    for channels in map(int, args.ch.split(",")):
                        reading = op != "write"
                        modes = [(c, "") for c in args.caches.split(",")] if reading else [("", t) for t in args.targets.split(",")]
                        for cache, target in modes:
                            lengths = args.cold_dur if cache == "cold" else args.dur
                            for seconds in map(int, lengths.split(",")):
                                blocks = map(int, args.stream_blocks.split(",")) if op == "stream" else [BLOCK_FRAMES]
                                for block in blocks:
                                    key = (fmt, op, cache, target, sample_type, channels, seconds, threads, block)
                                    if key in done:
                                        continue
                                    BLOCK_FRAMES = block
                                    if op == "read":
                                        funcs = readers(fmt, sample_type)
                                    elif op == "read-f32":
                                        funcs = float_readers(fmt, sample_type)
                                    elif op == "stream":
                                        funcs = stream_readers(fmt, sample_type)
                                    else:
                                        funcs = writers(args.dir, fmt, sample_type, channels, seconds)
                                    row = dict(fmt=fmt, op=op, cache=cache, target=target, type=sample_type, ch=channels, dur=seconds, threads=threads)
                                    if block != 4096:
                                        row["block"] = block
                                    run_row(args, row, funcs)
                                    out.write(json.dumps(row) + "\n")
                                    out.flush()
                                    print(describe(row), flush=True)
    print(f"\nResults: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
