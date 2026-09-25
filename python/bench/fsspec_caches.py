"""Replay async-tiff's metadata reads through fsspec's own cache classes.

bench/traces/<name>.npz holds every read the parser makes while opening a snapshot, in order,
before any cache sees it. Each fsspec cache here is the real class from fsspec.caching, given a
fetcher that counts requests, so the numbers are what fsspec would send for the same reads.

The traces were recorded by printing `range` at the top of `ReadaheadMetadataCache::fetch` and
opening each snapshot through score.py's replay store. They depend only on the parser, not the
cache, so they stay valid until the parser's read pattern changes.

    uv run --with fsspec --with aiohttp python bench/fsspec_caches.py
"""

from pathlib import Path

import fsspec
import numpy as np
from fsspec.caching import BlockCache, BytesCache, ReadAheadCache
from fsspec.implementations.http import DEFAULT_BLOCK_SIZE as HTTP_BLOCK_SIZE

HERE = Path(__file__).parent
SCENARIOS = [
    ("laptop -> S3", 0.111, 12e6),
    ("in-region A", 0.020, 100e6),
    ("in-region B", 0.005, 200e6),
]
S3FS_BLOCK_SIZE = 50 * 2**20

CACHES = [
    # (label, cache class, block size)
    (
        f"s3fs default: readahead, {S3FS_BLOCK_SIZE >> 20} MiB",
        ReadAheadCache,
        S3FS_BLOCK_SIZE,
    ),
    (f"HTTP default: bytes, {HTTP_BLOCK_SIZE >> 20} MiB", BytesCache, HTTP_BLOCK_SIZE),
    ("readahead, 32 KiB", ReadAheadCache, 32 * 1024),
    ("bytes, 1 MiB", BytesCache, 2**20),
    ("blockcache, 1 MiB blocks", BlockCache, 2**20),
]


def replay(name, cache_class, blocksize):
    size = int(np.load(HERE / "snapshots" / f"{name}.npz")["size"])
    trace = np.load(HERE / "traces" / f"{name}.npz")
    count = {"requests": 0, "bytes": 0}

    def fetcher(start, end):
        end = min(end, size)
        count["requests"] += 1
        count["bytes"] += max(0, end - start)
        return bytes(max(0, end - start))

    cache = cache_class(blocksize, fetcher, size)
    for start, end in zip(trace["starts"].tolist(), trace["ends"].tolist()):
        cache._fetch(start, end)
    return count


def oracle(name, rtt, bandwidth):
    """As in score.py: one request per run of metadata ranges whose gaps cost less than a round trip."""
    snapshot = np.load(HERE / "snapshots" / f"{name}.npz")
    starts, ends = snapshot["starts"], snapshot["ends"]
    requests, transferred = 1, int(ends[0] - starts[0])
    for prev_end, s, e in zip(ends[:-1], starts[1:], ends[1:]):
        gap = int(s - prev_end)
        if gap / bandwidth < rtt:
            transferred += gap
        else:
            requests += 1
        transferred += int(e - s)
    return requests * rtt + transferred / bandwidth


names = sorted(p.stem for p in (HERE / "traces").glob("*.npz"))
print(f"fsspec {fsspec.__version__}")
for label, cache_class, blocksize in CACHES:
    print(f"\n{label}")
    print(
        f"  {'file':<24}{'requests':>9}{'bytes':>15}"
        + "".join(f"{s:>16}" for s, *_ in SCENARIOS)
    )
    for n in names:
        r = replay(n, cache_class, blocksize)
        ratios = "".join(
            f"{(r['requests'] * rtt + r['bytes'] / bw) / oracle(n, rtt, bw):>15.2f}x"
            for _, rtt, bw in SCENARIOS
        )
        print(f"  {n:<24}{r['requests']:>9,}{r['bytes']:>15,}{ratios}")
