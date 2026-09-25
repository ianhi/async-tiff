"""Score the installed async-tiff's metadata cache: requests and bytes to open each snapshot.

Each snapshot in bench/snapshots/ holds one file's metadata bytes (see capture.py). A store
replays them, answering reads outside them with zeros, which the parser never looks at, and
counts every request TIFF.open makes. The loss for a scenario is the estimated time to open:

    loss = requests × RTT + bytes / bandwidth

    uv run python bench/score.py                      # score the installed build
    uv run python bench/score.py --save window        # also write bench/results/window.json
    uv run python bench/score.py --baseline window    # and compare against a saved run
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from async_tiff import TIFF

# (name, round-trip seconds, bytes per second). The laptop round trip is a measured median
# request. From m7g.large VMs in us-west-2 and us-east-1, compare_live.py measured median small
# requests of 20-30 ms, so in-region A is close to measured and in-region B is optimistic.
SCENARIOS = [
    ("laptop -> S3", 0.111, 12e6),
    ("in-region A", 0.020, 100e6),
    ("in-region B", 0.005, 200e6),
]
# A request this large means the cache is streaming the file; refuse it rather than allocate it.
REFUSE_OVER = 256 * 1024 * 1024

HERE = Path(__file__).parent


class Refused(Exception):
    pass


class ReplayStore:
    def __init__(self, snapshot):
        self.size = int(snapshot["size"])
        self.starts, self.ends = snapshot["starts"], snapshot["ends"]
        self.data = snapshot["data"].tobytes()
        self.offsets = np.concatenate([[0], np.cumsum(self.ends - self.starts)])
        self.requests, self.transferred = 0, 0

    async def get_range_async(self, path, *, start, end=None, length=None):
        end = min(start + length if end is None else end, self.size)
        self.requests += 1
        self.transferred += max(0, end - start)
        if end - start > REFUSE_OVER:
            raise Refused(f"{end - start:,} byte request")
        out = bytearray(max(0, end - start))
        first = np.searchsorted(self.ends, start, side="right")
        for i in range(first, len(self.starts)):
            s, e = int(self.starts[i]), int(self.ends[i])
            if s >= end:
                break
            lo, hi = max(s, start), min(e, end)
            src = int(self.offsets[i]) + lo - s
            out[lo - start : hi - start] = self.data[src : src + hi - lo]
        return bytes(out)

    async def get_ranges_async(self, path, *, starts, ends=None, lengths=None):
        raise NotImplementedError("only tile reads use this")


async def score(name):
    snapshot = np.load(HERE / "snapshots" / f"{name}.npz")
    store = ReplayStore(snapshot)
    try:
        await TIFF.open("x.tif", store=store)
        note = ""
    except Refused as e:
        note = f"refused a {e}"
    except Exception as e:
        # The requests before a failure still show what the cache fetched.
        note = f"failed after reading: {type(e).__name__}: {str(e)[:100]}"
    metadata = int((snapshot["ends"] - snapshot["starts"]).sum())
    return {
        "requests": store.requests,
        "bytes": store.transferred,
        "metadata": metadata,
        "size": store.size,
        "note": note,
    }


def loss(r, rtt, bandwidth):
    return r["requests"] * rtt + r["bytes"] / bandwidth


def oracle(name, rtt, bandwidth):
    """The least loss any cache could reach: one request per run of metadata ranges whose gaps
    cost less to download than a round trip. It knows the layout up front, so it is a bound."""
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
    return loss({"requests": requests, "bytes": transferred}, rtt, bandwidth)


parser = argparse.ArgumentParser()
parser.add_argument("--save")
parser.add_argument("--baseline")
args = parser.parse_args()

names = sorted(p.stem for p in (HERE / "snapshots").glob("*.npz"))
results = {n: asyncio.run(score(n)) for n in names}
base = (
    json.loads((HERE / "results" / f"{args.baseline}.json").read_text())
    if args.baseline
    else {}
)

print(f"{'file':<24}{'requests':>9}{'bytes':>15}{'metadata':>12}{'overfetch':>10}")
for n, r in results.items():
    over = r["bytes"] / r["metadata"] if r["metadata"] else float("nan")
    print(
        f"{n:<24}{r['requests']:>9,}{r['bytes']:>15,}{r['metadata']:>12,}{over:>9.1f}x  {r['note']}"
    )

print(f"\nloss = estimated seconds to open{'  (baseline -> now)' if base else ''}")
print(f"{'file':<24}" + "".join(f"{s:>22}" for s, *_ in SCENARIOS))
for n, r in list(results.items()) + [("TOTAL", None)]:
    cells = []
    for _, rtt, bw in SCENARIOS:
        now = (
            sum(loss(x, rtt, bw) for x in results.values())
            if r is None
            else loss(r, rtt, bw)
        )
        if base:
            was = (
                sum(loss(x, rtt, bw) for x in base.values())
                if r is None
                else loss(base[n], rtt, bw)
                if n in base
                else float("nan")
            )
            cells.append(f"{was:9.3f} -> {now:8.3f}")
        else:
            cells.append(f"{now:22.3f}")
    print(f"{n:<24}" + "".join(f"{c:>22}" for c in cells))

print("\nloss / oracle  (1.0 is the best any cache can do; worst file is the headline)")
print(f"{'file':<24}" + "".join(f"{s:>22}" for s, *_ in SCENARIOS))
worst = [
    (0, ""),
] * len(SCENARIOS)
for n, r in results.items():
    cells = []
    for i, (_, rtt, bw) in enumerate(SCENARIOS):
        best = oracle(n, rtt, bw)
        now = loss(r, rtt, bw) / best
        worst[i] = max(worst[i], (now, n))
        if base and n in base:
            cells.append(f"{loss(base[n], rtt, bw) / best:9.2f} -> {now:8.2f}")
        else:
            cells.append(f"{now:22.2f}")
    print(f"{n:<24}" + "".join(f"{c:>22}" for c in cells))
print(f"{'WORST (now)':<24}" + "".join(f"{f'{w:.2f} {n}':>22}" for w, n in worst))

if args.save:
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.save}.json").write_text(json.dumps(results, indent=1))
