# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy",
#   "obstore",
#   "obspec-utils",
#   "async-tiff-v072",
#   "async-tiff-pr328",
#   "async-tiff-new",
# ]
# ///
"""Open each snapshot's source file from S3 with three async-tiff builds, interleaved.

Every run opens the file once with each build in turn, so all three see the same S3 conditions.
The builds are installed side by side under different names with third-wheel:

- async_tiff_v072: the 0.7.2 release from PyPI
- async_tiff_pr328: PR #328's head, a854428
- async_tiff_new: this branch

Put the three renamed wheels in bench/wheels/ first. For the two local builds, from a checkout
of each commit's python/ directory:

    maturin build --release -i python3.13 -o raw/
    third-wheel rename raw/async_tiff-*.whl async_tiff_new -o bench/wheels/

and for the release, `pip download async-tiff==0.7.2 --no-deps` and the same rename. Add
`--zig --target aarch64-unknown-linux-gnu` to maturin (and `--platform manylinux_2_28_aarch64`
to pip) for a Linux arm64 VM. Then, from python/:

    uv run --find-links bench/wheels bench/compare_live.py
    uv run --find-links bench/wheels bench/compare_live.py --runs 5 --save laptop
"""

import argparse
import asyncio
import json
import platform
import statistics
import time
from pathlib import Path

import async_tiff_new
import async_tiff_pr328
import async_tiff_v072
import numpy as np
from obspec_utils.wrappers import RequestTrace, TracingReadableStore
from obstore.store import S3Store

HERE = Path(__file__).parent
REGIONS = {
    "camelyon-dataset": "us-west-2",
    "sentinel-cogs": "us-west-2",
    "cellpainting-gallery": "us-east-1",
}
BUILDS = {"0.7.2": async_tiff_v072, "#328": async_tiff_pr328, "new": async_tiff_new}
# 0.7.2 downloads the whole 3.26 GB slide to open it.
WHOLE_FILE = {("0.7.2", "camelyon_slide")}


async def open_once(module, store, path):
    trace = RequestTrace()
    t0 = time.perf_counter()
    try:
        await module.TIFF.open(path, store=TracingReadableStore(store, trace))
        note = ""
    except Exception as e:
        # The requests before a failure still show what the cache fetched.
        note = f"failed after reading: {type(e).__name__}"
    return time.perf_counter() - t0, trace.requests, note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--whole-file",
        action="store_true",
        help="also open files that a build downloads in full",
    )
    parser.add_argument("--save", help="write bench/results/live_<SAVE>.json")
    args = parser.parse_args()

    names = args.names or sorted(p.stem for p in (HERE / "snapshots").glob("*.npz"))
    print(f"{platform.node()}, {args.runs} runs per build, interleaved")
    print(
        f"{'file':<16}{'build':>7}{'open (s)':>10}{'runs (s)':>24}{'requests':>10}"
        f"{'bytes':>15}{'median req (ms)':>17}"
    )
    results = {}
    for name in names:
        snapshot = np.load(HERE / "snapshots" / f"{name}.npz")
        if not str(snapshot.get("url", "")).startswith("s3://"):
            continue
        bucket, path = str(snapshot["url"]).removeprefix("s3://").split("/", 1)
        store = S3Store(bucket, region=REGIONS[bucket], skip_signature=True)
        size = int(snapshot["size"])
        builds = {
            b: m
            for b, m in BUILDS.items()
            if args.whole_file or (b, name) not in WHOLE_FILE
        }

        runs = {b: [] for b in builds}
        for _ in range(args.runs):
            for build, module in builds.items():
                runs[build].append(asyncio.run(open_once(module, store, path)))

        for build, build_runs in runs.items():
            times = [t for t, _, _ in build_runs]
            _, requests, note = build_runs[-1]
            transferred = sum(min(r.end, size) - r.start for r in requests)
            median_request = statistics.median(r.duration for r in requests) * 1000
            results.setdefault(name, {})[build] = {
                "times": times,
                "requests": len(requests),
                "bytes": transferred,
                "median_request_ms": median_request,
                "note": note,
            }
            print(
                f"{name:<16}{build:>7}{statistics.median(times):>10.2f}"
                f"{' '.join(f'{t:.2f}' for t in times):>24}{len(requests):>10}"
                f"{transferred:>15,}{median_request:>17.0f}  {note}"
            )

    if args.save:
        out = HERE / "results" / f"live_{args.save}.json"
        out.write_text(
            json.dumps({"host": platform.node(), "results": results}, indent=1)
        )


main()
