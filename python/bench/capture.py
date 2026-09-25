"""Save the metadata bytes of a remote TIFF, so metadata reads can be replayed offline.

tifffile, not async-tiff, finds the metadata: the header, each IFD's entries and next
pointer, and every tag value stored outside its entry. Only those ranges are fetched and
saved, a few MB even for a multi-GB slide.

    uv run --with tifffile python bench/capture.py s3://bucket/key name [--region us-west-2]
    uv run --with tifffile python bench/capture.py path/to/local.tif name
"""

import argparse
from pathlib import Path

import numpy as np
import obstore
import tifffile
from obstore.store import LocalStore, S3Store

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("name")
parser.add_argument("--region", default="us-west-2")
args = parser.parse_args()

if args.url.startswith("s3://"):
    bucket, key = args.url.removeprefix("s3://").split("/", 1)
    store = S3Store(bucket, region=args.region, skip_signature=True)
else:
    store, key = LocalStore(), str(Path(args.url).resolve()).lstrip("/")
size = obstore.head(store, key)["size"]


class PlainBytesReader:
    """obstore's reader, returning `bytes`, which tifffile needs, instead of obstore's Bytes."""

    def __init__(self, reader):
        self.reader = reader

    def read(self, size=-1):
        return bytes(
            self.reader.read() if size is None or size < 0 else self.reader.read(size)
        )

    def seek(self, offset, whence=0):
        return self.reader.seek(offset, whence)

    def tell(self):
        return self.reader.tell()


f = obstore.open_reader(store, key, buffer_size=64 * 1024)
with tifffile.TiffFile(PlainBytesReader(f)) as tf:
    big = tf.is_bigtiff
    count_size, entry_size, word = (8, 20, 8) if big else (2, 12, 4)
    ranges = [(0, 16 if big else 8)]
    for page in tf.pages:
        n = len(page.tags)
        ranges.append((page.offset, page.offset + count_size + n * entry_size + word))
        for tag in page.tags.values():
            if tag.valuebytecount > word:
                ranges.append((tag.valueoffset, tag.valueoffset + tag.valuebytecount))

merged = []
for start, end in sorted(ranges):
    if merged and start <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

f.close()
blobs = obstore.get_ranges(
    store, key, starts=[s for s, _ in merged], ends=[e for _, e in merged]
)
out = Path(__file__).parent / "snapshots" / f"{args.name}.npz"
out.parent.mkdir(exist_ok=True)
np.savez_compressed(
    out,
    url=args.url,
    size=size,
    starts=np.array([s for s, _ in merged], dtype=np.uint64),
    ends=np.array([e for _, e in merged], dtype=np.uint64),
    data=np.frombuffer(b"".join(bytes(b) for b in blobs), dtype=np.uint8),
)
total = sum(e - s for s, e in merged)
print(
    f"{args.name}: {size:,} byte file, {len(tf.pages)} IFDs, "
    f"{total:,} metadata bytes in {len(merged)} ranges -> {out}"
)
