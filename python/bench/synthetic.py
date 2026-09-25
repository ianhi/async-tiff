"""Write the synthetic layouts the metadata-cache benchmark scores, into bench/synthetic/.

- pixels_then_ifd: a 4 MB image, then a second IFD after its pixels (a streaming writer).
- far_tile_tables: 16 MB of pixels, then an IFD with 65,536 tiles, so 655 KB of tile tables
  far from the start of the file.

    uv run --with tifffile python bench/synthetic.py
"""

from pathlib import Path

import numpy as np
import tifffile

out = Path(__file__).parent / "synthetic"
out.mkdir(exist_ok=True)

with tifffile.TiffWriter(out / "pixels_then_ifd.tif") as tw:
    tw.write(np.zeros((2048, 2048), np.uint8))
    tw.write(np.zeros((16, 16), np.uint8))

with tifffile.TiffWriter(out / "far_tile_tables.tif", bigtiff=True) as tw:
    tw.write(np.zeros((4096, 4096), np.uint8), tile=(256, 256))
    tw.write(
        np.zeros((4096, 4096), np.uint8),
        tile=(16, 16),
        compression="zlib",
        subfiletype=1,
    )

for path in sorted(out.glob("*.tif")):
    print(f"{path.name}: {path.stat().st_size:,} bytes")
