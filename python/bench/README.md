# Metadata cache benchmark

Measures how many requests and bytes `TIFF.open` costs on real files, offline, using the
installed build's cache.

| Script | What it does |
|---|---|
| `capture.py` | Saves a TIFF's metadata bytes (found with tifffile) to `snapshots/<name>.npz` |
| `synthetic.py` | Writes the two synthetic layouts into `synthetic/` |
| `score.py` | Replays every snapshot through `TIFF.open` and prints requests, bytes and estimated open time per network scenario |
| `fsspec_caches.py` | Replays the parser's recorded reads (`traces/`) through fsspec's cache classes, for comparison |
| `compare_live.py` | Opens the real files from S3 with 0.7.2, #328 and this branch side by side |

The loss is `requests × RTT + bytes / bandwidth`. `score.py` also prints it as a ratio to an
oracle: the fewest requests and bytes that cover every metadata range, given the layout in
advance. No cache can beat it, and 1.0 is the best possible.

```bash
uv run --reinstall-package async-tiff python bench/score.py --baseline pr328_window
```

## Snapshots

| Name | File | Metadata |
|---|---|---|
| `camelyon_slide` | `s3://camelyon-dataset/CAMELYON16/images/tumor_001.tif`, 3.26 GB, 11 IFDs each after its level's tiles | 2.76 MB in 12 ranges |
| `camelyon_mask` | `s3://camelyon-dataset/CAMELYON16/masks/tumor_001_mask.tif`, 35 MB, 9 IFDs | 1.77 MB |
| `jump_image` | A JUMP Cell Painting image (`cellpainting-gallery`), 2.56 MB, one IFD at 96.7% | 84 KB, mostly an 83 KB private-tag XML |
| `cog_s2_b04`, `cog_s2_tci` | Sentinel-2 COGs, 199 MB and 103 MB | about 3.5 KB each, at the front |
| `synth_pixels_then_ifd` | 4.2 MB, second IFD after the pixels | 456 B |
| `synth_far_tile_tables` | 18.2 MB, far IFD with 655 KB of tile tables | 659 KB |

Each snapshot's source is in its `url` field.
