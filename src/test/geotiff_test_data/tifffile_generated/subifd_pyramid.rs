//! Pyramids whose reduced resolutions are SubIFDs (tag 330), not top-level IFDs.

use std::ops::Range;
use std::path::PathBuf;

use async_trait::async_trait;
use bytes::Bytes;

use crate::error::AsyncTiffResult;
use crate::metadata::{MetadataFetch, TiffMetadataReader};
use crate::tags::Tag;
use crate::test::util::open_tiff;
use crate::TagValue;

const SUBIFDS: Tag = Tag::Unknown(330);

/// Tag 330 holds one offset per reduced level, as plain integers.
fn subifd_offsets(value: &TagValue) -> Vec<u64> {
    let as_offset = |v: &TagValue| match v {
        TagValue::Ifd(offset) => *offset as u64,
        TagValue::IfdBig(offset) => *offset,
        other => panic!("not an IFD offset: {other:?}"),
    };
    match value {
        TagValue::List(values) => values.iter().map(as_offset).collect(),
        single => vec![as_offset(single)],
    }
}

async fn assert_pyramid(filename: &str) {
    let (reader, tiff) = open_tiff(filename).await;

    // The reduced levels are not in the top-level chain.
    assert_eq!(tiff.ifds().len(), 1);
    let full = &tiff.ifds()[0];
    assert_eq!(full.image_height(), 256);
    assert_eq!(full.image_width(), 256);
    assert_eq!(full.tile_height(), Some(64));

    let offsets = subifd_offsets(&full.other_tags()[&SUBIFDS]);
    assert_eq!(offsets.len(), 2);

    // The offsets point at real IFDs: one per level, each half the size of the previous level.
    let metadata_reader = TiffMetadataReader::try_open(&reader).await.unwrap();
    for (offset, expected_size) in offsets.into_iter().zip([128, 64]) {
        let level = metadata_reader.read_ifd_at(&reader, offset).await.unwrap();
        assert_eq!(level.image_height(), expected_size);
        assert_eq!(level.image_width(), expected_size);

        let tile = level.fetch_tile(0, 0, &reader).await.unwrap();
        let array = tile.decode(&Default::default()).unwrap();
        assert_eq!(array.shape, [64, 64, 1]);
    }
}

/// In a classic TIFF the tag is typed IFD.
#[tokio::test]
async fn test_classic() {
    assert_pyramid("geotiff-test-data/tifffile_generated/fixtures/subifd_pyramid_classic.tif")
        .await;
}

/// In a BigTIFF the tag is typed IFD8.
#[tokio::test]
async fn test_bigtiff() {
    assert_pyramid("geotiff-test-data/tifffile_generated/fixtures/subifd_pyramid_bigtiff.tif")
        .await;
}

/// A file held in memory, which answers a read past its end with the bytes it has, as a store does
#[derive(Debug)]
struct InMemory(Bytes);

#[async_trait]
impl MetadataFetch for InMemory {
    async fn fetch(&self, range: Range<u64>) -> AsyncTiffResult<Bytes> {
        let len = self.0.len() as u64;
        Ok(self
            .0
            .slice(range.start.min(len) as usize..range.end.min(len) as usize))
    }
}

/// A SubIFD offset comes from the caller, so a wrong one must be refused, not crash the process.
async fn assert_every_offset_is_read_or_refused(filename: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(filename);
    let file = InMemory(std::fs::read(path).unwrap().into());
    let metadata_reader = TiffMetadataReader::try_open(&file).await.unwrap();

    let offsets = (0..file.0.len() as u64).chain([u64::MAX - 8, u64::MAX]);
    let read =
        futures::future::join_all(offsets.map(|offset| metadata_reader.read_ifd_at(&file, offset)))
            .await;
    assert!(read.iter().any(Result::is_ok));
}

#[tokio::test]
async fn test_classic_offsets_are_read_or_refused() {
    assert_every_offset_is_read_or_refused(
        "geotiff-test-data/tifffile_generated/fixtures/subifd_pyramid_classic.tif",
    )
    .await;
}

#[tokio::test]
async fn test_bigtiff_offsets_are_read_or_refused() {
    assert_every_offset_is_read_or_refused(
        "geotiff-test-data/tifffile_generated/fixtures/subifd_pyramid_bigtiff.tif",
    )
    .await;
}
