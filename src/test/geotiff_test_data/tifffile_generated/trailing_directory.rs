//! A file whose first IFD is at the end, as streaming writers produce.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::local::LocalFileSystem;

use crate::error::AsyncTiffResult;
use crate::metadata::cache::ReadaheadMetadataCache;
use crate::metadata::{MetadataFetch, TiffMetadataReader};
use crate::reader::{AsyncFileReader, ObjectReader};

const FILENAME: &str = "geotiff-test-data/tifffile_generated/fixtures/trailing_directory.tif";

#[tokio::test]
async fn test_trailing_directory() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let store = Arc::new(LocalFileSystem::new_with_prefix(&manifest_dir).unwrap());
    let path = format!("fixtures/{FILENAME}");
    let reader =
        Arc::new(ObjectReader::new(store, path.as_str().into())) as Arc<dyn AsyncFileReader>;

    // The directory sits past the default 32 KiB first fetch, near enough to it that the cache
    // fetches the gap in between.
    let cache = ReadaheadMetadataCache::new(reader.clone());
    let mut metadata_reader = TiffMetadataReader::try_open(&cache).await.unwrap();
    let tiff = metadata_reader.read(&cache).await.unwrap();

    assert_eq!(tiff.ifds().len(), 1);
    let ifd = &tiff.ifds()[0];
    assert_eq!(ifd.image_height(), 256);
    assert_eq!(ifd.image_width(), 256);
    assert_eq!(ifd.tile_height(), Some(64));

    let tile = ifd.fetch_tile(0, 0, &reader).await.unwrap();
    let array = tile.decode(&Default::default()).unwrap();
    assert_eq!(array.shape, [64, 64, 1]);
}

/// Records every range that reaches the reader beneath the cache
#[derive(Debug)]
struct Recording {
    inner: Arc<dyn AsyncFileReader>,
    fetched: std::sync::Mutex<Vec<Range<u64>>>,
}

#[async_trait]
impl MetadataFetch for Recording {
    async fn fetch(&self, range: Range<u64>) -> AsyncTiffResult<Bytes> {
        self.fetched.lock().unwrap().push(range.clone());
        self.inner.fetch(range).await
    }
}

#[tokio::test]
async fn test_trailing_directory_is_fetched_where_it_is() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let store = Arc::new(LocalFileSystem::new_with_prefix(&manifest_dir).unwrap());
    let path = format!("fixtures/{FILENAME}");
    let recording = Recording {
        inner: Arc::new(ObjectReader::new(store, path.as_str().into())),
        fetched: Default::default(),
    };

    // With a readahead smaller than the 20 KB between the first fetch and the directory, the
    // directory is far, so the bytes before it are not fetched.
    let cache = ReadaheadMetadataCache::new(recording).with_readahead(4096);
    let mut metadata_reader = TiffMetadataReader::try_open(&cache).await.unwrap();
    metadata_reader.read(&cache).await.unwrap();

    let directory = 53_374;
    assert_eq!(
        *cache.inner().fetched.lock().unwrap(),
        vec![0..32_768, directory..directory + 32_768]
    );
}
