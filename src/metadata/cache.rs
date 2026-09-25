//! Caching strategies for metadata fetching.

use std::collections::BTreeMap;
use std::ops::Range;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use tokio::sync::Mutex;

use crate::error::AsyncTiffResult;
use crate::metadata::MetadataFetch;

/// Byte ranges of a file held in memory, anywhere in the file.
///
/// # Invariant
/// - Blocks do not overlap and none is empty. Adjacent blocks are allowed.
#[derive(Debug, Default)]
struct BlockCache {
    /// Each block's bytes, keyed by the offset where it starts
    blocks: BTreeMap<u64, Bytes>,
}

impl BlockCache {
    /// The block that holds the byte at `offset`, as its start and bytes
    fn block_holding(&self, offset: u64) -> Option<(u64, &Bytes)> {
        let (&start, bytes) = self.blocks.range(..=offset).next_back()?;
        (offset < start + bytes.len() as u64).then_some((start, bytes))
    }

    /// The first byte of `range` that is not held, if any
    fn first_missing(&self, range: &Range<u64>) -> Option<u64> {
        let mut offset = range.start;
        while offset < range.end {
            let Some((start, bytes)) = self.block_holding(offset) else {
                return Some(offset);
            };
            offset = start + bytes.len() as u64;
        }
        None
    }

    /// Where the held bytes before `offset` end, when `offset` itself is not held
    fn held_end_before(&self, offset: u64) -> Option<u64> {
        let (&start, bytes) = self.blocks.range(..offset).next_back()?;
        Some(start + bytes.len() as u64)
    }

    /// Where the first block after `offset` starts
    fn next_start_after(&self, offset: u64) -> Option<u64> {
        self.blocks
            .range(offset + 1..)
            .next()
            .map(|(&start, _)| start)
    }

    fn insert(&mut self, start: u64, bytes: Bytes) {
        if !bytes.is_empty() {
            self.blocks.insert(start, bytes);
        }
    }

    /// Slice out `range`, which must be held
    fn slice(&self, range: Range<u64>) -> Bytes {
        let mut pieces = vec![];
        let mut offset = range.start;
        while offset < range.end {
            let (start, bytes) = self
                .block_holding(offset)
                .expect("slice is only called on held ranges");
            let block_end = start + bytes.len() as u64;
            let piece_end = range.end.min(block_end);
            pieces.push(bytes.slice((offset - start) as usize..(piece_end - start) as usize));
            offset = piece_end;
        }

        match pieces.len() {
            0 => Bytes::new(),
            1 => pieces.pop().unwrap(),
            _ => {
                let mut out = BytesMut::with_capacity((range.end - range.start) as usize);
                for piece in pieces {
                    out.extend_from_slice(&piece);
                }
                out.into()
            }
        }
    }
}

/// A MetadataFetch implementation that holds every range it fetches, wherever it is in the file.
///
/// TIFF metadata is not always at the front of the file. Writers that stream image data put each
/// directory after its pixels, so a file's metadata can be a few scattered regions, each far from
/// the last. On a read of bytes it does not hold, the cache fetches from one of two places:
///
/// - **Near held bytes.** If the read starts within `readahead` bytes after bytes already held,
///   the cache fetches the gap between them and `readahead` bytes past the read. The parser is
///   walking one region of metadata, such as a directory, its tag values and the next directory,
///   so the gap is likely metadata too.
/// - **Far from held bytes.** Otherwise the cache fetches `initial` bytes starting at the read,
///   which covers a whole directory in one request without assuming anything about what surrounds
///   it.
///
/// A fetch never includes bytes already held, so no byte is fetched twice.
///
/// Both thresholds weigh bytes against round trips. Downloading `readahead` extra bytes takes
/// about as long as one more request, so filling a smaller gap is cheaper than a second request,
/// and reading that far ahead costs at most a request's worth of time when the bytes go unused.
/// This is the bandwidth-delay product that Apache Arrow's read coalescing uses as its gap limit.
/// The default of 1 MiB matches object_store's range coalescing default.
#[derive(Debug)]
pub struct ReadaheadMetadataCache<F: MetadataFetch> {
    inner: F,
    cache: Mutex<BlockCache>,
    initial: u64,
    readahead: u64,
}

impl<F: MetadataFetch> ReadaheadMetadataCache<F> {
    /// Create a new ReadaheadMetadataCache wrapping the given MetadataFetch
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            cache: Mutex::new(BlockCache::default()),
            initial: 32 * 1024,
            readahead: 1024 * 1024,
        }
    }

    /// Access the inner MetadataFetch
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// Set the size of a fetch far from any held bytes, otherwise defaults to 32 KiB.
    ///
    /// The first fetch of a file is this size, from byte 0.
    pub fn with_initial_size(mut self, initial: u64) -> Self {
        self.initial = initial;
        self
    }

    /// Set how far past held bytes a read counts as near, and how far a fetch near held bytes
    /// reads ahead, otherwise defaults to 1 MiB.
    pub fn with_readahead(mut self, readahead: u64) -> Self {
        self.readahead = readahead;
        self
    }

    /// The range to fetch for a read whose first missing byte is `missing` and which ends at `end`
    fn next_fetch(&self, cache: &BlockCache, missing: u64, end: u64) -> Range<u64> {
        let (start, ahead) = match cache.held_end_before(missing) {
            Some(held_end) if missing - held_end <= self.readahead => (held_end, self.readahead),
            _ => (missing, self.initial),
        };
        let mut fetch_end = end.max(missing + ahead);
        if let Some(next) = cache.next_start_after(missing) {
            fetch_end = fetch_end.min(next);
        }
        start..fetch_end
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<F: MetadataFetch + Send + Sync> MetadataFetch for ReadaheadMetadataCache<F> {
    async fn fetch(&self, range: Range<u64>) -> AsyncTiffResult<Bytes> {
        // Holding the async mutex across the fetch keeps two concurrent reads of the same region
        // from both fetching it.
        let mut cache = self.cache.lock().await;

        while let Some(missing) = cache.first_missing(&range) {
            let fetch_range = self.next_fetch(&cache, missing, range.end);
            let bytes = self.inner.fetch(fetch_range.clone()).await?;

            // A store clamps a range that runs past the end of the file, so a short fetch is
            // expected there. One that stops before `missing` means the read itself runs past the
            // end, which is for the inner fetch to answer.
            if fetch_range.start + (bytes.len() as u64) <= missing {
                drop(cache);
                return self.inner.fetch(range).await;
            }
            cache.insert(fetch_range.start, bytes);
        }

        Ok(cache.slice(range))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use super::*;

    #[derive(Debug)]
    struct TestFetch {
        data: Bytes,
        /// Every range that reaches the raw Fetch implementation, in order
        fetched: Arc<std::sync::Mutex<Vec<Range<u64>>>>,
    }

    impl TestFetch {
        fn new(data: Bytes) -> Self {
            Self {
                data,
                fetched: Default::default(),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl MetadataFetch for TestFetch {
        async fn fetch(&self, range: Range<u64>) -> crate::error::AsyncTiffResult<Bytes> {
            self.fetched.lock().unwrap().push(range.clone());
            let len = self.data.len() as u64;
            let start = range.start.min(len) as usize;
            let end = range.end.min(len) as usize;
            Ok(self.data.slice(start..end))
        }
    }

    /// A file of `len` bytes whose byte at offset `i` is `i % 251`, so any slice identifies where
    /// it came from
    fn numbered(len: usize) -> Bytes {
        (0..len).map(|i| (i % 251) as u8).collect::<Vec<_>>().into()
    }

    fn cache(data: &Bytes) -> ReadaheadMetadataCache<TestFetch> {
        ReadaheadMetadataCache::new(TestFetch::new(data.clone()))
            .with_initial_size(64)
            .with_readahead(256)
    }

    fn fetched(cache: &ReadaheadMetadataCache<TestFetch>) -> Vec<Range<u64>> {
        cache.inner.fetched.lock().unwrap().clone()
    }

    #[tokio::test]
    async fn test_reads_within_the_first_fetch_make_one_request() {
        // A COG keeps its header and directories at the front, all inside the first fetch.
        let data = numbered(10_000);
        let cache = cache(&data);

        for range in [0..2, 2..4, 4..8, 8..20, 20..64, 30..40] {
            assert_eq!(
                cache.fetch(range.clone()).await.unwrap(),
                data.slice(range.start as usize..range.end as usize)
            );
        }
        assert_eq!(fetched(&cache), vec![0..64]);
    }

    #[tokio::test]
    async fn test_a_far_read_is_fetched_where_it_is() {
        // A directory written after 9 KB of pixels.
        let data = numbered(10_000);
        let cache = cache(&data);

        cache.fetch(0..8).await.unwrap();
        assert_eq!(
            cache.fetch(9_000..9_012).await.unwrap(),
            data.slice(9_000..9_012)
        );

        assert_eq!(fetched(&cache), vec![0..64, 9_000..9_064]);
    }

    #[tokio::test]
    async fn test_a_near_read_fills_the_gap_and_reads_ahead() {
        // A directory's entries, then a tag value 100 bytes past them.
        let data = numbered(10_000);
        let cache = cache(&data);

        cache.fetch(0..8).await.unwrap();
        cache.fetch(9_000..9_012).await.unwrap();
        assert_eq!(
            cache.fetch(9_164..9_172).await.unwrap(),
            data.slice(9_164..9_172)
        );

        // The gap from the held end at 9_064 is fetched with the read, plus `readahead` past it.
        assert_eq!(fetched(&cache), vec![0..64, 9_000..9_064, 9_064..9_420]);

        // So the next values along come from memory.
        cache.fetch(9_172..9_400).await.unwrap();
        assert_eq!(fetched(&cache).len(), 3);
    }

    #[tokio::test]
    async fn test_held_bytes_are_never_fetched_again() {
        let data = numbered(10_000);
        let cache = cache(&data);

        cache.fetch(0..8).await.unwrap();
        cache.fetch(5_000..5_008).await.unwrap();
        cache.fetch(9_000..9_008).await.unwrap();
        // Values stored before their directory: a read that starts just before a held block.
        cache.fetch(4_990..5_004).await.unwrap();
        // One read spanning two held blocks and the holes around them.
        let wide = 4_900..9_100;
        assert_eq!(
            cache.fetch(wide.clone()).await.unwrap(),
            data.slice(4_900..9_100)
        );

        let mut ranges = fetched(&cache);
        ranges.sort_by_key(|r| r.start);
        for pair in ranges.windows(2) {
            assert!(pair[0].end <= pair[1].start, "fetched {:?} twice", pair);
        }
    }

    #[tokio::test]
    async fn test_a_read_past_the_end_of_the_file_is_answered_by_the_inner_fetch() {
        let data = numbered(100);
        let cache = cache(&data);

        // The first fetch is clamped by the store.
        assert_eq!(cache.fetch(0..8).await.unwrap(), data.slice(0..8));
        // A read that runs off the end returns what the inner fetch returns for it.
        assert_eq!(cache.fetch(90..120).await.unwrap(), data.slice(90..100));
        assert_eq!(cache.fetch(200..208).await.unwrap(), Bytes::new());
    }

    #[test]
    fn test_block_cache_slices_across_adjacent_blocks() {
        let mut cache = BlockCache::default();
        cache.insert(0, Bytes::from_static(b"012"));
        cache.insert(3, Bytes::from_static(b""));
        cache.insert(3, Bytes::from_static(b"34"));
        cache.insert(5, Bytes::from_static(b"5"));
        cache.insert(10, Bytes::from_static(b"ab"));

        let test_cases = [
            (0..3, None, Bytes::from_static(b"012")),
            (2..6, None, Bytes::from_static(b"2345")),
            (6..6, None, Bytes::new()),
            (4..7, Some(6), Bytes::new()),
            (10..12, None, Bytes::from_static(b"ab")),
            (9..11, Some(9), Bytes::new()),
            (11..13, Some(12), Bytes::new()),
        ];

        for (range, missing, expected) in test_cases {
            assert_eq!(cache.first_missing(&range), missing, "{range:?}");
            if missing.is_none() {
                assert_eq!(cache.slice(range), expected);
            }
        }
        assert_eq!(cache.held_end_before(8), Some(6));
        assert_eq!(cache.next_start_after(8), Some(10));
    }
}
