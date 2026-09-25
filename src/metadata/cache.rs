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
    /// The length of the file, once a fetch has come back short
    len: Option<u64>,
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

    /// Where the run of adjacent blocks that ends at `end` starts
    fn run_start(&self, end: u64) -> u64 {
        let mut start = end;
        while let Some((&block_start, bytes)) = self.blocks.range(..start).next_back() {
            if block_start + bytes.len() as u64 != start {
                break;
            }
            start = block_start;
        }
        start
    }

    /// Where the first block after `offset` starts
    fn next_start_after(&self, offset: u64) -> Option<u64> {
        self.blocks
            .range(offset.saturating_add(1)..)
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
///   the cache fetches the gap between them, the read, and as much again past the read as the run
///   of held bytes it extends, between `initial` and `readahead` bytes. The parser is walking one
///   region of metadata, such as a directory, its tag values and the next directory, and the
///   growth doubles each fetch, as a sequential readahead does, so a large region costs a few
///   requests and a small one little more than its size.
/// - **Far from held bytes.** Otherwise the cache fetches `initial` bytes starting at the read,
///   which covers a whole directory in one request without assuming anything about what surrounds
///   it.
///
/// A fetch never includes bytes already held, so no byte is fetched twice.
///
/// `readahead` weighs bytes against round trips: downloading that many bytes takes about as long
/// as one more request, so filling a smaller gap is cheaper than a second request. This is the
/// bandwidth-delay product that Apache Arrow's read coalescing uses as its gap limit, and the
/// default of 1 MiB matches object_store's range coalescing default. A gap can be image data,
/// as in a file that writes each directory after a small image; it is then downloaded and held
/// too.
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

    /// Set how far past held bytes a read counts as near, and the most a fetch near held bytes
    /// reads ahead, otherwise defaults to 1 MiB.
    pub fn with_readahead(mut self, readahead: u64) -> Self {
        self.readahead = readahead;
        self
    }

    /// The range to fetch for a read whose first missing byte is `missing` and which ends at `end`
    fn next_fetch(&self, cache: &BlockCache, missing: u64, end: u64) -> Range<u64> {
        let (start, mut fetch_end) = match cache.held_end_before(missing) {
            Some(held_end) if missing - held_end <= self.readahead => {
                let run = end - cache.run_start(held_end);
                let ahead = run.clamp(self.initial, self.readahead.max(self.initial));
                (held_end, end.saturating_add(ahead))
            }
            _ => (missing, end.max(missing.saturating_add(self.initial))),
        };
        if let Some(next) = cache.next_start_after(missing) {
            fetch_end = fetch_end.min(next);
        }
        if let Some(len) = cache.len {
            fetch_end = fetch_end.min(len);
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

        let mut range = range;
        loop {
            // A read past the end of the file gets the bytes that exist, as from a store.
            if let Some(len) = cache.len {
                range.end = range.end.min(len);
                range.start = range.start.min(range.end);
            }
            let Some(missing) = cache.first_missing(&range) else {
                break;
            };
            let fetch_range = self.next_fetch(&cache, missing, range.end);
            let bytes = self.inner.fetch(fetch_range.clone()).await?;
            // A store clamps a range that runs past the end of the file, so a short fetch means
            // the file ends there.
            if (bytes.len() as u64) < fetch_range.end - fetch_range.start {
                cache.len = Some(fetch_range.start + bytes.len() as u64);
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

        // The gap from the held end at 9_064 is fetched with the read, and past the read as much
        // again as the run of held bytes it extends: 9_000..9_172 is 172 bytes.
        assert_eq!(fetched(&cache), vec![0..64, 9_000..9_064, 9_064..9_344]);

        // So the next values along come from memory.
        cache.fetch(9_172..9_340).await.unwrap();
        assert_eq!(fetched(&cache).len(), 3);
    }

    #[tokio::test]
    async fn test_sequential_reads_double_up_to_the_readahead() {
        // A COG whose tile tables are larger than the first fetch.
        let data = numbered(10_000);
        let cache = cache(&data);

        for range in [0..8, 64..72, 144..150, 300..310, 620..630] {
            cache.fetch(range).await.unwrap();
        }
        assert_eq!(
            fetched(&cache),
            vec![0..64, 64..144, 144..300, 300..566, 566..886]
        );
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
    async fn test_a_read_past_the_end_of_the_file_returns_what_exists() {
        let data = numbered(100);
        let cache = cache(&data);

        assert_eq!(cache.fetch(0..8).await.unwrap(), data.slice(0..8));
        assert_eq!(cache.fetch(90..120).await.unwrap(), data.slice(90..100));
        // The short fetch taught the cache where the file ends, so these fetch nothing.
        assert_eq!(cache.fetch(90..120).await.unwrap(), data.slice(90..100));
        assert_eq!(cache.fetch(200..208).await.unwrap(), Bytes::new());
        assert_eq!(fetched(&cache), vec![0..64, 64..240]);
    }

    #[tokio::test]
    async fn test_an_empty_read_fetches_nothing() {
        let cache = cache(&numbered(100));
        assert_eq!(cache.fetch(50..50).await.unwrap(), Bytes::new());
        assert!(fetched(&cache).is_empty());
    }

    #[tokio::test]
    async fn test_a_read_at_the_largest_offsets_does_not_overflow() {
        let cache = cache(&numbered(100));
        assert_eq!(
            cache.fetch(u64::MAX - 8..u64::MAX).await.unwrap(),
            Bytes::new()
        );
    }

    #[tokio::test]
    async fn test_concurrent_reads_of_the_same_bytes_fetch_once() {
        let data = numbered(10_000);
        let cache = cache(&data);

        let (a, b) = tokio::join!(cache.fetch(9_000..9_008), cache.fetch(9_004..9_012));
        assert_eq!(a.unwrap(), data.slice(9_000..9_008));
        assert_eq!(b.unwrap(), data.slice(9_004..9_012));
        assert_eq!(fetched(&cache), vec![9_000..9_064]);
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
