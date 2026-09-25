use std::fmt::Debug;
use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;

use crate::error::{AsyncTiffError, AsyncTiffResult};
use crate::reader::{AsyncFileReader, EndianAwareReader, Endianness};

/// A data source that can be used with [`TiffMetadataReader`][crate::metadata::TiffMetadataReader]
/// and [`ImageFileDirectoryReader`][crate::metadata::ImageFileDirectoryReader] to load
/// [`ImageFileDirectory`][crate::ImageFileDirectory]s.
///
/// Note that implementation is provided for [`AsyncFileReader`].
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait MetadataFetch: Debug + Send + Sync + 'static {
    /// Return a future that fetches the specified range of bytes asynchronously
    ///
    /// Note the returned type is a boxed future, often created by
    /// [futures::FutureExt::boxed]. See the trait documentation for an example.
    async fn fetch(&self, range: Range<u64>) -> AsyncTiffResult<Bytes>;
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<T: AsyncFileReader> MetadataFetch for T {
    async fn fetch(&self, range: Range<u64>) -> AsyncTiffResult<Bytes> {
        self.get_bytes(range).await
    }
}

pub(crate) struct MetadataCursor<'a, F: MetadataFetch> {
    fetch: &'a F,
    offset: u64,
    endianness: Endianness,
}

impl<'a, F: MetadataFetch> MetadataCursor<'a, F> {
    pub fn new(fetch: &'a F, endianness: Endianness) -> Self {
        Self {
            fetch,
            offset: 0,
            endianness,
        }
    }

    pub fn new_with_offset(fetch: &'a F, endianness: Endianness, offset: u64) -> Self {
        Self {
            fetch,
            offset,
            endianness,
        }
    }

    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    pub fn seek(&mut self, offset: u64) {
        self.offset = offset;
    }

    /// Read the given number of bytes, advancing the internal cursor state by the same amount.
    ///
    /// Fails if the file ends before `length` bytes, so a length read from a malformed file never
    /// reaches an allocation.
    pub(crate) async fn read(&mut self, length: u64) -> AsyncTiffResult<EndianAwareReader> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| past_the_end(self.offset, length))?;
        let bytes = self.fetch.fetch(self.offset..end).await?;
        if (bytes.len() as u64) < length {
            return Err(past_the_end(self.offset, length));
        }
        self.offset = end;
        Ok(EndianAwareReader::new(bytes, self.endianness))
    }

    /// Read a u16 from the cursor, advancing the internal state by 2 bytes.
    pub(crate) async fn read_u16(&mut self) -> AsyncTiffResult<u16> {
        self.read(2).await?.read_u16()
    }

    /// Read a u32 from the cursor, advancing the internal state by 4 bytes.
    pub(crate) async fn read_u32(&mut self) -> AsyncTiffResult<u32> {
        self.read(4).await?.read_u32()
    }

    /// Read a u64 from the cursor, advancing the internal state by 8 bytes.
    pub(crate) async fn read_u64(&mut self) -> AsyncTiffResult<u64> {
        self.read(8).await?.read_u64()
    }
}

fn past_the_end(offset: u64, length: u64) -> AsyncTiffError {
    std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        format!("{length} bytes at offset {offset} run past the end of the file"),
    )
    .into()
}
