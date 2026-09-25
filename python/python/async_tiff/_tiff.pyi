from typing import Sequence

from ._ifd import ImageFileDirectory
from ._input import ObspecInput
from ._tile import Tile
from .enums import Endianness
from .store import ObjectStore

class TIFF:
    @classmethod
    async def open(
        cls,
        path: str,
        *,
        store: ObjectStore | ObspecInput,
        prefetch: int = 32768,
        readahead: int = 1048576,
    ) -> TIFF:
        """Open a new TIFF.

        Args:
            path: The path within the store to read from.
            store: The backend to use for data fetching.
            prefetch: The number of bytes to read at the start of the file, and at
                any metadata far from what has already been read.
            readahead: How far past already-read bytes a metadata read counts as near.
                A near read also fetches the gap before it and `readahead` bytes past
                it, since downloading that much costs about as long as one more request.

        Returns:
            A TIFF instance.
        """

    @property
    def endianness(self) -> Endianness:
        """The endianness of this TIFF file."""

    @property
    def header_byte_size(self) -> int:
        """Minimum prefetch size that covers all metadata.

        Pass this value as `prefetch` on a future [`TIFF.open`][async_tiff.TIFF.open]
        call to complete metadata reading in a single request.

        This is computed as the minimum non-zero offset across every IFD's `TileOffsets`
        and `StripOffsets`.
        """

    def ifd(self, index: int) -> ImageFileDirectory:
        """Access a specific IFD by index.

        Args:
            index: The IFD index to access.

        Returns:
            The requested IFD.
        """

    @property
    def ifds(self) -> list[ImageFileDirectory]:
        """Access the underlying IFDs of this TIFF.

        Each ImageFileDirectory (IFD) represents one of the internal "sub images" of
        this file.
        """
    async def fetch_tile(self, x: int, y: int, z: int) -> Tile:
        """Fetch a single tile.

        Args:
            x: The column index within the ifd to read from.
            y: The row index within the ifd to read from.
            z: The IFD index to read from.

        Returns:
            Tile response.
        """
    async def fetch_tiles(self, xy: Sequence[tuple[int, int]], z: int) -> list[Tile]:
        """Fetch multiple tiles concurrently.

        Args:
            xy: The (column, row) indexes within the ifd to read from.
            z: The IFD index to read from.

        Returns:
            Tile responses.
        """
