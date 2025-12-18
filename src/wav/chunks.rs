use core::fmt::{Display, Formatter, Result as FmtResult};

/// FourCC chunk identifier wrapper -- does not own the data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ChunkID {
    pub id: [u8; 4],
}

impl AsRef<[u8]> for ChunkID {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.id
    }
}

impl Display for ChunkID {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match core::str::from_utf8(&self.id) {
            Ok(s) => write!(f, "{}", s),
            Err(e) => {
                writeln!(f, "ChunkID Display error: {}", e)?;
                write!(
                    f,
                    "0x{:02X}{:02X}{:02X}{:02X}",
                    self.id[0], self.id[1], self.id[2], self.id[3]
                )
            }
        }
    }
}

impl From<&[u8; 4]> for ChunkID {
    fn from(value: &[u8; 4]) -> Self {
        ChunkID { id: *value }
    }
}

impl ChunkID {
    #[inline]
    pub const fn new(id: &[u8; 4]) -> Self {
        ChunkID { id: *id }
    }

    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 4] {
        &self.id
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(&self.id).ok()
    }
}

/// Lightweight description of a RIFF/WAV chunk
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkDesc {
    pub id: ChunkID,
    pub offset: usize,
    /// Logical size of the chunk data (excluding header and padding)
    pub logical_size: usize,
    /// Total size including header and padding (for file positioning)
    pub total_size: usize,
}

impl ChunkDesc {
    /// Returns the logical size of the chunk data (excluding header and padding)
    #[inline]
    pub const fn len(&self) -> usize {
        self.logical_size
    }

    /// Returns true if the chunk has no logical data
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.logical_size == 0
    }

    /// Returns the range of bytes containing the logical chunk data (no header, no padding)
    #[inline]
    pub const fn data_range(&self) -> std::ops::Range<usize> {
        let start = self.offset + 8; // Skip chunk header
        start..(start + self.logical_size)
    }

    /// Returns total chunk size including header and padding (for file positioning)
    #[inline]
    pub const fn total_size(&self) -> usize {
        self.total_size
    }
}

impl Display for ChunkDesc {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Chunk ID: {}, Offset: {}, Logical Size: {}, Total Size: {}",
            self.id, self.offset, self.logical_size, self.total_size
        )
    }
}

// Chunk Management
pub trait ChunkAccessor {
    fn get_chunk(&self, chunk_id: &ChunkID) -> Option<ChunkDesc>;
}

pub const RIFF_CHUNK: ChunkID = ChunkID::new(b"RIFF");
pub const WAVE_CHUNK: ChunkID = ChunkID::new(b"WAVE");
pub const FMT_CHUNK: ChunkID = ChunkID::new(b"fmt ");
pub const DATA_CHUNK: ChunkID = ChunkID::new(b"data");
pub const FACT_CHUNK: ChunkID = ChunkID::new(b"fact");
pub const LIST_CHUNK: ChunkID = ChunkID::new(b"LIST");
pub const CUE_CHUNK: ChunkID = ChunkID::new(b"cue ");
