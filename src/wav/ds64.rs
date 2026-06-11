//! `ds64` chunk parsing for RF64/BW64 files (EBU Tech 3306 / ITU-R BS.2088).
//!
//! An RF64/BW64 file is a RIFF/WAVE file whose 32-bit size fields have outgrown
//! `u32`. The form id is `RF64` (or `BW64`) instead of `RIFF`, the 32-bit RIFF
//! size carries the placeholder `0xFFFFFFFF`, and the first chunk after `WAVE`
//! must be `ds64`, which holds the real 64-bit sizes:
//!
//! | Offset | Field | Type |
//! |---|---|---|
//! | 0  | RIFF size | `u64` |
//! | 8  | `data` chunk size | `u64` |
//! | 16 | sample count | `u64` |
//! | 24 | table length | `u32` |
//! | 28 | table entries: chunk id (`[u8; 4]`) + chunk size (`u64`) | — |
//!
//! Any other chunk whose 32-bit size field is `0xFFFFFFFF` finds its true size
//! in the table.

use crate::error::{AudioIOError, AudioIOResult};
use crate::wav::chunks::ChunkID;

/// Size fields of an RF64/BW64 `ds64` chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ds64 {
    /// True 64-bit RIFF size (everything after the 8-byte RIFF header).
    pub riff_size: u64,
    /// True 64-bit size of the `data` chunk payload.
    pub data_size: u64,
    /// Sample frame count as declared by the writer (may be zero for streams).
    pub sample_count: u64,
    /// 64-bit sizes for additional chunks whose 32-bit field is `0xFFFFFFFF`.
    pub table: Vec<(ChunkID, u64)>,
}

/// Minimum `ds64` body length: the three `u64` sizes plus the table length.
pub const DS64_MIN_BODY_LEN: usize = 28;

impl Ds64 {
    /// Parse a `ds64` chunk body (the bytes after its 8-byte chunk header).
    ///
    /// Tolerates a truncated table (entries that run past `body` are dropped),
    /// matching the reader's general policy of accepting files that real-world
    /// encoders produce.
    pub fn from_bytes(body: &[u8]) -> AudioIOResult<Self> {
        if body.len() < DS64_MIN_BODY_LEN {
            return Err(AudioIOError::corrupted_data_simple(
                "ds64 chunk too small",
                format!("Need at least {DS64_MIN_BODY_LEN} bytes, found {}", body.len()),
            ));
        }

        let u64_at = |off: usize| u64::from_le_bytes(body[off..off + 8].try_into().expect("8-byte slice"));
        let riff_size = u64_at(0);
        let data_size = u64_at(8);
        let sample_count = u64_at(16);
        let table_len = u32::from_le_bytes(body[24..28].try_into().expect("4-byte slice")) as usize;

        let mut table = Vec::with_capacity(table_len.min(64));
        let mut off = DS64_MIN_BODY_LEN;
        for _ in 0..table_len {
            if off + 12 > body.len() {
                break; // truncated table — keep what we have
            }
            let id = ChunkID::new(body[off..off + 4].try_into().expect("4-byte slice"));
            let size = u64_at(off + 4);
            table.push((id, size));
            off += 12;
        }

        Ok(Ds64 {
            riff_size,
            data_size,
            sample_count,
            table,
        })
    }

    /// The true 64-bit size for `id`, when its 32-bit size field holds the
    /// `0xFFFFFFFF` placeholder.
    pub fn chunk_size(&self, id: ChunkID) -> Option<u64> {
        use crate::wav::chunks::DATA_CHUNK;
        if id == DATA_CHUNK {
            return Some(self.data_size);
        }
        self.table.iter().find(|(tid, _)| *tid == id).map(|&(_, size)| size)
    }

    /// Resolve a chunk's declared 32-bit size to its true size in bytes.
    ///
    /// Returns the ds64 size when `declared` is the `0xFFFFFFFF` placeholder and
    /// this ds64 knows the chunk; otherwise returns `declared` unchanged.
    pub fn resolve(&self, id: ChunkID, declared: u32) -> u64 {
        if declared == u32::MAX
            && let Some(size) = self.chunk_size(id)
        {
            return size;
        }
        declared as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::chunks::{BEXT_CHUNK, DATA_CHUNK};

    fn body(riff: u64, data: u64, samples: u64, table: &[([u8; 4], u64)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&riff.to_le_bytes());
        b.extend_from_slice(&data.to_le_bytes());
        b.extend_from_slice(&samples.to_le_bytes());
        b.extend_from_slice(&(table.len() as u32).to_le_bytes());
        for (id, size) in table {
            b.extend_from_slice(id);
            b.extend_from_slice(&size.to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_minimal_ds64() {
        let ds = Ds64::from_bytes(&body(0x1_0000_0000, 0xFFFF_FFF0, 1_000_000, &[])).expect("parse");
        assert_eq!(ds.riff_size, 0x1_0000_0000);
        assert_eq!(ds.data_size, 0xFFFF_FFF0);
        assert_eq!(ds.sample_count, 1_000_000);
        assert!(ds.table.is_empty());
    }

    #[test]
    fn parses_table_and_resolves() {
        let ds = Ds64::from_bytes(&body(100, 50, 10, &[(*b"bext", 0x2_0000_0001)])).expect("parse");
        assert_eq!(ds.chunk_size(BEXT_CHUNK), Some(0x2_0000_0001));
        assert_eq!(ds.chunk_size(DATA_CHUNK), Some(50));
        assert_eq!(ds.resolve(DATA_CHUNK, u32::MAX), 50);
        assert_eq!(ds.resolve(DATA_CHUNK, 1234), 1234); // not the placeholder
        assert_eq!(ds.resolve(ChunkID::new(b"junk"), u32::MAX), u32::MAX as u64); // unknown id
    }

    #[test]
    fn rejects_short_body_and_tolerates_truncated_table() {
        assert!(Ds64::from_bytes(&[0u8; 27]).is_err());
        // Table claims 2 entries but only 1 fits.
        let mut b = body(100, 50, 10, &[(*b"bext", 7)]);
        b[24..28].copy_from_slice(&2u32.to_le_bytes());
        let ds = Ds64::from_bytes(&b).expect("parse");
        assert_eq!(ds.table.len(), 1);
    }
}
