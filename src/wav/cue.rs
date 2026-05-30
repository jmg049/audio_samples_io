use super::{chunks::ChunkID, error::WavError};

/// A single cue point (named marker) within a WAV file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuePoint {
    /// Unique identifier for this cue point within the file.
    pub id: u32,
    /// Position in the playlist order. For files without a PLST chunk,
    /// use `sample_offset` for the actual playback position.
    pub position: u32,
    /// FourCC of the data chunk this point refers to (almost always `data`).
    pub data_chunk_id: ChunkID,
    /// Byte offset of the start of the referenced data chunk.
    pub chunk_start: u32,
    /// Byte offset of the compressed block containing this point.
    /// Zero for uncompressed PCM.
    pub block_start: u32,
    /// Sample frame offset from `block_start` to this cue point.
    /// For uncompressed PCM this equals the absolute sample frame index.
    pub sample_offset: u32,
}

/// Parsed CUE chunk — an ordered list of named positions in the audio.
pub struct CueChunk<'a> {
    bytes: &'a [u8],
}

impl<'a> CueChunk<'a> {
    const HEADER_BYTES: usize = 4;
    const POINT_BYTES: usize = 24;

    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, WavError> {
        if bytes.len() < Self::HEADER_BYTES {
            return Err(WavError::chunk_parsing(
                "cue ",
                "0",
                format!("cue chunk must be at least 4 bytes, got {}", bytes.len()),
            ));
        }
        Ok(CueChunk { bytes })
    }

    /// Number of cue points declared in the chunk header.
    pub fn num_cue_points(&self) -> u32 {
        u32::from_le_bytes([self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]])
    }

    /// Parse and return all cue points.
    pub fn cue_points(&self) -> Result<Vec<CuePoint>, WavError> {
        let count = self.num_cue_points() as usize;
        let required = Self::HEADER_BYTES + count * Self::POINT_BYTES;
        if self.bytes.len() < required {
            return Err(WavError::chunk_parsing(
                "cue ",
                "4",
                format!(
                    "cue chunk declares {count} points but only {} bytes available (need {required})",
                    self.bytes.len(),
                ),
            ));
        }
        let mut points = Vec::with_capacity(count);
        for i in 0..count {
            let b = Self::HEADER_BYTES + i * Self::POINT_BYTES;
            points.push(CuePoint {
                id: u32_at(self.bytes, b),
                position: u32_at(self.bytes, b + 4),
                data_chunk_id: ChunkID::new(&[
                    self.bytes[b + 8],
                    self.bytes[b + 9],
                    self.bytes[b + 10],
                    self.bytes[b + 11],
                ]),
                chunk_start: u32_at(self.bytes, b + 12),
                block_start: u32_at(self.bytes, b + 16),
                sample_offset: u32_at(self.bytes, b + 20),
            });
        }
        Ok(points)
    }
}

/// Serialise cue points to a complete `cue ` chunk (header + size + point records).
///
/// Returns `None` when there are no points, so callers can skip emitting an empty chunk. Each
/// point record is 24 bytes, so the chunk body is always word-aligned.
pub fn cue_chunk_bytes(points: &[CuePoint]) -> Option<Vec<u8>> {
    if points.is_empty() {
        return None;
    }
    let mut body =
        Vec::with_capacity(CueChunk::HEADER_BYTES + points.len() * CueChunk::POINT_BYTES);
    body.extend_from_slice(&(points.len() as u32).to_le_bytes());
    for p in points {
        body.extend_from_slice(&p.id.to_le_bytes());
        body.extend_from_slice(&p.position.to_le_bytes());
        body.extend_from_slice(p.data_chunk_id.as_bytes());
        body.extend_from_slice(&p.chunk_start.to_le_bytes());
        body.extend_from_slice(&p.block_start.to_le_bytes());
        body.extend_from_slice(&p.sample_offset.to_le_bytes());
    }

    let mut chunk = Vec::with_capacity(8 + body.len());
    chunk.extend_from_slice(b"cue ");
    chunk.extend_from_slice(&(body.len() as u32).to_le_bytes());
    chunk.extend_from_slice(&body);
    Some(chunk)
}

#[inline]
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::type_complexity)]
    fn make_cue_bytes(points: &[(u32, u32, &[u8; 4], u32, u32, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(points.len() as u32).to_le_bytes());
        for &(id, pos, chunk_id, chunk_start, block_start, sample_offset) in points {
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&pos.to_le_bytes());
            buf.extend_from_slice(chunk_id);
            buf.extend_from_slice(&chunk_start.to_le_bytes());
            buf.extend_from_slice(&block_start.to_le_bytes());
            buf.extend_from_slice(&sample_offset.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_cue_no_points() {
        let bytes = make_cue_bytes(&[]);
        let chunk = CueChunk::from_bytes(&bytes).expect("valid cue chunk");
        assert_eq!(chunk.num_cue_points(), 0);
        assert!(chunk.cue_points().expect("no points").is_empty());
    }

    #[test]
    fn test_cue_two_points() {
        let bytes = make_cue_bytes(&[(1, 0, b"data", 0, 0, 44100), (2, 0, b"data", 0, 0, 88200)]);
        let chunk = CueChunk::from_bytes(&bytes).expect("valid cue chunk");
        assert_eq!(chunk.num_cue_points(), 2);
        let pts = chunk.cue_points().expect("two points");
        assert_eq!(pts[0].id, 1);
        assert_eq!(pts[0].sample_offset, 44100);
        assert_eq!(pts[1].id, 2);
        assert_eq!(pts[1].sample_offset, 88200);
        assert_eq!(pts[1].data_chunk_id, ChunkID::new(b"data"));
    }

    #[test]
    fn test_cue_too_small() {
        assert!(CueChunk::from_bytes(&[0u8; 3]).is_err());
    }

    #[test]
    fn test_cue_truncated_points() {
        let mut bytes = make_cue_bytes(&[]);
        // Claim 1 point but provide no point data
        bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
        let chunk = CueChunk::from_bytes(&bytes).expect("header ok");
        assert!(chunk.cue_points().is_err());
    }
}
