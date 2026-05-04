use super::error::WavError;

/// Parsed FACT chunk — stores the number of sample frames.
///
/// Required by the WAV spec for all non-PCM formats; optional for PCM.
/// The canonical field is `dwSampleLength` (4 bytes, little-endian u32).
pub struct FactChunk<'a> {
    bytes: &'a [u8],
}

impl<'a> FactChunk<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, WavError> {
        if bytes.len() < 4 {
            return Err(WavError::chunk_parsing(
                "fact",
                "0",
                format!("fact chunk must be at least 4 bytes, got {}", bytes.len()),
            ));
        }
        Ok(FactChunk { bytes })
    }

    pub fn num_samples(&self) -> u32 {
        u32::from_le_bytes([self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_chunk_parses_num_samples() {
        let bytes = 12345u32.to_le_bytes();
        let chunk = FactChunk::from_bytes(&bytes).expect("valid fact chunk");
        assert_eq!(chunk.num_samples(), 12345);
    }

    #[test]
    fn test_fact_chunk_too_small() {
        assert!(FactChunk::from_bytes(&[0u8; 3]).is_err());
    }

    #[test]
    fn test_fact_chunk_accepts_extra_bytes() {
        // Some encoders write > 4 bytes; we only need the first 4
        let mut bytes = vec![0xFFu8; 8];
        bytes[0..4].copy_from_slice(&999u32.to_le_bytes());
        let chunk = FactChunk::from_bytes(&bytes).expect("valid fact chunk");
        assert_eq!(chunk.num_samples(), 999);
    }
}
