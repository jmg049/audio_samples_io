use super::error::WavError;

/// Loop playback mode for a [`SampleLoop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopType {
    /// Play the loop region forward on each repetition.
    Forward,
    /// Alternate forward and backward on each repetition (ping-pong).
    PingPong,
    /// Play the loop region backward on each repetition.
    Reverse,
    /// Vendor-defined or reserved loop type.
    Unknown(u32),
}

impl From<u32> for LoopType {
    fn from(v: u32) -> Self {
        match v {
            0 => LoopType::Forward,
            1 => LoopType::PingPong,
            2 => LoopType::Reverse,
            other => LoopType::Unknown(other),
        }
    }
}

/// A single loop region defined in an [`SmplChunk`].
#[derive(Debug, Clone)]
pub struct SampleLoop {
    /// Loop identifier (unique within the file).
    pub id: u32,
    /// Playback mode for this loop.
    pub loop_type: LoopType,
    /// First sample frame of the loop region.
    pub start: u32,
    /// Last sample frame of the loop region (inclusive).
    pub end: u32,
    /// Sub-sample fractional loop point in 1/2^32 of a sample. Usually 0.
    pub fraction: u32,
    /// Number of times to play the loop. 0 means infinite.
    pub play_count: u32,
}

/// Parsed SMPL chunk — MIDI sampler metadata: pitch tuning, unity note, loop points.
///
/// Widely used by sample libraries, hardware samplers, and DAW instruments
/// to control how the audio is pitched and looped during playback.
pub struct SmplChunk<'a> {
    bytes: &'a [u8],
}

impl<'a> SmplChunk<'a> {
    const HEADER_SIZE: usize = 36;
    const LOOP_SIZE: usize = 24;

    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, WavError> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(WavError::chunk_parsing(
                "smpl",
                "0",
                format!(
                    "smpl chunk must be at least {} bytes, got {}",
                    Self::HEADER_SIZE,
                    bytes.len()
                ),
            ));
        }
        Ok(SmplChunk { bytes })
    }

    /// MIDI manufacturer ID (0 = not device-specific).
    pub fn manufacturer(&self) -> u32 {
        u32_at(self.bytes, 0)
    }

    /// MIDI product ID (0 = not device-specific).
    pub fn product(&self) -> u32 {
        u32_at(self.bytes, 4)
    }

    /// Duration of one sample in nanoseconds (= 1_000_000_000 / sample_rate).
    pub fn sample_period(&self) -> u32 {
        u32_at(self.bytes, 8)
    }

    /// MIDI note number of the recorded pitch (60 = middle C).
    pub fn midi_unity_note(&self) -> u32 {
        u32_at(self.bytes, 12)
    }

    /// Fine-pitch offset above `midi_unity_note` in 1/2^32 semitones.
    pub fn midi_pitch_fraction(&self) -> u32 {
        u32_at(self.bytes, 16)
    }

    /// SMPTE format (0 if unused; otherwise 24, 25, 29, or 30).
    pub fn smpte_format(&self) -> u32 {
        u32_at(self.bytes, 20)
    }

    /// SMPTE time offset packed as `0xHHMMSSFF` (hours/minutes/seconds/frames).
    pub fn smpte_offset(&self) -> u32 {
        u32_at(self.bytes, 24)
    }

    /// Number of loop records following the header.
    pub fn num_loops(&self) -> u32 {
        u32_at(self.bytes, 28)
    }

    /// Byte size of optional extra sampler data after all loop records.
    pub fn sampler_data_size(&self) -> u32 {
        u32_at(self.bytes, 32)
    }

    /// Parse and return all loop records.
    pub fn loops(&self) -> Result<Vec<SampleLoop>, WavError> {
        let count = self.num_loops() as usize;
        let required = Self::HEADER_SIZE + count * Self::LOOP_SIZE;
        if self.bytes.len() < required {
            return Err(WavError::chunk_parsing(
                "smpl",
                "36",
                format!(
                    "smpl declares {count} loops but only {} bytes available (need {required})",
                    self.bytes.len(),
                ),
            ));
        }
        let mut loops = Vec::with_capacity(count);
        for i in 0..count {
            let b = Self::HEADER_SIZE + i * Self::LOOP_SIZE;
            loops.push(SampleLoop {
                id: u32_at(self.bytes, b),
                loop_type: LoopType::from(u32_at(self.bytes, b + 4)),
                start: u32_at(self.bytes, b + 8),
                end: u32_at(self.bytes, b + 12),
                fraction: u32_at(self.bytes, b + 16),
                play_count: u32_at(self.bytes, b + 20),
            });
        }
        Ok(loops)
    }
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

    fn make_smpl_bytes(
        manufacturer: u32,
        product: u32,
        sample_period: u32,
        unity_note: u32,
        pitch_fraction: u32,
        smpte_format: u32,
        smpte_offset: u32,
        loops: &[(u32, u32, u32, u32, u32, u32)],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&manufacturer.to_le_bytes());
        buf.extend_from_slice(&product.to_le_bytes());
        buf.extend_from_slice(&sample_period.to_le_bytes());
        buf.extend_from_slice(&unity_note.to_le_bytes());
        buf.extend_from_slice(&pitch_fraction.to_le_bytes());
        buf.extend_from_slice(&smpte_format.to_le_bytes());
        buf.extend_from_slice(&smpte_offset.to_le_bytes());
        buf.extend_from_slice(&(loops.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // sampler_data_size
        for &(id, loop_type, start, end, fraction, play_count) in loops {
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&loop_type.to_le_bytes());
            buf.extend_from_slice(&start.to_le_bytes());
            buf.extend_from_slice(&end.to_le_bytes());
            buf.extend_from_slice(&fraction.to_le_bytes());
            buf.extend_from_slice(&play_count.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_smpl_header_fields() {
        // sample_period = 1e9 / 44100 ≈ 22675
        let bytes = make_smpl_bytes(0, 0, 22675, 60, 0, 0, 0, &[]);
        let chunk = SmplChunk::from_bytes(&bytes).expect("valid smpl chunk");
        assert_eq!(chunk.manufacturer(), 0);
        assert_eq!(chunk.sample_period(), 22675);
        assert_eq!(chunk.midi_unity_note(), 60);
        assert_eq!(chunk.num_loops(), 0);
    }

    #[test]
    fn test_smpl_forward_loop() {
        let bytes = make_smpl_bytes(0, 0, 22675, 69, 0, 0, 0, &[(1, 0, 1000, 44000, 0, 0)]);
        let chunk = SmplChunk::from_bytes(&bytes).expect("valid smpl chunk");
        let loops = chunk.loops().expect("one loop");
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].id, 1);
        assert_eq!(loops[0].loop_type, LoopType::Forward);
        assert_eq!(loops[0].start, 1000);
        assert_eq!(loops[0].end, 44000);
        assert_eq!(loops[0].play_count, 0);
    }

    #[test]
    fn test_smpl_ping_pong_loop() {
        let bytes = make_smpl_bytes(0, 0, 0, 60, 0, 0, 0, &[(1, 1, 0, 1000, 0, 4)]);
        let chunk = SmplChunk::from_bytes(&bytes).expect("valid smpl");
        let loops = chunk.loops().expect("one loop");
        assert_eq!(loops[0].loop_type, LoopType::PingPong);
        assert_eq!(loops[0].play_count, 4);
    }

    #[test]
    fn test_smpl_unknown_loop_type() {
        let bytes = make_smpl_bytes(0, 0, 0, 60, 0, 0, 0, &[(1, 99, 0, 1000, 0, 0)]);
        let chunk = SmplChunk::from_bytes(&bytes).expect("valid smpl");
        let loops = chunk.loops().expect("one loop");
        assert_eq!(loops[0].loop_type, LoopType::Unknown(99));
    }

    #[test]
    fn test_smpl_too_small() {
        assert!(SmplChunk::from_bytes(&[0u8; 35]).is_err());
    }

    #[test]
    fn test_smpl_truncated_loops() {
        let mut bytes = make_smpl_bytes(0, 0, 0, 60, 0, 0, 0, &[]);
        // Claim 1 loop but supply no loop data
        bytes[28..32].copy_from_slice(&1u32.to_le_bytes());
        let chunk = SmplChunk::from_bytes(&bytes).expect("header ok");
        assert!(chunk.loops().is_err());
    }
}
