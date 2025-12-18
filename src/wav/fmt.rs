use core::fmt::{Display, Formatter, Result as FmtResult};

use audio_samples::SampleType;

use crate::{
    types::ValidatedSampleType,
    wav::{FormatCode, error::WavError},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FmtChunk<'a> {
    Base(&'a [u8; 16]),
    Extensible(&'a [u8; 40]),
}

impl<'a> FmtChunk<'a> {
    /// Primary constructor for FmtChunk
    ///
    /// # Arguments
    ///
    /// * `bytes` - A byte slice representing the FMT chunk data
    ///
    /// Must be either 16 bytes (base FMT) or 40 bytes (extensible FMT)
    ///
    /// # Returns
    ///
    /// Ok(FmtChunk) if the byte slice is valid, Err(WavError) otherwise
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, WavError> {
        match bytes.len() {
            16 => {
                let b: &[u8; 16] = bytes
                    .try_into()
                    .map_err(|_| WavError::InvalidFmtChunkSize(bytes.len()))?;
                Ok(FmtChunk::Base(b))
            }
            40 => {
                let b: &[u8; 40] = bytes
                    .try_into()
                    .map_err(|_| WavError::InvalidFmtChunkSize(bytes.len()))?;
                Ok(FmtChunk::Extensible(b))
            }
            len => Err(WavError::InvalidFmtChunkSize(len)),
        }
    }

    /// Constructor that also runs consistency validation.
    pub fn from_bytes_validated(bytes: &'a [u8]) -> Result<Self, WavError> {
        let fmt_chunk = Self::from_bytes(bytes)?;
        fmt_chunk.validate_format_consistency()?;
        Ok(fmt_chunk)
    }

    /// Get the raw bytes of the FMT chunk
    ///
    /// # Returns
    ///
    /// Byte slice representing the FMT chunk data -- guaranteed to be either 16 or 40 bytes
    pub const fn as_bytes(&self) -> &[u8] {
        match self {
            FmtChunk::Base(slice) => *slice,
            FmtChunk::Extensible(slice) => *slice,
        }
    }

    /// Attempt to convert to base FMT chunk bytes
    ///
    /// # Returns
    ///
    /// Some(&[u8; 16]) if base FMT chunk, None if extensible FMT chunk
    pub const fn try_into_base(&'a self) -> Option<&'a [u8; 16]> {
        match self {
            FmtChunk::Base(bytes) => Some(bytes),
            FmtChunk::Extensible(_) => None,
        }
    }

    /// Attempt to convert to extensible FMT chunk bytes
    ///
    /// # Returns
    ///
    /// Some(&[u8; 40]) if extensible FMT chunk, None if base FMT chunk
    pub const fn try_into_extensible(&'a self) -> Option<&'a [u8; 40]> {
        match self {
            FmtChunk::Base(_) => None,
            FmtChunk::Extensible(bytes) => Some(bytes),
        }
    }

    /// Get the format code from the FMT chunk
    ///
    /// # Returns
    ///
    /// FormatCode representing the audio format
    pub const fn format_code(&self) -> FormatCode {
        FormatCode::const_from(match self {
            FmtChunk::Base(bytes) => u16::from_le_bytes([bytes[0], bytes[1]]),
            FmtChunk::Extensible(bytes) => u16::from_le_bytes([bytes[0], bytes[1]]),
        })
    }

    /// Get the number of channels from the FMT chunk
    ///
    /// # Returns
    ///
    /// Number of audio channels
    pub const fn channels(&self) -> u16 {
        match self {
            FmtChunk::Base(bytes) => u16::from_le_bytes([bytes[2], bytes[3]]),
            FmtChunk::Extensible(bytes) => u16::from_le_bytes([bytes[2], bytes[3]]),
        }
    }

    /// Get the sample rate from the FMT chunk
    ///
    /// # Returns
    ///
    /// Sample rate in Hz
    pub const fn sample_rate(&self) -> u32 {
        match self {
            FmtChunk::Base(bytes) => u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            FmtChunk::Extensible(bytes) => {
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
            }
        }
    }

    /// Get the byte rate from the FMT chunk
    ///
    /// # Returns
    ///
    /// Number of bytes per second of audio data
    pub const fn byte_rate(&self) -> u32 {
        match self {
            FmtChunk::Base(bytes) => u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            FmtChunk::Extensible(bytes) => {
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]])
            }
        }
    }

    /// Get the block align from the FMT chunk
    ///
    /// # Returns
    ///
    /// Number of bytes per sample frame (all channels)
    pub const fn block_align(&self) -> u16 {
        match self {
            FmtChunk::Base(bytes) => u16::from_le_bytes([bytes[12], bytes[13]]),
            FmtChunk::Extensible(bytes) => u16::from_le_bytes([bytes[12], bytes[13]]),
        }
    }

    /// Get the bits per sample from the FMT chunk
    ///
    /// # Returns
    ///
    /// Number of bits per sample
    pub const fn bits_per_sample(&self) -> u16 {
        match self {
            FmtChunk::Base(bytes) => u16::from_le_bytes([bytes[14], bytes[15]]),
            FmtChunk::Extensible(bytes) => u16::from_le_bytes([bytes[14], bytes[15]]),
        }
    }

    /// Get the bytes per sample from the FMT chunk
    ///
    /// # Returns
    ///
    /// Number of bytes per sample (bits_per_sample / 8)
    pub const fn bytes_per_sample(&self) -> u16 {
        self.bits_per_sample() / 8
    }

    /// Convenience method to get all FMT chunk fields as a tuple.
    ///
    /// # Returns
    ///
    /// (FormatCode, u16 channels, u32 sample_rate, u32 byte_rate, u16 block_align, u16 bits_per_sample)
    pub const fn fmt_chunk(&self) -> (FormatCode, u16, u32, u32, u16, u16) {
        (
            self.format_code(),
            self.channels(),
            self.sample_rate(),
            self.byte_rate(),
            self.block_align(),
            self.bits_per_sample(),
        )
    }

    /// If this is an extensible FMT chunk, returns a reference to the extended bytes
    ///
    /// # Returns
    ///
    /// Some(&[u8; 24]) if extensible, None if base FMT chunk
    ///
    /// # Panics
    ///
    /// This function will never panic provided the FmtChunk enum is constructed via ``from_bytes``
    pub fn extended_bytes(&'a self) -> Option<&'a [u8; 24]> {
        match self {
            FmtChunk::Base(_) => None,
            FmtChunk::Extensible(bytes) => {
                let b: &[u8; 24] = bytes[16..40]
                    .try_into()
                    .expect("Guaranteed by enum variant and constructor");
                Some(b)
            }
        }
    }

    /// If this is an extensible FMT chunk, returns the subformat (format code and sample type)
    ///
    /// # Returns
    ///
    /// Some((FormatCode, SampleType)) if extensible, None if base FMT
    pub const fn subformat(&'a self) -> Result<Option<(FormatCode, SampleType)>, WavError> {
        match self {
            FmtChunk::Base(_) => Ok(None),
            FmtChunk::Extensible(bytes) => {
                let format_code =
                    FormatCode::const_from(u16::from_le_bytes([bytes[18], bytes[19]]));
                let bits_per_sample = self.bits_per_sample();
                let sample_type = SampleType::from_bits(bits_per_sample);

                Ok(Some((format_code, sample_type)))
            }
        }
    }

    /// Get the actual sample type of the audio data, considering extensible format if present
    ///
    /// # Returns
    ///
    /// SampleType representing the audio sample type
    ///
    /// # Errors
    ///
    /// Err(WavError) if the sample type is unsupported or cannot be determined
    pub fn actual_sample_type(&'a self) -> Result<ValidatedSampleType, WavError> {
        let bits_per_sample = self.bits_per_sample();

        // Extensible carries an explicit subformat we should honor.
        if let Some((format_code, _)) = self.subformat()? {
            return match format_code {
                FormatCode::IeeeFloat => match bits_per_sample {
                    32 => Ok(ValidatedSampleType::F32),
                    64 => Ok(ValidatedSampleType::F64),
                    _ => Err(WavError::UnsupportedSampleType),
                },
                _ => ValidatedSampleType::try_from(SampleType::from_bits(bits_per_sample))
                    .map_err(|_| WavError::UnsupportedSampleType),
            };
        }

        // Base FMT relies on the format code to disambiguate PCM vs float for 32/64-bit.
        match self.format_code() {
            FormatCode::IeeeFloat => match bits_per_sample {
                32 => Ok(ValidatedSampleType::F32),
                64 => Ok(ValidatedSampleType::F64),
                _ => Err(WavError::UnsupportedSampleType),
            },
            _ => ValidatedSampleType::try_from(SampleType::from_bits(bits_per_sample))
                .map_err(|_| WavError::UnsupportedSampleType),
        }
    }

    /// Validate the consistency of FMT chunk fields
    ///
    /// Checks that:
    /// - byte_rate = sample_rate * block_align
    /// - block_align = channels * bytes_per_sample
    /// - bits_per_sample is byte-aligned
    /// - All fields are within reasonable ranges
    pub fn validate_format_consistency(&self) -> Result<(), WavError> {
        let channels = self.channels();
        let sample_rate = self.sample_rate();
        let byte_rate = self.byte_rate();
        let block_align = self.block_align();
        let bits_per_sample = self.bits_per_sample();

        // Check for zero values
        if channels == 0 {
            return Err(WavError::invalid_format("Channels cannot be zero"));
        }
        if sample_rate == 0 {
            return Err(WavError::invalid_format("Sample rate cannot be zero"));
        }
        if byte_rate == 0 {
            return Err(WavError::invalid_format("Byte rate cannot be zero"));
        }
        if block_align == 0 {
            return Err(WavError::invalid_format("Block align cannot be zero"));
        }
        if bits_per_sample == 0 {
            return Err(WavError::invalid_format("Bits per sample cannot be zero"));
        }

        // Check bits per sample is byte-aligned
        if !bits_per_sample.is_multiple_of(8) {
            return Err(WavError::invalid_format(&format!(
                "Bits per sample {} is not byte-aligned",
                bits_per_sample
            )));
        }

        let bytes_per_sample = bits_per_sample / 8;

        // Validate block_align = channels * bytes_per_sample
        let expected_block_align = channels * bytes_per_sample;
        if block_align != expected_block_align {
            return Err(WavError::invalid_format(&format!(
                "Block align {} does not match expected {} (channels {} * bytes_per_sample {})",
                block_align, expected_block_align, channels, bytes_per_sample
            )));
        }

        // Validate byte_rate = sample_rate * block_align
        let expected_byte_rate = sample_rate * block_align as u32;
        if byte_rate != expected_byte_rate {
            return Err(WavError::invalid_format(&format!(
                "Byte rate {} does not match expected {} (sample_rate {} * block_align {})",
                byte_rate, expected_byte_rate, sample_rate, block_align
            )));
        }

        // Check reasonable ranges
        if channels > 256 {
            return Err(WavError::invalid_format(&format!(
                "Too many channels: {} (maximum 256)",
                channels
            )));
        }
        if sample_rate > 384000 {
            return Err(WavError::invalid_format(&format!(
                "Sample rate too high: {} Hz (maximum 384000)",
                sample_rate
            )));
        }
        if bits_per_sample > 64 {
            return Err(WavError::invalid_format(&format!(
                "Bits per sample too high: {} (maximum 64)",
                bits_per_sample
            )));
        }

        // Validate sample type
        let _ = self.actual_sample_type()?;

        Ok(())
    }
}

// Todo: Properly implement Display for FmtChunk
// - Alternative formatting options
// - use of the crate feature "colored" which enables the use of the ``colored`` crate for colored terminal output
impl Display for FmtChunk<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let (format, channels, sample_rate, byte_rate, block_align, bits_per_sample) =
            self.fmt_chunk();
        write!(
            f,
            "FmtChunk {{ format: {:?}, channels: {}, sample_rate: {}, byte_rate: {}, block_align: {}, bits_per_sample: {} }}",
            format, channels, sample_rate, byte_rate, block_align, bits_per_sample
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_base_fmt_bytes(
        format_code: u16,
        channels: u16,
        sample_rate: u32,
        byte_rate: u32,
        block_align: u16,
        bits_per_sample: u16,
    ) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..2].copy_from_slice(&format_code.to_le_bytes());
        bytes[2..4].copy_from_slice(&channels.to_le_bytes());
        bytes[4..8].copy_from_slice(&sample_rate.to_le_bytes());
        bytes[8..12].copy_from_slice(&byte_rate.to_le_bytes());
        bytes[12..14].copy_from_slice(&block_align.to_le_bytes());
        bytes[14..16].copy_from_slice(&bits_per_sample.to_le_bytes());
        bytes
    }

    #[test]
    fn test_fmt_validate_rejects_zero_channels() {
        let bytes = make_base_fmt_bytes(1, 0, 44_100, 176_400, 4, 16);
        let fmt = FmtChunk::from_bytes(&bytes).unwrap();
        let err = fmt.validate_format_consistency().unwrap_err();
        assert!(err.to_string().contains("Channels cannot be zero"));
    }

    #[test]
    fn test_fmt_validate_rejects_block_align_mismatch() {
        // For 2ch, 16-bit, expected block_align = 4, but we set 2
        let bytes = make_base_fmt_bytes(1, 2, 44_100, 176_400, 2, 16);
        let fmt = FmtChunk::from_bytes(&bytes).unwrap();
        let err = fmt.validate_format_consistency().unwrap_err();
        assert!(
            err.to_string()
                .contains("Block align 2 does not match expected 4")
        );
    }

    #[test]
    fn test_fmt_validate_rejects_byte_rate_mismatch() {
        // Expected byte_rate = sample_rate * block_align = 48_000 * 4 = 192_000
        let bytes = make_base_fmt_bytes(1, 2, 48_000, 1_000, 4, 16);
        let fmt = FmtChunk::from_bytes(&bytes).unwrap();
        let err = fmt.validate_format_consistency().unwrap_err();
        assert!(
            err.to_string()
                .contains("Byte rate 1000 does not match expected 192000")
        );
    }

    #[test]
    fn test_fmt_validate_rejects_non_byte_aligned_bits() {
        let bytes = make_base_fmt_bytes(1, 1, 44_100, 132_300, 3, 12);
        let fmt = FmtChunk::from_bytes(&bytes).unwrap();
        let err = fmt.validate_format_consistency().unwrap_err();
        assert!(
            err.to_string()
                .contains("Bits per sample 12 is not byte-aligned")
        );
    }

    #[test]
    fn test_fmt_validate_rejects_excess_channels() {
        // 300 channels exceeds the 256-channel guardrail
        let channels = 300u16;
        let bits_per_sample = 16u16;
        let bytes_per_sample = bits_per_sample / 8; // 2
        let block_align = channels * bytes_per_sample; // 600
        let sample_rate = 44_100u32;
        let byte_rate = sample_rate * block_align as u32; // 26_460_000
        let bytes = make_base_fmt_bytes(
            1,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        );
        let fmt = FmtChunk::from_bytes(&bytes).unwrap();
        let err = fmt.validate_format_consistency().unwrap_err();
        assert!(err.to_string().contains("Too many channels"));
    }
}
