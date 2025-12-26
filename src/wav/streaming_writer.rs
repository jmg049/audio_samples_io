//! Streaming WAV file writer for memory-efficient audio encoding.
//!
//! This module provides `StreamedWavWriter`, a streaming writer that writes WAV headers
//! on construction and encodes audio data incrementally, enabling writing of large files
//! without buffering all audio in memory.

use std::io::{Seek, SeekFrom, Write};

use audio_samples::{AudioSample, AudioSamples, ConvertTo, I24};

use crate::{
    error::{AudioIOError, AudioIOResult},
    traits::{AudioStreamWrite, AudioStreamWriter},
    types::ValidatedSampleType,
    wav::FormatCode,
};

/// A streaming WAV file writer that encodes audio data incrementally.
///
/// Unlike `write_wav()` which requires all audio data in memory, `StreamedWavWriter`
/// writes headers on construction and encodes audio frames as they're provided.
/// This is ideal for:
/// - Writing files larger than available memory
/// - Real-time recording applications
/// - Streaming to network destinations implementing `Write + Seek`
///
/// # Sample Type
///
/// The writer is configured with a target sample type at construction using one of:
/// - [`StreamedWavWriter::new_i16()`] for 16-bit PCM
/// - [`StreamedWavWriter::new_i24()`] for 24-bit PCM
/// - [`StreamedWavWriter::new_i32()`] for 32-bit PCM
/// - [`StreamedWavWriter::new_f32()`] for 32-bit float
/// - [`StreamedWavWriter::new_f64()`] for 64-bit float
///
/// Input samples are automatically converted to the target type.
///
/// # Finalization
///
/// WAV files require header updates with final size information. Always call
/// [`finalize()`](AudioStreamWriter::finalize) when done writing:
///
/// ```no_run
/// use audio_samples_io::wav::StreamedWavWriter;
/// use audio_samples_io::traits::AudioStreamWriter;
/// use std::fs::File;
/// use std::io::BufWriter;
///
/// let file = BufWriter::new(File::create("output.wav")?);
/// let mut writer = StreamedWavWriter::new_f32(file, 2, 44100)?;
///
/// // Write audio frames...
///
/// writer.finalize()?; // Updates headers with final sizes
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[derive(Debug)]
pub struct StreamedWavWriter<W: Write + Seek> {
    /// The underlying writer
    writer: W,
    /// Number of channels
    channels: u16,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Target sample type for encoding
    sample_type: ValidatedSampleType,
    /// Bytes per sample in output (kept for potential future use)
    #[allow(dead_code)]
    bytes_per_sample: u16,
    /// Block align (bytes per frame)
    block_align: u16,
    /// Number of frames written
    frames_written: usize,
    /// Total bytes of audio data written
    data_bytes_written: u64,
    /// Offset where RIFF size field is located (for backpatching)
    riff_size_offset: u64,
    /// Offset where data chunk size field is located (for backpatching)
    data_size_offset: u64,
    /// Whether finalize() has been called
    finalized: bool,
}

impl<W: Write + Seek> StreamedWavWriter<W> {
    /// Create a new streaming WAV writer for 16-bit PCM output.
    pub fn new_i16(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new_with_sample_type(writer, channels, sample_rate, ValidatedSampleType::I16)
    }

    /// Create a new streaming WAV writer for 24-bit PCM output.
    pub fn new_i24(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new_with_sample_type(writer, channels, sample_rate, ValidatedSampleType::I24)
    }

    /// Create a new streaming WAV writer for 32-bit PCM output.
    pub fn new_i32(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new_with_sample_type(writer, channels, sample_rate, ValidatedSampleType::I32)
    }

    /// Create a new streaming WAV writer for 32-bit float output.
    pub fn new_f32(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new_with_sample_type(writer, channels, sample_rate, ValidatedSampleType::F32)
    }

    /// Create a new streaming WAV writer for 64-bit float output.
    pub fn new_f64(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new_with_sample_type(writer, channels, sample_rate, ValidatedSampleType::F64)
    }

    /// Create a new streaming WAV writer with explicit sample type.
    fn new_with_sample_type(
        mut writer: W,
        channels: u16,
        sample_rate: u32,
        sample_type: ValidatedSampleType,
    ) -> AudioIOResult<Self> {
        if channels == 0 {
            return Err(AudioIOError::corrupted_data_simple(
                "Invalid channel count",
                "Channel count must be at least 1",
            ));
        }

        let bytes_per_sample = sample_type.bytes_per_sample() as u16;
        let block_align = channels * bytes_per_sample;

        // Determine format parameters
        let use_extensible = Self::needs_extensible_format(channels, sample_type);
        let _fmt_chunk_size: u32 = if use_extensible { 40 } else { 16 };

        // Write RIFF header with placeholder size
        writer.write_all(b"RIFF")?;
        let riff_size_offset = writer.stream_position()?;
        writer.write_all(&0u32.to_le_bytes())?; // Placeholder - will be updated in finalize()
        writer.write_all(b"WAVE")?;

        // Write FMT chunk
        if use_extensible {
            Self::write_extensible_fmt(&mut writer, channels, sample_rate, sample_type)?;
        } else {
            Self::write_base_fmt(&mut writer, channels, sample_rate, sample_type)?;
        }

        // Write DATA chunk header with placeholder size
        writer.write_all(b"data")?;
        let data_size_offset = writer.stream_position()?;
        writer.write_all(&0u32.to_le_bytes())?; // Placeholder - will be updated in finalize()

        // Calculate where audio data starts (for potential future seeking)
        let _data_start_offset = writer.stream_position()?;

        Ok(StreamedWavWriter {
            writer,
            channels,
            sample_rate,
            sample_type,
            bytes_per_sample,
            block_align,
            frames_written: 0,
            data_bytes_written: 0,
            riff_size_offset,
            data_size_offset,
            finalized: false,
        })
    }

    /// Check if extensible format is needed.
    const fn needs_extensible_format(channels: u16, sample_type: ValidatedSampleType) -> bool {
        // Use extensible format for more than 2 channels or non-standard bit depths
        channels > 2
            || matches!(
                sample_type,
                ValidatedSampleType::I24 | ValidatedSampleType::F64
            )
    }

    /// Write standard 16-byte FMT chunk.
    fn write_base_fmt(
        writer: &mut W,
        channels: u16,
        sample_rate: u32,
        sample_type: ValidatedSampleType,
    ) -> AudioIOResult<()> {
        let format_code = Self::sample_type_to_format(sample_type);
        let bits_per_sample = sample_type.bits_per_sample();
        let bytes_per_sample = sample_type.bytes_per_sample() as u16;
        let block_align = channels * bytes_per_sample;
        let byte_rate = sample_rate * block_align as u32;

        writer.write_all(b"fmt ")?;
        writer.write_all(&16u32.to_le_bytes())?; // Chunk size
        writer.write_all(&format_code.as_u16().to_le_bytes())?;
        writer.write_all(&channels.to_le_bytes())?;
        writer.write_all(&sample_rate.to_le_bytes())?;
        writer.write_all(&byte_rate.to_le_bytes())?;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&bits_per_sample.to_le_bytes())?;

        Ok(())
    }

    /// Write 40-byte extensible FMT chunk.
    fn write_extensible_fmt(
        writer: &mut W,
        channels: u16,
        sample_rate: u32,
        sample_type: ValidatedSampleType,
    ) -> AudioIOResult<()> {
        let format_code = Self::sample_type_to_format(sample_type);
        let bits_per_sample = sample_type.bits_per_sample();
        let bytes_per_sample = sample_type.bytes_per_sample() as u16;
        let block_align = channels * bytes_per_sample;
        let byte_rate = sample_rate * block_align as u32;

        // Windows standard speaker position channel masks
        let channel_mask: u32 = match channels {
            1 => 0x4,   // SPEAKER_FRONT_CENTER
            2 => 0x3,   // SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT
            3 => 0x7,   // FRONT_LEFT | FRONT_RIGHT | FRONT_CENTER
            4 => 0x33,  // FRONT_LEFT | FRONT_RIGHT | BACK_LEFT | BACK_RIGHT
            5 => 0x37,  // 4.0 + FRONT_CENTER
            6 => 0x3F,  // 5.0 + LFE (5.1)
            7 => 0x13F, // 5.1 + BACK_CENTER
            8 => 0x63F, // 5.1 + SIDE_LEFT | SIDE_RIGHT (7.1)
            _ => {
                if channels < 32 {
                    (1u32 << channels) - 1
                } else {
                    0xFFFFFFFF
                }
            }
        };

        // FMT chunk header
        writer.write_all(b"fmt ")?;
        writer.write_all(&40u32.to_le_bytes())?;

        // Base FMT chunk data (16 bytes)
        writer.write_all(&FormatCode::Extensible.as_u16().to_le_bytes())?;
        writer.write_all(&channels.to_le_bytes())?;
        writer.write_all(&sample_rate.to_le_bytes())?;
        writer.write_all(&byte_rate.to_le_bytes())?;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&bits_per_sample.to_le_bytes())?;

        // Extension size (2 bytes)
        writer.write_all(&22u16.to_le_bytes())?;

        // Extended format info (22 bytes)
        writer.write_all(&bits_per_sample.to_le_bytes())?; // Valid bits per sample
        writer.write_all(&channel_mask.to_le_bytes())?;

        // Sub-format GUID (16 bytes)
        let mut sub_format = [0u8; 16];
        sub_format[0..2].copy_from_slice(&format_code.as_u16().to_le_bytes());
        sub_format[2..16].copy_from_slice(&[
            0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71, 0x00, 0x00,
        ]);
        writer.write_all(&sub_format)?;

        Ok(())
    }

    /// Convert sample type to format code.
    const fn sample_type_to_format(sample_type: ValidatedSampleType) -> FormatCode {
        match sample_type {
            ValidatedSampleType::I16 | ValidatedSampleType::I24 | ValidatedSampleType::I32 => {
                FormatCode::Pcm
            }
            ValidatedSampleType::F32 | ValidatedSampleType::F64 => FormatCode::IeeeFloat,
        }
    }

    /// Get the target sample type this writer was configured with.
    pub const fn target_sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    /// Write raw audio bytes directly (for advanced use cases).
    ///
    /// The caller is responsible for ensuring the bytes are in the correct
    /// format (little-endian, interleaved, matching the configured sample type).
    pub fn write_raw_bytes(&mut self, bytes: &[u8]) -> AudioIOResult<usize> {
        if self.finalized {
            return Err(AudioIOError::corrupted_data_simple(
                "Cannot write to finalized stream",
                "Call write_frames before finalize()",
            ));
        }

        let frame_bytes = self.block_align as usize;
        if !bytes.len().is_multiple_of(frame_bytes) {
            return Err(AudioIOError::corrupted_data_simple(
                "Byte count must be a multiple of frame size",
                format!(
                    "Got {} bytes, frame size is {} bytes",
                    bytes.len(),
                    frame_bytes
                ),
            ));
        }

        self.writer.write_all(bytes)?;
        let frames = bytes.len() / frame_bytes;
        self.frames_written += frames;
        self.data_bytes_written += bytes.len() as u64;

        Ok(frames)
    }
}

// Implement AudioStreamWriter (object-safe trait)
impl<W: Write + Seek> AudioStreamWriter for StreamedWavWriter<W> {
    fn flush(&mut self) -> AudioIOResult<()> {
        self.writer.flush()?;
        Ok(())
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        if self.finalized {
            return Ok(()); // Already finalized, idempotent
        }

        // Add padding byte if data size is odd
        if self.data_bytes_written % 2 == 1 {
            self.writer.write_all(&[0])?;
        }

        // Calculate final sizes
        let data_size = self.data_bytes_written as u32;
        let padded_data_size = if self.data_bytes_written % 2 == 1 {
            self.data_bytes_written + 1
        } else {
            self.data_bytes_written
        };

        // Calculate FMT chunk size
        let use_extensible = Self::needs_extensible_format(self.channels, self.sample_type);
        let fmt_chunk_size: u64 = if use_extensible { 40 } else { 16 };
        let fmt_total_size = 8 + fmt_chunk_size;

        // RIFF size = 4 (WAVE) + fmt total size + 8 (data header) + padded data size
        let riff_size = 4 + fmt_total_size + 8 + padded_data_size;

        // Remember current position
        let current_pos = self.writer.stream_position()?;

        // Backpatch RIFF size
        self.writer.seek(SeekFrom::Start(self.riff_size_offset))?;
        self.writer.write_all(&(riff_size as u32).to_le_bytes())?;

        // Backpatch data size
        self.writer.seek(SeekFrom::Start(self.data_size_offset))?;
        self.writer.write_all(&data_size.to_le_bytes())?;

        // Restore position and flush
        self.writer.seek(SeekFrom::Start(current_pos))?;
        self.writer.flush()?;

        self.finalized = true;
        Ok(())
    }

    fn is_finalized(&self) -> bool {
        self.finalized
    }

    fn frames_written(&self) -> usize {
        self.frames_written
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn num_channels(&self) -> u16 {
        self.channels
    }
}

// Implement AudioStreamWrite (generic trait)
impl<W: Write + Seek> AudioStreamWrite for StreamedWavWriter<W> {
    fn write_frames<T>(&mut self, samples: &AudioSamples<'_, T>) -> AudioIOResult<usize>
    where
        T: AudioSample + 'static,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        if self.finalized {
            return Err(AudioIOError::corrupted_data_simple(
                "Cannot write to finalized stream",
                "Call write_frames before finalize()",
            ));
        }

        // Validate channel count
        let input_channels = samples.num_channels();
        if input_channels != self.channels as usize {
            return Err(AudioIOError::corrupted_data_simple(
                "Channel count mismatch",
                format!(
                    "Writer configured for {} channels, got {} channels",
                    self.channels, input_channels
                ),
            ));
        }

        let frames_per_channel = samples.samples_per_channel();
        if frames_per_channel == 0 {
            return Ok(0);
        }

        // Get interleaved samples and convert to target format
        let interleaved = samples.data.as_interleaved_vec();

        // Convert and write based on target sample type
        let bytes_written = match self.sample_type {
            ValidatedSampleType::I16 => self.write_samples_as::<T, i16>(&interleaved)?,
            ValidatedSampleType::I24 => self.write_samples_as::<T, I24>(&interleaved)?,
            ValidatedSampleType::I32 => self.write_samples_as::<T, i32>(&interleaved)?,
            ValidatedSampleType::F32 => self.write_samples_as::<T, f32>(&interleaved)?,
            ValidatedSampleType::F64 => self.write_samples_as::<T, f64>(&interleaved)?,
        };

        self.frames_written += frames_per_channel;
        self.data_bytes_written += bytes_written as u64;

        Ok(frames_per_channel)
    }
}

impl<W: Write + Seek> StreamedWavWriter<W> {
    /// Convert samples from type T to type U and write to the underlying writer.
    fn write_samples_as<T, U>(&mut self, samples: &[T]) -> AudioIOResult<usize>
    where
        T: AudioSample + 'static,
        U: AudioSample + 'static,
        T: ConvertTo<U>,
    {
        // Stream in chunks to avoid large allocations
        const TARGET_CHUNK_BYTES: usize = 256 * 1024; // 256 KiB
        let bytes_per_sample = U::BYTES;
        let samples_per_chunk = TARGET_CHUNK_BYTES / bytes_per_sample;
        let samples_per_chunk = samples_per_chunk.max(self.channels as usize);

        let mut buf = vec![0u8; samples_per_chunk * bytes_per_sample];
        let mut total_bytes = 0usize;

        for chunk in samples.chunks(samples_per_chunk) {
            let mut write_idx = 0;
            for sample in chunk {
                let converted: U = sample.convert_to();
                let bytes = converted.to_le_bytes();
                let dst = &mut buf[write_idx..write_idx + bytes_per_sample];
                dst.copy_from_slice(bytes.as_ref());
                write_idx += bytes_per_sample;
            }

            self.writer.write_all(&buf[..write_idx])?;
            total_bytes += write_idx;
        }

        Ok(total_bytes)
    }
}

/// Drop implementation to warn if not finalized.
///
/// Note: We intentionally don't auto-finalize on drop because:
/// 1. finalize() can fail, and we can't return errors from drop
/// 2. Auto-finalize might hide bugs where the user forgot to finalize
/// 3. The WAV file will be invalid without proper headers anyway
impl<W: Write + Seek> Drop for StreamedWavWriter<W> {
    fn drop(&mut self) {
        if !self.finalized && self.frames_written > 0 {
            // Log a warning in debug builds
            #[cfg(debug_assertions)]
            eprintln!(
                "Warning: StreamedWavWriter dropped without calling finalize(). \
                 The output file may have invalid headers."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::num::NonZeroU32;

    #[test]
    fn test_streaming_writer_basic() {
        let mut buffer = Vec::new();

        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer =
                StreamedWavWriter::new_f32(cursor, 2, 44100).expect("Failed to create writer");

            // Create test audio
            let sample_rate = NonZeroU32::new(44100).unwrap();
            let samples = AudioSamples::<f32>::zeros_multi(2, 1024, sample_rate);

            let frames = writer.write_frames(&samples).expect("Write failed");
            assert_eq!(frames, 1024);
            assert_eq!(writer.frames_written(), 1024);

            writer.finalize().expect("Finalize failed");
            assert!(writer.is_finalized());
        }

        // Verify buffer has valid WAV structure (writer is dropped, borrow released)
        assert!(buffer.len() > 44); // At least header size
        assert_eq!(&buffer[0..4], b"RIFF");
        assert_eq!(&buffer[8..12], b"WAVE");
    }

    #[test]
    fn test_streaming_writer_multiple_writes() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        let mut writer =
            StreamedWavWriter::new_i16(cursor, 1, 22050).expect("Failed to create writer");

        // Write multiple chunks
        let sample_rate = NonZeroU32::new(22050).unwrap();
        let chunk1 = AudioSamples::<f32>::zeros_mono(512, sample_rate);
        let chunk2 = AudioSamples::<f32>::zeros_mono(512, sample_rate);

        writer.write_frames(&chunk1).expect("Write 1 failed");
        writer.write_frames(&chunk2).expect("Write 2 failed");

        assert_eq!(writer.frames_written(), 1024);

        writer.finalize().expect("Finalize failed");
    }

    #[test]
    fn test_streaming_writer_idempotent_finalize() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        let mut writer =
            StreamedWavWriter::new_f32(cursor, 1, 44100).expect("Failed to create writer");

        writer.finalize().expect("First finalize failed");
        writer.finalize().expect("Second finalize should succeed");

        assert!(writer.is_finalized());
    }

    #[test]
    fn test_streaming_writer_channel_mismatch() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        let mut writer =
            StreamedWavWriter::new_f32(cursor, 2, 44100).expect("Failed to create writer");

        // Try to write mono audio to stereo writer
        let sample_rate = NonZeroU32::new(44100).unwrap();
        let mono_samples = AudioSamples::<f32>::zeros_mono(1024, sample_rate);
        let result = writer.write_frames(&mono_samples);

        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_writer_write_after_finalize() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        let mut writer =
            StreamedWavWriter::new_f32(cursor, 1, 44100).expect("Failed to create writer");

        writer.finalize().expect("Finalize failed");

        let sample_rate = NonZeroU32::new(44100).unwrap();
        let samples = AudioSamples::<f32>::zeros_mono(1024, sample_rate);
        let result = writer.write_frames(&samples);

        assert!(result.is_err());
    }
}
