//! FLAC file implementation with AudioFile trait support.
//!
//! This module provides `FlacFile`, the main entry point for reading and writing
//! FLAC files. It follows the same patterns as `WavFile`:
//!
//! - Uses `AudioDataSource` for backing (owned, memory-mapped, or borrowed)
//! - Implements `AudioFile`, `AudioFileMetadata`, `AudioFileRead`, `AudioFileWrite`
//! - Uses `ValidatedSampleType` and the `ConvertTo` traits for sample conversion
//! - Delegates to `DataChunk`-style abstractions for raw sample access

use audio_samples::{AudioSample, AudioSamples, ConvertTo, I24};
use core::fmt::{Display, Formatter, Result as FmtResult};
use memmap2::MmapOptions;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    num::NonZeroU32,
    ops::Range,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    MAX_MMAP_SIZE,
    error::{AudioIOError, AudioIOResult, ErrorPosition},
    flac::frame::encode_frame,
    traits::{AudioFile, AudioFileMetadata, AudioFileRead, AudioFileWrite, AudioInfoMarker},
    types::{AudioDataSource, BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType},
};

use super::{
    CompressionLevel,
    constants::FLAC_MARKER,
    data::DecodedAudio,
    error::FlacError,
    frame::decode_frame,
    metadata::{MetadataBlock, MetadataBlockType, StreamInfo},
};

/// Maximum FLAC file size we'll handle (4GB)
pub(crate) const MAX_FLAC_SIZE: u64 = 4 * 1024 * 1024 * 1024;

/// FLAC-specific file information.
#[derive(Debug, Clone)]
pub struct FlacFileInfo {
    /// Available metadata blocks
    pub metadata_blocks: Vec<MetadataBlockType>,
    /// MD5 signature of uncompressed audio (if present)
    pub md5_signature: Option<[u8; 16]>,
    /// Minimum block size in samples
    pub min_block_size: u16,
    /// Maximum block size in samples  
    pub max_block_size: u16,
    /// Minimum frame size in bytes (0 = unknown)
    pub min_frame_size: u32,
    /// Maximum frame size in bytes (0 = unknown)
    pub max_frame_size: u32,
}

impl Display for FlacFileInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "FLAC File Info:")?;
        writeln!(
            f,
            "Block Size: {}-{} samples",
            self.min_block_size, self.max_block_size
        )?;
        if self.min_frame_size > 0 {
            writeln!(
                f,
                "Frame Size: {}-{} bytes",
                self.min_frame_size, self.max_frame_size
            )?;
        }
        writeln!(f, "Metadata Blocks: {:?}", self.metadata_blocks)?;
        if let Some(md5) = &self.md5_signature {
            if md5.iter().any(|&b| b != 0) {
                writeln!(f, "MD5: {:02x?}", md5)?;
            }
        }
        Ok(())
    }
}

impl AudioInfoMarker for FlacFileInfo {}

/// High-level FLAC file representation.
///
/// Mirrors the design of `WavFile`:
/// - Stores `AudioDataSource` for file data
/// - Parses metadata on open
/// - Decodes frames lazily on `read()`
#[derive(Debug)]
pub struct FlacFile<'a> {
    /// Data source (owned or memory-mapped)
    data_source: AudioDataSource<'a>,
    /// File path
    file_path: PathBuf,
    /// Parsed STREAMINFO
    stream_info: StreamInfo,
    /// Metadata block info (types and positions)
    metadata_blocks: Vec<(MetadataBlockType, Range<usize>)>,
    /// Byte offset where audio frames start
    audio_data_offset: usize,
    /// Validated sample type
    sample_type: ValidatedSampleType,
    /// Total samples (cached from STREAMINFO)
    total_samples: u64,
}

impl<'a> FlacFile<'a> {
    /// Get the STREAMINFO metadata.
    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }

    /// Get the audio frames data slice.
    fn audio_data(&self) -> &[u8] {
        &self.data_source.as_bytes()[self.audio_data_offset..]
    }

    /// Decode all frames and return as `DecodedAudio`.
    ///
    /// This is analogous to WAV's `DataChunk` - it holds the decoded samples
    /// and provides generic conversion via `read_samples<T>()`.
    fn decode_all_frames(&self) -> Result<DecodedAudio, FlacError> {
        let num_channels = self.stream_info.channels as usize;
        let total_frames = if num_channels > 0 {
            self.stream_info.total_samples as usize
        } else {
            0
        };

        // Pre-allocate channel buffers
        let mut channels: Vec<Vec<i32>> = (0..num_channels)
            .map(|_| Vec::with_capacity(total_frames))
            .collect();

        let data = self.audio_data();
        let mut offset = 0;

        while offset < data.len() {
            // Try to decode a frame
            let frame_data = &data[offset..];

            // Skip to next sync code if needed
            if frame_data.len() < 2 {
                break;
            }

            // Look for sync code 0xFFF8 or 0xFFF9
            if frame_data[0] != 0xFF || (frame_data[1] & 0xFC) != 0xF8 {
                offset += 1;
                continue;
            }

            match decode_frame(
                frame_data,
                self.stream_info.sample_rate,
                self.stream_info.bits_per_sample,
                self.stream_info.channels,
            ) {
                Ok((frame, bytes_consumed)) => {
                    // Extract samples from decoded frame (already de-correlated)
                    let frame_channels = frame.into_channel_samples();

                    for (ch_idx, samples) in frame_channels.into_iter().enumerate() {
                        if ch_idx < channels.len() {
                            channels[ch_idx].extend(samples);
                        }
                    }

                    offset += bytes_consumed;
                }
                Err(FlacError::InvalidFrameSync { .. }) => {
                    // Not a valid frame, skip byte
                    offset += 1;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(DecodedAudio::new(
            channels,
            self.stream_info.bits_per_sample,
            self.stream_info.sample_rate,
        ))
    }
}

impl<'a> AudioFileMetadata for FlacFile<'a> {
    fn open_metadata<P: AsRef<Path>>(path: P) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        Self::open_with_options(path, OpenOptions::default())
    }

    fn base_info(&self) -> AudioIOResult<BaseAudioInfo> {
        let si = &self.stream_info;
        let channels = si.channels as u16;
        let bits_per_sample = si.bits_per_sample as u16;
        let bytes_per_sample = (bits_per_sample + 7) / 8;
        let block_align = channels * bytes_per_sample;
        let byte_rate = si.sample_rate * block_align as u32;

        let total_samples = si.total_samples as usize;
        let frames = if channels > 0 {
            total_samples / channels as usize
        } else {
            0
        };
        let duration = Duration::from_secs_f64(frames as f64 / si.sample_rate as f64);

        Ok(BaseAudioInfo::new(
            si.sample_rate,
            channels,
            bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            total_samples,
            duration,
            FileType::FLAC,
            self.sample_type.into(),
        ))
    }

    #[allow(refining_impl_trait)]
    fn specific_info(&self) -> FlacFileInfo {
        let si = &self.stream_info;
        FlacFileInfo {
            metadata_blocks: self.metadata_blocks.iter().map(|(t, _)| *t).collect(),
            md5_signature: Some(si.md5_signature),
            min_block_size: si.min_block_size,
            max_block_size: si.max_block_size,
            min_frame_size: si.min_frame_size,
            max_frame_size: si.max_frame_size,
        }
    }

    fn file_type(&self) -> FileType {
        FileType::FLAC
    }

    fn file_path(&self) -> &Path {
        self.file_path.as_path()
    }

    fn total_samples(&self) -> usize {
        self.total_samples as usize
    }

    fn duration(&self) -> AudioIOResult<Duration> {
        self.base_info().map(|info| info.duration)
    }

    fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    fn num_channels(&self) -> u16 {
        self.stream_info.channels as u16
    }
}

impl<'a> AudioFile for FlacFile<'a> {
    fn open_with_options<P: AsRef<Path>>(fp: P, options: OpenOptions) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        let path = fp.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_size = file.metadata()?.len();

        if file_size > MAX_FLAC_SIZE {
            return Err(AudioIOError::corrupted_data_simple(
                "File too large",
                format!(
                    "File size {} exceeds maximum {} bytes",
                    file_size, MAX_FLAC_SIZE
                ),
            ));
        }

        let use_mmap = options.use_memory_map && file_size <= MAX_MMAP_SIZE;

        let data_source: AudioDataSource<'a> = if use_mmap {
            AudioDataSource::MemoryMapped(unsafe { MmapOptions::new().map(&file)? })
        } else {
            let mut buf_reader = BufReader::new(file);
            let mut bytes = Vec::new();
            buf_reader.read_to_end(&mut bytes)?;
            AudioDataSource::Owned(bytes)
        };

        let bytes = data_source.as_bytes();

        // Validate FLAC marker
        if bytes.len() < 4 {
            return Err(AudioIOError::corrupted_data(
                "File too small to be a valid FLAC file",
                format!("File size: {}", bytes.len()),
                ErrorPosition::new(0).with_description("start of file"),
            ));
        }

        let marker: [u8; 4] = bytes[0..4].try_into().map_err(|_| {
            AudioIOError::corrupted_data_simple("Cannot read FLAC marker", "Insufficient data")
        })?;

        if marker != FLAC_MARKER {
            return Err(AudioIOError::FlacError(FlacError::invalid_marker(marker)));
        }

        // Parse metadata blocks
        let mut offset = 4;
        let mut stream_info: Option<StreamInfo> = None;
        let mut metadata_blocks = Vec::new();
        let mut is_last = false;

        while !is_last && offset < bytes.len() {
            if offset + 4 > bytes.len() {
                return Err(AudioIOError::corrupted_data(
                    "Truncated metadata block header",
                    format!("Offset: {}", offset),
                    ErrorPosition::new(offset),
                ));
            }

            let header_byte = bytes[offset];
            is_last = (header_byte & 0x80) != 0;
            let block_type = header_byte & 0x7F;
            let block_size =
                u32::from_be_bytes([0, bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
                    as usize;

            let block_type_enum = MetadataBlockType::from_byte(block_type);

            // Check for reserved/invalid types
            if matches!(block_type_enum, MetadataBlockType::Reserved(n) if n > 126) {
                return Err(AudioIOError::FlacError(
                    FlacError::InvalidMetadataBlockType(block_type),
                ));
            }

            let data_start = offset + 4;
            let data_end = data_start + block_size;

            if data_end > bytes.len() {
                return Err(AudioIOError::corrupted_data(
                    "Metadata block extends beyond file",
                    format!("Block type {:?}, size {}", block_type_enum, block_size),
                    ErrorPosition::new(offset),
                ));
            }

            let block_data = &bytes[data_start..data_end];

            // Parse STREAMINFO (required, must be first)
            if block_type_enum == MetadataBlockType::StreamInfo {
                stream_info =
                    Some(StreamInfo::from_bytes(block_data).map_err(AudioIOError::FlacError)?);
            }

            metadata_blocks.push((block_type_enum, data_start..data_end));
            offset = data_end;
        }

        let stream_info =
            stream_info.ok_or_else(|| AudioIOError::FlacError(FlacError::MissingStreamInfo))?;

        // Determine sample type from bits per sample
        let sample_type = match stream_info.bits_per_sample {
            1..=16 => ValidatedSampleType::I16,
            17..=24 => ValidatedSampleType::I24,
            25..=32 => ValidatedSampleType::I32,
            _ => {
                return Err(AudioIOError::FlacError(FlacError::InvalidBitsPerSample {
                    bits: stream_info.bits_per_sample,
                }));
            }
        };

        let total_samples = stream_info.total_samples;

        Ok(FlacFile {
            data_source,
            file_path: path,
            stream_info,
            metadata_blocks,
            audio_data_offset: offset,
            sample_type,
            total_samples,
        })
    }

    fn len(&self) -> u64 {
        self.data_source.len() as u64
    }
}

impl<'a> AudioFileRead<'a> for FlacFile<'a> {
    fn read<T>(&'a self) -> AudioIOResult<AudioSamples<'a, T>>
    where
        T: AudioSample + 'static,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        // Decode all frames to DecodedAudio
        let decoded = self.decode_all_frames().map_err(AudioIOError::FlacError)?;

        // Use DecodedAudio's read_samples method to get properly converted AudioSamples
        // SAFETY: FLAC spec requires non-zero sample rate, validated during parsing
        let sample_rate = NonZeroU32::new(self.stream_info.sample_rate).ok_or_else(|| {
            AudioIOError::corrupted_data_simple("Invalid sample rate", "sample rate cannot be zero")
        })?;
        decoded.read_samples::<T>(sample_rate)
    }

    fn read_into<T>(&'a self, audio: &mut AudioSamples<'a, T>) -> AudioIOResult<()>
    where
        T: AudioSample + 'static,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        // Decode all frames
        let decoded = self.decode_all_frames().map_err(AudioIOError::FlacError)?;

        let num_channels = decoded.num_channels();
        let samples_per_channel = decoded.samples_per_channel();
        let total_samples = num_channels * samples_per_channel;

        if total_samples != audio.total_samples() {
            return Err(AudioIOError::corrupted_data_simple(
                "Sample count mismatch",
                format!(
                    "FLAC has {} samples, buffer has {}",
                    total_samples,
                    audio.total_samples()
                ),
            ));
        }

        if num_channels != audio.num_channels() {
            return Err(AudioIOError::corrupted_data_simple(
                "Channel count mismatch",
                format!(
                    "FLAC has {} channels, buffer has {}",
                    num_channels,
                    audio.num_channels()
                ),
            ));
        }

        // Use DecodedAudio's conversion to get planar samples
        let planar_data = decoded.read_samples_planar::<T>()?;
        audio.replace_with_vec(planar_data).map_err(|e| e.into())
    }
}

impl<'a> AudioFileWrite for FlacFile<'a> {
    fn write<P: AsRef<Path>, T: AudioSample>(&mut self, out_fp: P) -> AudioIOResult<()>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        // Read current audio as target type
        let audio = self.read::<T>()?;

        // Write using the public function
        let file = File::create(out_fp)?;
        let writer = BufWriter::new(file);

        write_flac(writer, &audio, CompressionLevel::DEFAULT)
    }
}

/// Write AudioSamples to a FLAC file.
///
/// FLAC only supports integer PCM audio with 4-24 bits per sample.
/// - For f32/f64 input: automatically converts to 24-bit integers
/// - For i16 input: writes as 16-bit
/// - For I24 input: writes as 24-bit  
/// - For i32 input: writes as 24-bit (truncated from 32-bit)
pub fn write_flac<W: Write, T>(
    mut writer: W,
    audio: &AudioSamples<T>,
    level: CompressionLevel,
) -> AudioIOResult<()>
where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    use std::any::TypeId;

    let sample_rate = audio.sample_rate();
    let num_channels = audio.num_channels() as u8;
    let samples_per_channel = audio.samples_per_channel();
    let total_samples = audio.total_samples() as u64;

    // Validate parameters
    if num_channels == 0 || num_channels > 8 {
        return Err(AudioIOError::FlacError(FlacError::InvalidChannelCount {
            channels: num_channels,
        }));
    }

    // Determine target bit depth based on input type
    // FLAC only supports 4-24 bits per sample
    let bits_per_sample: u8 = if TypeId::of::<T>() == TypeId::of::<i16>() {
        16
    } else if TypeId::of::<T>() == TypeId::of::<I24>() {
        24
    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
        // i32 gets truncated to 24-bit for FLAC
        24
    } else {
        // f32/f64 get converted to 24-bit
        24
    };

    // Write FLAC marker
    writer.write_all(&FLAC_MARKER)?;

    // Create STREAMINFO block
    let block_size = level.block_size().min(samples_per_channel as u32);
    let sample_rate_u32 = sample_rate.get();
    let stream_info = StreamInfo {
        min_block_size: block_size as u16,
        max_block_size: block_size as u16,
        min_frame_size: 0, // Unknown until encoded
        max_frame_size: 0,
        sample_rate: sample_rate_u32,
        channels: num_channels,
        bits_per_sample,
        total_samples,
        md5_signature: [0; 16], // Will be computed
    };

    // Write STREAMINFO (last metadata block for now)
    let streaminfo_bytes = stream_info.to_bytes();
    writer.write_all(&[0x80])?; // Last block, type 0 (STREAMINFO)
    writer.write_all(&[(streaminfo_bytes.len() >> 16) as u8])?;
    writer.write_all(&[(streaminfo_bytes.len() >> 8) as u8])?;
    writer.write_all(&[streaminfo_bytes.len() as u8])?;
    writer.write_all(&streaminfo_bytes)?;

    // Convert audio samples to i32 for FLAC encoding
    // FLAC stores samples as signed integers at the target bit depth
    let interleaved_i32: Vec<i32> = if TypeId::of::<T>() == TypeId::of::<i16>() {
        // i16 -> i32: just cast, samples are already in correct range
        audio
            .to_interleaved_vec()
            .into_iter()
            .map(|s| {
                let bytes = s.to_le_bytes();
                i16::from_le_bytes([bytes.as_ref()[0], bytes.as_ref()[1]]) as i32
            })
            .collect()
    } else if TypeId::of::<T>() == TypeId::of::<I24>() {
        // I24 -> i32: extract 24-bit value
        audio
            .to_interleaved_vec()
            .into_iter()
            .map(|s| {
                let bytes = s.to_le_bytes();
                let val = i32::from_le_bytes([
                    bytes.as_ref()[0],
                    bytes.as_ref()[1],
                    bytes.as_ref()[2],
                    0,
                ]);
                // Sign-extend from 24-bit
                if val & 0x800000 != 0 {
                    val | (0xFF << 24)
                } else {
                    val
                }
            })
            .collect()
    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
        // i32 -> 24-bit: shift right by 8 bits to fit in 24 bits
        audio
            .to_interleaved_vec()
            .into_iter()
            .map(|s| {
                let bytes = s.to_le_bytes();
                let val = i32::from_le_bytes([
                    bytes.as_ref()[0],
                    bytes.as_ref()[1],
                    bytes.as_ref()[2],
                    bytes.as_ref()[3],
                ]);
                val >> 8 // Truncate to 24-bit range
            })
            .collect()
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        // f32 -> 24-bit: scale from [-1.0, 1.0] to [-8388608, 8388607]
        audio
            .to_interleaved_vec()
            .into_iter()
            .map(|s| {
                let bytes = s.to_le_bytes();
                let val = f32::from_le_bytes([
                    bytes.as_ref()[0],
                    bytes.as_ref()[1],
                    bytes.as_ref()[2],
                    bytes.as_ref()[3],
                ]);
                // Scale to 24-bit range and clamp
                let scaled = (val * 8388607.0).clamp(-8388608.0, 8388607.0);
                scaled as i32
            })
            .collect()
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        // f64 -> 24-bit: scale from [-1.0, 1.0] to [-8388608, 8388607]
        audio
            .to_interleaved_vec()
            .into_iter()
            .map(|s| {
                let bytes = s.to_le_bytes();
                let val = f64::from_le_bytes([
                    bytes.as_ref()[0],
                    bytes.as_ref()[1],
                    bytes.as_ref()[2],
                    bytes.as_ref()[3],
                    bytes.as_ref()[4],
                    bytes.as_ref()[5],
                    bytes.as_ref()[6],
                    bytes.as_ref()[7],
                ]);
                // Scale to 24-bit range and clamp
                let scaled = (val * 8388607.0).clamp(-8388608.0, 8388607.0);
                scaled as i32
            })
            .collect()
    } else {
        return Err(AudioIOError::corrupted_data_simple(
            "Unsupported sample type for FLAC encoding",
            format!("Type has {} bits", T::BITS),
        ));
    };

    // Deinterleave into per-channel vectors
    let channel_samples: Vec<Vec<i32>> = (0..num_channels as usize)
        .map(|ch| {
            interleaved_i32
                .iter()
                .skip(ch)
                .step_by(num_channels as usize)
                .copied()
                .collect()
        })
        .collect();

    // Get compression parameters from level
    let max_lpc_order = level.max_lpc_order() as usize;
    let qlp_precision = level.qlp_precision();
    let (min_partition_order, max_partition_order) = level.rice_partition_order_range();
    let try_mid_side = level.try_mid_side();
    let exhaustive_rice = level.exhaustive_rice_search();

    // Encode frames
    let mut frame_number = 0u64;
    let mut sample_offset = 0usize;

    while sample_offset < samples_per_channel {
        let frame_samples = (samples_per_channel - sample_offset).min(block_size as usize);

        // Interleave frame samples for encode_frame
        let mut interleaved = Vec::with_capacity(frame_samples * num_channels as usize);
        for i in 0..frame_samples {
            for ch in &channel_samples {
                interleaved.push(ch[sample_offset + i]);
            }
        }

        // Encode frame
        let frame_bytes = encode_frame(
            &interleaved,
            num_channels,
            frame_samples as u32,
            sample_rate_u32,
            bits_per_sample,
            frame_number,
            max_lpc_order,
            qlp_precision,
            min_partition_order,
            max_partition_order,
            try_mid_side,
            exhaustive_rice,
        )
        .map_err(AudioIOError::FlacError)?;

        writer.write_all(&frame_bytes)?;

        frame_number += 1;
        sample_offset += frame_samples;
    }

    writer.flush()?;
    Ok(())
}

impl Display for FlacFile<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "FLAC File:")?;
        writeln!(f, "  Path: {:?}", self.file_path)?;
        writeln!(f, "  Sample Rate: {} Hz", self.stream_info.sample_rate)?;
        writeln!(f, "  Channels: {}", self.stream_info.channels)?;
        writeln!(f, "  Bits per Sample: {}", self.stream_info.bits_per_sample)?;
        writeln!(f, "  Total Samples: {}", self.total_samples)?;
        writeln!(f, "  Metadata Blocks: {}", self.metadata_blocks.len())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_file_info_display() {
        let info = FlacFileInfo {
            metadata_blocks: vec![
                MetadataBlockType::StreamInfo,
                MetadataBlockType::VorbisComment,
            ],
            md5_signature: Some([0; 16]),
            min_block_size: 4096,
            max_block_size: 4096,
            min_frame_size: 0,
            max_frame_size: 0,
        };
        let display = format!("{}", info);
        assert!(display.contains("FLAC File Info"));
        assert!(display.contains("4096"));
    }
}
