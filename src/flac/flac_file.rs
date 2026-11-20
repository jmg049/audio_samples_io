//! FLAC file implementation with AudioFile trait support.
//!
//! This module provides `FlacFile`, the main entry point for reading and writing
//! FLAC files. It follows the same patterns as `WavFile`:
//!
//! - Uses `AudioDataSource` for backing (owned, memory-mapped, or borrowed)
//! - Implements `AudioFile`, `AudioFileMetadata`, `AudioFileRead`, `AudioFileWrite`
//! - Uses `ValidatedSampleType` and the `ConvertTo` traits for sample conversion
//! - Delegates to `DataChunk`-style abstractions for raw sample access

use core::fmt::{Display, Formatter, Result as FmtResult};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    num::NonZeroU32,
    ops::Range,
    path::{Path, PathBuf},
    time::Duration,
};

use audio_samples::{
    AudioSamples, I24, SampleType,
    traits::{ConvertTo, StandardSample},
};
use memmap2::MmapOptions;

use super::{
    CompressionLevel,
    constants::FLAC_MARKER,
    data::DecodedAudio,
    error::FlacError,
    frame::{decode_frame_into_channels, decode_frame_into_scratch},
    metadata::{MetadataBlockType, StreamInfo},
};
use crate::{
    MAX_MMAP_SIZE,
    error::{AudioIOError, AudioIOResult, ErrorPosition},
    flac::{
        encoder::{EncoderConfig, FrameEncoder},
        md5::Md5,
    },
    traits::{AudioFile, AudioFileMetadata, AudioFileRead, AudioFileWrite, AudioInfoMarker},
    types::{AudioDataSource, BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType, channel_frame_view},
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
        writeln!(f, "Block Size: {}-{} samples", self.min_block_size, self.max_block_size)?;
        if self.min_frame_size > 0 {
            writeln!(f, "Frame Size: {}-{} bytes", self.min_frame_size, self.max_frame_size)?;
        }
        writeln!(f, "Metadata Blocks: {:?}", self.metadata_blocks)?;
        if let Some(md5) = &self.md5_signature
            && md5.iter().any(|&b| b != 0)
        {
            writeln!(f, "MD5: {:02x?}", md5)?;
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
    pub const fn stream_info(&self) -> &StreamInfo {
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
        let mut channels: Vec<Vec<i32>> = (0..num_channels).map(|_| Vec::with_capacity(total_frames)).collect();

        // Scratch buffers for stereo decorrelation (reused across frames, never reallocated
        // after the first frame grows them to block_size).
        let block_size_hint = self.stream_info.max_block_size as usize;
        let mut scratch = (Vec::with_capacity(block_size_hint), Vec::with_capacity(block_size_hint));

        let data = self.audio_data();
        let mut offset = 0;

        while offset < data.len() {
            let frame_data = &data[offset..];

            if frame_data.len() < 2 {
                break;
            }

            // Look for sync code 0xFFF8 or 0xFFF9
            if frame_data[0] != 0xFF || (frame_data[1] & 0xFC) != 0xF8 {
                offset += 1;
                continue;
            }

            match decode_frame_into_channels(
                frame_data,
                self.stream_info.sample_rate,
                self.stream_info.bits_per_sample,
                self.stream_info.channels,
                &mut channels,
                &mut scratch,
            ) {
                Ok(bytes_consumed) => {
                    offset += bytes_consumed;
                },
                Err(FlacError::InvalidFrameSync { .. }) => {
                    offset += 1;
                },
                Err(e) => return Err(e),
            }
        }

        Ok(DecodedAudio::new(
            channels,
            self.stream_info.bits_per_sample,
            self.stream_info.sample_rate,
        ))
    }

    /// Decode all frames into planar storage: channel `c` occupies
    /// `out[c * total_spc .. (c + 1) * total_spc]`.
    ///
    /// Returns the samples per channel written, or `None` when the stream holds more
    /// than STREAMINFO's `total_spc` (nothing beyond `out` is written).
    ///
    /// # Safety
    ///
    /// `out` must be valid for writes of `channels * total_spc` values of `T`.
    pub(crate) unsafe fn decode_planar_into<T>(&self, out: *mut T, total_spc: usize) -> Result<Option<usize>, FlacError>
    where
        T: StandardSample + 'static,
    {
        use audio_samples::I24;

        let num_channels = self.stream_info.channels as usize;
        let bits = self.stream_info.bits_per_sample;
        let block_size_hint = self.stream_info.max_block_size as usize;

        let data = self.audio_data();
        // Pages of a mapped file that have been decoded go back at once, so that the resident
        // size does not grow by the size of the file.
        let offset = self.audio_data_offset;
        let source = &self.data_source;
        let release = move |range: std::ops::Range<usize>| source.release(offset + range.start..offset + range.end);
        // SAFETY: the caller's guarantee on `out` is the one this call needs.
        if super::parallel::worth_decoding_in_parallel(data)
            && unsafe { super::parallel::decode_planar_parallel(data, &self.stream_info, out, total_spc, &release) }
        {
            return Ok(Some(total_spc));
        }

        let mut scratch: Vec<Vec<i32>> = (0..num_channels).map(|_| Vec::with_capacity(block_size_hint)).collect();

        let convert = |s: i32| -> T {
            match bits {
                1..=8 => T::convert_from(((s) << (16 - bits)) as i16),
                9..=16 => T::convert_from(s as i16),
                17..=24 => T::convert_from(I24::wrapping_from_i32(s)),
                _ => T::convert_from(s),
            }
        };

        const RELEASE_BYTES: usize = 32 * 1024;
        let mut released = 0usize;
        let mut offset = 0;
        let mut written = 0usize;

        while offset < data.len() {
            let frame_data = &data[offset..];
            if frame_data.len() < 2 {
                break;
            }
            if frame_data[0] != 0xFF || (frame_data[1] & 0xFC) != 0xF8 {
                offset += 1;
                continue;
            }
            match decode_frame_into_scratch(
                frame_data,
                self.stream_info.sample_rate,
                self.stream_info.bits_per_sample,
                &mut scratch,
            ) {
                Ok(bytes_consumed) => {
                    let block = scratch[0].len();
                    if written + block > total_spc {
                        return Ok(None);
                    }
                    if offset - released >= RELEASE_BYTES {
                        release(released..offset);
                        released = offset;
                    }
                    for (ch, samples) in scratch.iter().enumerate() {
                        // SAFETY: `written + block <= total_spc`, so the range lies in
                        // channel `ch`'s part of `out`.
                        let dst = unsafe { out.add(ch * total_spc + written) };
                        for (i, &s) in samples.iter().enumerate() {
                            unsafe { dst.add(i).write(convert(s)) };
                        }
                    }
                    written += block;
                    offset += bytes_consumed;
                },
                Err(FlacError::InvalidFrameSync { .. }) => {
                    offset += 1;
                },
                Err(e) => return Err(e),
            }
        }

        Ok(Some(written))
    }

    /// Decode all frames, converting directly to type T frame-by-frame.
    ///
    /// Each frame is decoded into small per-channel scratch buffers, which stay in the L1 or L2
    /// cache while the predictor is restored, and is then converted to `T` and appended to the
    /// per-channel output. Decoding and conversion take one pass over the samples.
    fn decode_all_frames_typed<T>(&self) -> Result<Vec<T>, FlacError>
    where
        T: StandardSample + 'static,
    {
        use audio_samples::I24;

        let num_channels = self.stream_info.channels as usize;
        let total_spc = self.stream_info.total_samples as usize;
        let bits = self.stream_info.bits_per_sample;
        let block_size_hint = self.stream_info.max_block_size as usize;

        if num_channels == 0 || total_spc == 0 {
            return Ok(Vec::new());
        }

        // The usual case: STREAMINFO's sample count holds, and the frames are decoded
        // straight into the planar result.
        let mut planar: Vec<T> = Vec::with_capacity(num_channels * total_spc);
        // SAFETY: the capacity covers `num_channels * total_spc` values.
        if let Some(written) = unsafe { self.decode_planar_into(planar.as_mut_ptr(), total_spc)? } {
            let p = planar.as_mut_ptr();
            if written < total_spc {
                // A truncated stream: close the gaps between the channels.
                for ch in 1..num_channels {
                    // SAFETY: both ranges lie inside the capacity and were written above.
                    unsafe { std::ptr::copy(p.add(ch * total_spc), p.add(ch * written), written) };
                }
            }
            // SAFETY: the first `num_channels * written` values are initialised.
            unsafe { planar.set_len(num_channels * written) };
            return Ok(planar);
        }
        drop(planar);

        // Per-channel typed output: one Vec<T> per channel, capacity = total_spc.
        // Written frame by frame; flattened into planar layout at the end.
        let mut ch_out: Vec<Vec<T>> = (0..num_channels).map(|_| Vec::with_capacity(total_spc)).collect();

        // Per-channel i32 scratch: pre-allocated to max_block_size.
        // For 2ch/4096: 2 × 16 KB = 32 KB → fits in L1 cache.
        // Predictor restoration and stereo decorrelation run on these hot buffers.
        let mut scratch: Vec<Vec<i32>> = (0..num_channels).map(|_| Vec::with_capacity(block_size_hint)).collect();

        // Inline conversion closure: resolved at monomorphisation time, so the match
        // on `bits` is hoisted out of the inner loop by LLVM's LICM pass.
        let convert = |s: i32| -> T {
            match bits {
                1..=8 => T::convert_from(((s) << (16 - bits)) as i16),
                9..=16 => T::convert_from(s as i16),
                17..=24 => T::convert_from(I24::wrapping_from_i32(s)),
                _ => T::convert_from(s),
            }
        };

        let data = self.audio_data();
        let mut offset = 0;

        while offset < data.len() {
            let frame_data = &data[offset..];
            if frame_data.len() < 2 {
                break;
            }

            if frame_data[0] != 0xFF || (frame_data[1] & 0xFC) != 0xF8 {
                offset += 1;
                continue;
            }

            match decode_frame_into_scratch(
                frame_data,
                self.stream_info.sample_rate,
                self.stream_info.bits_per_sample,
                &mut scratch,
            ) {
                Ok(bytes_consumed) => {
                    // scratch[ch] is now L1/L2-hot with block_size decoded i32 samples.
                    // Convert each channel immediately — reads are cache-warm.
                    for ch in 0..num_channels {
                        ch_out[ch].extend(scratch[ch].iter().map(|&s| convert(s)));
                    }
                    offset += bytes_consumed;
                },
                Err(FlacError::InvalidFrameSync { .. }) => {
                    offset += 1;
                },
                Err(e) => return Err(e),
            }
        }

        // Flatten per-channel Vecs into a single planar Vec<T>:
        // [ch0[0..spc], ch1[0..spc], ..., ch(N-1)[0..spc]]
        // This is one sequential bulk copy — fast bandwidth-bound operation.
        let actual_spc = ch_out.first().map(|v| v.len()).unwrap_or(0);
        let mut flat = Vec::with_capacity(num_channels * actual_spc);
        for ch_data in ch_out {
            flat.extend(ch_data);
        }

        Ok(flat)
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
        let bytes_per_sample = bits_per_sample.div_ceil(8);
        let block_align = channels * bytes_per_sample;
        let sample_rate = NonZeroU32::new(si.sample_rate)
            .ok_or_else(|| AudioIOError::corrupted_data_simple("Invalid sample rate", "sample rate cannot be zero"))?;
        let byte_rate = sample_rate.get() * block_align as u32;

        // In FLAC STREAMINFO, `total_samples` is ALREADY the per-channel sample count
        // (the FLAC spec calls this "inter-channel samples").
        let samples_per_channel = si.total_samples as usize;
        let total_all_channels = samples_per_channel.saturating_mul(channels as usize);
        let duration = Duration::from_secs_f64(samples_per_channel as f64 / sample_rate.get() as f64);

        Ok(BaseAudioInfo::new(
            sample_rate,
            channels,
            bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            total_all_channels,
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
                format!("File size {} exceeds maximum {} bytes", file_size, MAX_FLAC_SIZE),
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

        let marker: [u8; 4] = bytes[0..4]
            .try_into()
            .map_err(|_| AudioIOError::corrupted_data_simple("Cannot read FLAC marker", "Insufficient data"))?;

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
            let block_size = u32::from_be_bytes([0, bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as usize;

            let block_type_enum = MetadataBlockType::from_byte(block_type);

            // Check for reserved/invalid types
            if matches!(block_type_enum, MetadataBlockType::Reserved(n) if n > 126) {
                return Err(AudioIOError::FlacError(FlacError::InvalidMetadataBlockType(block_type)));
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
                stream_info = Some(StreamInfo::from_bytes(block_data).map_err(AudioIOError::FlacError)?);
            }

            metadata_blocks.push((block_type_enum, data_start..data_end));
            offset = data_end;
        }

        let stream_info = stream_info.ok_or(AudioIOError::FlacError(FlacError::MissingStreamInfo))?;

        // Determine sample type from bits per sample
        let sample_type = match stream_info.bits_per_sample {
            1..=16 => ValidatedSampleType::I16,
            17..=24 => ValidatedSampleType::I24,
            25..=32 => ValidatedSampleType::I32,
            _ => {
                return Err(AudioIOError::FlacError(FlacError::InvalidBitsPerSample {
                    bits: stream_info.bits_per_sample,
                }));
            },
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
        T: StandardSample + 'static,
    {
        let sample_rate = NonZeroU32::new(self.stream_info.sample_rate)
            .ok_or_else(|| AudioIOError::corrupted_data_simple("Invalid sample rate", "sample rate cannot be zero"))?;

        let num_channels = self.stream_info.channels as usize;
        let flat = self.decode_all_frames_typed::<T>().map_err(AudioIOError::FlacError)?;

        if num_channels == 1 {
            let arr = ndarray::Array1::from_vec(flat);
            AudioSamples::new_mono(arr, sample_rate).map_err(Into::into)
        } else {
            let spc = flat.len() / num_channels;
            let arr = ndarray::Array2::from_shape_vec((num_channels, spc), flat)
                .map_err(|e| AudioIOError::corrupted_data_simple("Array shape error", e.to_string()))?;
            AudioSamples::new_multi_channel(arr, sample_rate).map_err(Into::into)
        }
    }

    fn read_into<T>(&'a self, audio: &mut AudioSamples<'a, T>) -> AudioIOResult<()>
    where
        T: StandardSample + 'static,
    {
        // Decode all frames
        let decoded = self.decode_all_frames().map_err(AudioIOError::FlacError)?;

        let num_channels = decoded.num_channels();
        let samples_per_channel = decoded.samples_per_channel();
        let total_samples = num_channels * samples_per_channel;

        if total_samples != audio.total_samples().get() {
            return Err(AudioIOError::corrupted_data_simple(
                "Sample count mismatch",
                format!(
                    "FLAC has {} samples, buffer has {}",
                    total_samples,
                    audio.total_samples()
                ),
            ));
        }

        if num_channels != audio.num_channels().get() as usize {
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
        let non_empty = non_empty_slice::NonEmptyVec::try_from(planar_data)
            .map_err(|_| AudioIOError::corrupted_data_simple("Empty planar data", "No samples to replace with"))?;
        audio.replace_with_vec(&non_empty).map_err(Into::into)
    }
}

impl<'a> AudioFileWrite for FlacFile<'a> {
    fn write<P, T>(&mut self, out_fp: P) -> AudioIOResult<()>
    where
        P: AsRef<Path>,
        T: StandardSample + 'static,
    {
        // Read current audio as target type
        let audio = self.read::<T>()?;

        // Write using the public function
        let mut file = crate::fastio::open_for_overwrite(out_fp.as_ref())?;
        write_flac(BufWriter::new(&mut file), &audio, CompressionLevel::DEFAULT)?;
        Ok(crate::fastio::finish_overwrite(&mut file)?)
    }
}

/// Write AudioSamples to a FLAC file.
///
/// FLAC stores integer PCM only, here at 16 or 24 bits per sample, so the sample type decides
/// what reaches the file:
/// - `i16` writes 16-bit samples and `I24` writes 24-bit samples, both unchanged.
/// - `i32` writes its 24 most significant bits.
/// - `u8` is rescaled to 24 bits.
/// - `f32` and `f64` are rounded to 24-bit integers, with values outside \[-1, 1\] clipped.
///
/// The signal is encoded block by block, so memory use depends on the block size and channel
/// count and not on the length of the signal. STREAMINFO holds the MD5 signature of the audio, so
/// that decoders can verify the file; [`write_flac_without_md5`] leaves it out.
///
/// With the `flac-parallel` feature, frames are encoded on rayon's global thread pool.
pub fn write_flac<W: Write, T>(writer: W, audio: &AudioSamples<T>, level: CompressionLevel) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
{
    let md5 = audio_md5::<T, T>(channel_frame_view(audio));
    write_flac_stream::<_, T, T>(writer, audio, level, md5)
}

/// [`write_flac`] without the MD5 signature of the audio: STREAMINFO says it is unknown, which
/// the format allows, and decoders cannot verify the file against it. Encoding takes less time.
pub fn write_flac_without_md5<W: Write, T>(
    writer: W,
    audio: &AudioSamples<T>,
    level: CompressionLevel,
) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
{
    write_flac_stream::<_, T, T>(writer, audio, level, [0; 16])
}

/// Write a whole FLAC file of samples of type `U`, each converted from `T` as it is encoded, so
/// that no copy of the signal is made. With `md5`, the signature is worked out on another thread
/// while the frames are encoded when that helps, and written into STREAMINFO afterwards.
pub(crate) fn write_flac_file<T, U>(
    file: &mut File,
    buffer: usize,
    audio: &AudioSamples<T>,
    level: CompressionLevel,
    md5: bool,
) -> AudioIOResult<()>
where
    T: StandardSample + ConvertTo<U> + 'static,
    U: StandardSample + 'static,
{
    const MD5_OFFSET: u64 = 4 + 4 + 18;
    let view = channel_frame_view(audio);
    if !md5 {
        return write_flac_stream::<_, T, U>(BufWriter::with_capacity(buffer, file), audio, level, [0; 16]);
    }
    let large = view.len() * size_of::<T>() >= 1024 * 1024;
    if !large || crate::fastio::transfer_threads() < 2 {
        let digest = audio_md5::<T, U>(view);
        return write_flac_stream::<_, T, U>(BufWriter::with_capacity(buffer, file), audio, level, digest);
    }
    let digest = std::thread::scope(|scope| {
        let md5 = std::thread::Builder::new().spawn_scoped(scope, move || audio_md5::<T, U>(view));
        let encoded = write_flac_stream::<_, T, U>(BufWriter::with_capacity(buffer, &mut *file), audio, level, [0; 16]);
        let digest = match md5 {
            Ok(handle) => handle.join().unwrap_or_else(|_| audio_md5::<T, U>(view)),
            Err(_) => audio_md5::<T, U>(view),
        };
        encoded.map(|()| digest)
    })?;
    let end = file.stream_position()?;
    file.seek(SeekFrom::Start(MD5_OFFSET))?;
    file.write_all(&digest)?;
    file.seek(SeekFrom::Start(end))?;
    Ok(())
}

/// MD5 of the samples as FLAC stores them for `U`, converted from `T`: interleaved, little-endian,
/// two or three bytes.
pub(crate) fn audio_md5<T, U>(view: ndarray::ArrayView2<'_, T>) -> [u8; 16]
where
    T: StandardSample + ConvertTo<U> + 'static,
    U: StandardSample + 'static,
{
    let mut md5 = Md5::new();
    let channels = view.dim().0;
    let frames = view.dim().1;
    let width = usize::from(flac_bits_per_sample::<U>()) / 8;

    // These types are stored unchanged, so their bytes go straight through.
    let id = std::any::TypeId::of::<U>();
    if id == std::any::TypeId::of::<i16>() || id == std::any::TypeId::of::<I24>() {
        if crate::fastio::write_interleaved_le::<T, U, _>(&mut md5, &[], view, 0).is_ok() {
            return md5.finish();
        }
        md5 = Md5::new();
    }

    const BLOCK_FRAMES: usize = 4096;
    let mut block: Vec<Vec<i32>> = vec![Vec::with_capacity(BLOCK_FRAMES); channels];
    let mut bytes = vec![0u8; BLOCK_FRAMES * channels * width];
    let mut start = 0usize;
    while start < frames {
        let len = BLOCK_FRAMES.min(frames - start);
        for (c, out) in block.iter_mut().enumerate() {
            out.clear();
            append_channel_i32_as::<T, U>(view, c, start, len, out);
        }
        md5.update(interleave_le(&block, len, width, &mut bytes));
        start += len;
    }
    md5.finish()
}

/// `len` samples of each channel, interleaved at `width` little-endian bytes, into `bytes`.
pub(crate) fn interleave_le<'a, S: AsRef<[i32]>>(
    channels: &[S],
    len: usize,
    width: usize,
    bytes: &'a mut [u8],
) -> &'a [u8] {
    let stride = channels.len() * width;
    for (c, channel) in channels.iter().enumerate() {
        let samples = &channel.as_ref()[..len];
        if width == 2 {
            for (out, &s) in bytes[c * 2..].chunks_mut(stride).zip(samples) {
                out[..2].copy_from_slice(&(s as i16).to_le_bytes());
            }
        } else {
            for (out, &s) in bytes[c * 3..].chunks_mut(stride).zip(samples) {
                out[..3].copy_from_slice(&s.to_le_bytes()[..3]);
            }
        }
    }
    &bytes[..len * stride]
}

/// Write the stream with `md5` in STREAMINFO.
fn write_flac_stream<W: Write, T, U>(
    mut writer: W,
    audio: &AudioSamples<T>,
    level: CompressionLevel,
    md5: [u8; 16],
) -> AudioIOResult<()>
where
    T: StandardSample + ConvertTo<U> + 'static,
    U: StandardSample + 'static,
{
    let sample_rate = audio.sample_rate().get();
    let channels = audio.num_channels().get() as usize;
    let samples_per_channel = audio.samples_per_channel().get();

    if channels == 0 || channels > 8 {
        return Err(AudioIOError::FlacError(FlacError::InvalidChannelCount {
            channels: channels.min(255) as u8,
        }));
    }

    let bits_per_sample = flac_bits_per_sample::<U>();

    writer.write_all(&FLAC_MARKER)?;

    let block_size = (level.block_size() as usize).min(samples_per_channel);
    let stream_info = StreamInfo {
        min_block_size: block_size as u16,
        max_block_size: block_size as u16,
        min_frame_size: 0,
        max_frame_size: 0,
        sample_rate,
        channels: channels as u8,
        bits_per_sample,
        total_samples: samples_per_channel as u64,
        md5_signature: md5,
    };
    let streaminfo_bytes = stream_info.to_bytes();
    writer.write_all(&[0x80, 0, 0, streaminfo_bytes.len() as u8])?; // Last block, type 0 (STREAMINFO)
    writer.write_all(&streaminfo_bytes)?;

    let config = EncoderConfig::for_level(level, bits_per_sample, level.block_size());
    let view = channel_frame_view(audio);

    #[cfg(feature = "flac-parallel")]
    {
        use rayon::prelude::*;
        let blocks = samples_per_channel.div_ceil(block_size);
        // Batches keep at most this many encoded frames in memory per worker.
        let batch = 16 * rayon::current_num_threads();
        for first in (0..blocks).step_by(batch.max(1)) {
            let last = (first + batch).min(blocks);
            let frames: Vec<Vec<u8>> = (first..last)
                .into_par_iter()
                .map_init(
                    || {
                        (
                            FrameEncoder::new(config, channels, block_size),
                            vec![Vec::with_capacity(block_size); channels],
                        )
                    },
                    |(encoder, block), index| {
                        let start = index * block_size;
                        let len = block_size.min(samples_per_channel - start);
                        for (c, out) in block.iter_mut().enumerate() {
                            out.clear();
                            append_channel_i32_as::<T, U>(view, c, start, len, out);
                        }
                        encoder
                            .encode(block.as_slice(), bits_per_sample, sample_rate, index as u64)
                            .map(<[u8]>::to_vec)
                    },
                )
                .collect::<Result<_, FlacError>>()
                .map_err(AudioIOError::FlacError)?;
            for frame in &frames {
                writer.write_all(frame)?;
            }
        }
    }
    #[cfg(not(feature = "flac-parallel"))]
    {
        let mut encoder = FrameEncoder::new(config, channels, block_size);
        let mut block = vec![Vec::with_capacity(block_size); channels];
        let mut frame_number = 0u64;
        let mut start = 0usize;
        while start < samples_per_channel {
            let len = block_size.min(samples_per_channel - start);
            for (c, out) in block.iter_mut().enumerate() {
                out.clear();
                append_channel_i32_as::<T, U>(view, c, start, len, out);
            }
            let frame = encoder
                .encode(block.as_slice(), bits_per_sample, sample_rate, frame_number)
                .map_err(AudioIOError::FlacError)?;
            writer.write_all(frame)?;
            frame_number += 1;
            start += len;
        }
    }

    writer.flush()?;
    Ok(())
}

/// The FLAC sample width of `T`: 16 bits for `i16`, 24 for every other type.
pub(crate) const fn flac_bits_per_sample<T: StandardSample>() -> u8 {
    match T::SAMPLE_TYPE {
        SampleType::I16 => 16,
        _ => 24,
    }
}

/// One sample as the integer FLAC stores for `T`. See [`write_flac`] for the rule per type.
#[inline(always)]
fn flac_sample<T: StandardSample + 'static>(s: T) -> i32 {
    use std::any::TypeId;
    let id = TypeId::of::<T>();
    // SAFETY (all branches): the TypeId comparison proves T is the type being read.
    if id == TypeId::of::<i16>() {
        i32::from(unsafe { std::mem::transmute_copy::<T, i16>(&s) })
    } else if id == TypeId::of::<I24>() {
        unsafe { std::mem::transmute_copy::<T, I24>(&s) }.to_i32()
    } else if id == TypeId::of::<i32>() {
        unsafe { std::mem::transmute_copy::<T, i32>(&s) >> 8 }
    } else if id == TypeId::of::<f32>() {
        float_to_24(f64::from(unsafe { std::mem::transmute_copy::<T, f32>(&s) }))
    } else if id == TypeId::of::<f64>() {
        float_to_24(unsafe { std::mem::transmute_copy::<T, f64>(&s) })
    } else {
        let v: I24 = s.convert_to();
        v.to_i32()
    }
}

/// Round to 24 bits with the same asymmetric scale that decoding uses, so that decoded audio
/// encodes back to the same integers. The cast sends NaN to zero and saturates.
#[inline(always)]
fn float_to_24(v: f64) -> i32 {
    if v >= 0.0 {
        ((v * 8_388_607.0 + 0.5) as i32).min(8_388_607)
    } else {
        ((v * 8_388_608.0 - 0.5) as i32).max(-8_388_608)
    }
}

/// Append `len` samples of channel `channel` from frame `start` to `out`, as FLAC integers.
pub(crate) fn append_channel_i32<T>(
    view: ndarray::ArrayView2<'_, T>,
    channel: usize,
    start: usize,
    len: usize,
    out: &mut Vec<i32>,
) where
    T: StandardSample + 'static,
{
    append_channel_i32_as::<T, T>(view, channel, start, len, out);
}

/// [`append_channel_i32`] for samples converted to `U` first.
pub(crate) fn append_channel_i32_as<T, U>(
    view: ndarray::ArrayView2<'_, T>,
    channel: usize,
    start: usize,
    len: usize,
    out: &mut Vec<i32>,
) where
    T: StandardSample + ConvertTo<U> + 'static,
    U: StandardSample + 'static,
{
    let row = view.index_axis(ndarray::Axis(0), channel);
    if let Some(slice) = row.to_slice() {
        out.extend(
            slice[start..start + len]
                .iter()
                .map(|&s| flac_sample::<U>(s.convert_to())),
        );
    } else {
        let stride = row.strides()[0];
        let base = row.as_ptr();
        out.reserve(len);
        for f in start..start + len {
            // SAFETY: f is within the row, whose elements lie `stride` apart from `base`.
            let sample: U = unsafe { *base.offset(f as isize * stride) }.convert_to();
            out.push(flac_sample(sample));
        }
    }
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
    use std::fs;
    use std::time::Duration as StdDuration;

    use audio_samples::{AudioTypeConversion, sample_rate, sine_wave};

    use super::*;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Convert a slice of i16 values to f64 in [-1.0, 1.0].
    fn i16_to_f64(v: i16) -> f64 {
        v as f64 / 32768.0
    }

    /// Mean squared error between two f64 slices of equal length.
    fn mse(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "length mismatch in mse");
        let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum / a.len() as f64
    }

    // =========================================================================
    // A. Read correctness with resources/test.flac
    // =========================================================================

    #[test]
    fn test_flac_file_info_display() {
        let info = FlacFileInfo {
            metadata_blocks: vec![MetadataBlockType::StreamInfo, MetadataBlockType::VorbisComment],
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

    #[test]
    fn test_read_test_flac_as_i16() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file =
            FlacFile::open_with_options(flac_path, OpenOptions::default()).expect("Failed to open test FLAC file");
        let audio = flac_file.read::<i16>().expect("read as i16");
        assert!(audio.num_channels().get() > 0);
        assert!(audio.sample_rate().get() > 0);
        assert!(audio.samples_per_channel().get() > 0);
        assert_eq!(
            audio.total_samples().get(),
            audio.num_channels().get() as usize * audio.samples_per_channel().get()
        );
    }

    #[test]
    fn test_read_test_flac_as_i32() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file =
            FlacFile::open_with_options(flac_path, OpenOptions::default()).expect("Failed to open test FLAC file");
        let audio = flac_file.read::<i32>().expect("read as i32");
        assert!(audio.num_channels().get() > 0);
        assert!(audio.sample_rate().get() > 0);
        assert!(audio.samples_per_channel().get() > 0);
    }

    #[test]
    fn test_read_test_flac_as_f32() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file =
            FlacFile::open_with_options(flac_path, OpenOptions::default()).expect("Failed to open test FLAC file");
        let audio = flac_file.read::<f32>().expect("read as f32");
        assert!(audio.num_channels().get() > 0);
        assert!(audio.sample_rate().get() > 0);
        assert!(audio.samples_per_channel().get() > 0);
    }

    #[test]
    fn test_read_test_flac_as_f64() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file =
            FlacFile::open_with_options(flac_path, OpenOptions::default()).expect("Failed to open test FLAC file");
        let audio = flac_file.read::<f64>().expect("read as f64");
        assert!(audio.num_channels().get() > 0);
        assert!(audio.sample_rate().get() > 0);
        assert!(audio.samples_per_channel().get() > 0);
    }

    #[test]
    fn test_read_test_flac_as_i24() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file =
            FlacFile::open_with_options(flac_path, OpenOptions::default()).expect("Failed to open test FLAC file");
        let audio = flac_file.read::<I24>().expect("read as I24");
        assert!(audio.num_channels().get() > 0);
        assert!(audio.sample_rate().get() > 0);
        assert!(audio.samples_per_channel().get() > 0);
    }

    /// Verify all types return the same sample count and metadata.
    #[test]
    fn test_read_test_flac_consistent_metadata() {
        let flac_path = Path::new("resources/test.flac");

        let flac_i16 = FlacFile::open_with_options(flac_path, OpenOptions::default()).unwrap();
        let a_i16 = flac_i16.read::<i16>().unwrap();
        let flac_f32 = FlacFile::open_with_options(flac_path, OpenOptions::default()).unwrap();
        let a_f32 = flac_f32.read::<f32>().unwrap();
        let flac_f64 = FlacFile::open_with_options(flac_path, OpenOptions::default()).unwrap();
        let a_f64 = flac_f64.read::<f64>().unwrap();

        assert_eq!(a_i16.num_channels(), a_f32.num_channels());
        assert_eq!(a_i16.num_channels(), a_f64.num_channels());
        assert_eq!(a_i16.sample_rate(), a_f32.sample_rate());
        assert_eq!(a_i16.sample_rate(), a_f64.sample_rate());
        assert_eq!(a_i16.samples_per_channel(), a_f32.samples_per_channel());
        assert_eq!(a_i16.samples_per_channel(), a_f64.samples_per_channel());
    }

    /// f64 and i16 conversions of the same source should be close in content.
    #[test]
    fn test_read_test_flac_cross_type_content_similarity() {
        let flac_path = Path::new("resources/test.flac");

        let flac_i16 = FlacFile::open_with_options(flac_path, OpenOptions::default()).unwrap();
        let a_i16 = flac_i16.read::<i16>().unwrap();
        let flac_f64 = FlacFile::open_with_options(flac_path, OpenOptions::default()).unwrap();
        let a_f64 = flac_f64.read::<f64>().unwrap();

        let interleaved_i16 = a_i16.to_interleaved_vec();
        let interleaved_f64 = a_f64.to_interleaved_vec();

        let i16_as_f64: Vec<f64> = interleaved_i16.iter().copied().map(i16_to_f64).collect();

        let err = mse(&i16_as_f64, &interleaved_f64);
        assert!(err < 1e-3, "MSE between i16-converted-to-f64 and raw f64 was {err}");
    }

    #[test]
    fn test_flac_base_info() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file = FlacFile::open_metadata(flac_path).expect("open metadata");

        let info = flac_file.base_info().expect("base_info");
        assert_eq!(info.file_type, FileType::FLAC);
        assert!(info.sample_rate.get() > 0);
        assert!(info.channels > 0);
        assert!(info.total_samples > 0);
        assert!(info.duration.as_secs_f64() > 0.0);
    }

    /// total_samples from base_info == channels * samples_per_channel.
    #[test]
    fn test_flac_base_info_total_samples_consistent() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file = FlacFile::open_with_options(flac_path, OpenOptions::default()).unwrap();
        let info = flac_file.base_info().unwrap();
        let audio = flac_file.read::<i16>().unwrap();

        let expected_total = audio.num_channels().get() as usize * audio.samples_per_channel().get();
        assert_eq!(
            info.total_samples, expected_total,
            "base_info total_samples should equal channels * samples_per_channel"
        );
    }

    // =========================================================================
    // B. Round-trip tests
    //
    // Most of these check structure (sample count, rate, channels); the content
    // fidelity tests compare the decoded samples with the input.
    // =========================================================================

    fn roundtrip_check_structure(level: CompressionLevel, tag: &str) {
        let sr = sample_rate!(44100);
        let sine_f32 = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.25), sr, 0.5);
        let sine_i16 = sine_f32.to_format::<i16>();

        let out_path = std::env::temp_dir().join(format!("flac_rt_{tag}.flac"));
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine_i16, level).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();

        assert_eq!(back.sample_rate(), sr, "sample rate mismatch ({tag})");
        assert_eq!(
            back.num_channels(),
            sine_i16.num_channels(),
            "channel count mismatch ({tag})"
        );
        assert_eq!(
            back.samples_per_channel(),
            sine_i16.samples_per_channel(),
            "sample count mismatch ({tag})"
        );
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_roundtrip_i16() {
        roundtrip_check_structure(CompressionLevel::FASTEST, "i16_fast");
    }

    #[test]
    fn test_roundtrip_i16_default_level() {
        roundtrip_check_structure(CompressionLevel::DEFAULT, "i16_default");
    }

    #[test]
    fn test_roundtrip_i24() {
        let sr = sample_rate!(44100);
        let sine_i24 = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.25), sr, 0.5).to_format::<I24>();

        let out_path = std::env::temp_dir().join("flac_rt_i24.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine_i24, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<I24>().unwrap();

        assert_eq!(back.sample_rate(), sr);
        assert_eq!(back.num_channels(), sine_i24.num_channels());
        assert_eq!(back.samples_per_channel(), sine_i24.samples_per_channel());
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_roundtrip_i32() {
        let sr = sample_rate!(44100);
        let sine_i32 = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.25), sr, 0.5).to_format::<i32>();

        let out_path = std::env::temp_dir().join("flac_rt_i32.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine_i32, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i32>().unwrap();

        assert_eq!(back.sample_rate(), sr);
        assert_eq!(back.num_channels(), sine_i32.num_channels());
        assert_eq!(back.samples_per_channel(), sine_i32.samples_per_channel());
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_roundtrip_f32() {
        let sr = sample_rate!(44100);
        let sine = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.25), sr, 0.5);

        let out_path = std::env::temp_dir().join("flac_rt_f32.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<f32>().unwrap();

        assert_eq!(back.sample_rate(), sr);
        assert_eq!(back.num_channels(), sine.num_channels());
        assert_eq!(back.samples_per_channel(), sine.samples_per_channel());
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_roundtrip_f64() {
        let sr = sample_rate!(44100);
        let sine_f64 = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.25), sr, 0.5).to_format::<f64>();

        let out_path = std::env::temp_dir().join("flac_rt_f64.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine_f64, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<f64>().unwrap();

        assert_eq!(back.sample_rate(), sr);
        assert_eq!(back.num_channels(), sine_f64.num_channels());
        assert_eq!(back.samples_per_channel(), sine_f64.samples_per_channel());
        fs::remove_file(&out_path).ok();
    }

    /// Content fidelity test for i16 sine roundtrip.
    #[test]
    fn test_roundtrip_i16_content_fidelity() {
        let sr = sample_rate!(44100);
        let sine_i16 = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.25), sr, 0.5).to_format::<i16>();

        let out_path = std::env::temp_dir().join("flac_rt_i16_content.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine_i16, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();

        let orig: Vec<f64> = sine_i16
            .to_interleaved_vec()
            .iter()
            .map(|&s| s as f64 / 32768.0)
            .collect();
        let got: Vec<f64> = back.to_interleaved_vec().iter().map(|&s| s as f64 / 32768.0).collect();
        let err = mse(&orig, &got);
        assert!(err < 1e-6, "i16 content fidelity MSE: {err}");

        fs::remove_file(&out_path).ok();
    }

    // =========================================================================
    // C. Multi-channel tests
    // =========================================================================

    fn make_stereo_audio(sr: std::num::NonZeroU32) -> audio_samples::AudioSamples<'static, i16> {
        // Create stereo by combining two different sines as a 2-row Array2
        let left: Vec<i16> = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.1), sr, 0.5)
            .to_format::<i16>()
            .to_interleaved_vec()
            .into_iter()
            .collect();
        let right: Vec<i16> = sine_wave::<f32>(880.0, StdDuration::from_secs_f64(0.1), sr, 0.5)
            .to_format::<i16>()
            .to_interleaved_vec()
            .into_iter()
            .collect();
        let n = left.len();
        let mut flat = Vec::with_capacity(n * 2);
        flat.extend_from_slice(&left);
        flat.extend_from_slice(&right);
        let arr = ndarray::Array2::from_shape_vec((2, n), flat).unwrap();
        audio_samples::AudioSamples::new_multi_channel(arr, sr).unwrap()
    }

    #[test]
    fn test_roundtrip_stereo_default_level() {
        use audio_samples::cosine_wave;

        let sr = sample_rate!(44100);
        let stereo = make_stereo_audio(sr);

        let out_path = std::env::temp_dir().join("flac_rt_stereo_default.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &stereo, CompressionLevel::DEFAULT).expect("write");
        }

        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().expect("read stereo DEFAULT (1)");
        assert_eq!(back.num_channels().get(), 2, "should have 2 channels");
        assert_eq!(back.samples_per_channel(), stereo.samples_per_channel(), "sample count");
        fs::remove_file(&out_path).ok();

        // Lengths around the block size:
        // 4096 samples → 1 full frame exactly
        // 4097 samples → 1 full + 1 tiny
        // 8192 samples → 2 full frames
        for n_samples in [4096usize, 4097, 8192] {
            let ch0: Vec<i16> = (0..n_samples)
                .map(|i| {
                    let t = i as f32 / 44100.0;
                    (11468.0 * (2.0 * std::f32::consts::PI * 110.0 * t).sin()) as i16
                })
                .collect();
            let ch1: Vec<i16> = (0..n_samples)
                .map(|i| {
                    let t = i as f32 / 44100.0;
                    (14745.0 * (2.0 * std::f32::consts::PI * 165.0 * t).cos()) as i16
                })
                .collect();
            let n = ch0.len();
            let mut flat = Vec::with_capacity(n * 2);
            flat.extend_from_slice(&ch0);
            flat.extend_from_slice(&ch1);
            let arr = ndarray::Array2::from_shape_vec((2, n), flat).unwrap();
            let audio: audio_samples::AudioSamples<'static, i16> =
                audio_samples::AudioSamples::new_multi_channel(arr, sr).unwrap();
            let path = std::env::temp_dir().join(format!("flac_rt_stereo_{n_samples}.flac"));
            {
                let file = File::create(&path).expect("create");
                write_flac(BufWriter::new(file), &audio, CompressionLevel::DEFAULT)
                    .unwrap_or_else(|e| panic!("write {n_samples} samples: {e}"));
            }
            let flac = FlacFile::open_with_options(&path, OpenOptions::default()).unwrap();
            flac.read::<i16>()
                .unwrap_or_else(|e| panic!("read {n_samples} samples: {e}"));
            fs::remove_file(&path).ok();
        }

        let duration = StdDuration::from_millis(250);
        let ch0: Vec<i16> = sine_wave::<f32>(110.0, duration, sr, 0.35)
            .to_format::<i16>()
            .to_interleaved_vec()
            .into_iter()
            .collect();
        let ch1: Vec<i16> = cosine_wave::<f32>(165.0, duration, sr, 0.45)
            .to_format::<i16>()
            .to_interleaved_vec()
            .into_iter()
            .collect();
        let n = ch0.len();
        let mut flat = Vec::with_capacity(n * 2);
        flat.extend_from_slice(&ch0);
        flat.extend_from_slice(&ch1);
        let arr = ndarray::Array2::from_shape_vec((2, n), flat).unwrap();
        let stereo_250ms: audio_samples::AudioSamples<'static, i16> =
            audio_samples::AudioSamples::new_multi_channel(arr, sr).unwrap();

        let out_path2 = std::env::temp_dir().join("flac_rt_stereo_250ms.flac");
        {
            let file = File::create(&out_path2).expect("create");
            write_flac(BufWriter::new(file), &stereo_250ms, CompressionLevel::DEFAULT).expect("write 250 ms stereo");
        }
        let flac2 = FlacFile::open_with_options(&out_path2, OpenOptions::default()).unwrap();
        let back2 = flac2.read::<i16>().expect("read 250 ms stereo at DEFAULT");
        assert_eq!(back2.num_channels().get(), 2);
        assert_eq!(back2.samples_per_channel(), stereo_250ms.samples_per_channel());
        fs::remove_file(&out_path2).ok();
    }

    #[test]
    fn test_roundtrip_stereo() {
        let sr = sample_rate!(44100);
        let stereo = make_stereo_audio(sr);

        let out_path = std::env::temp_dir().join("flac_rt_stereo.flac");
        {
            let file = File::create(&out_path).expect("create");
            // Use FASTEST (independent channels, no mid-side)
            write_flac(BufWriter::new(file), &stereo, CompressionLevel::FASTEST).expect("write");
        }

        // Verify the file can be opened and has correct metadata
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let info = flac.base_info().unwrap();
        assert_eq!(info.channels, 2, "should have 2 channels");
        assert_eq!(info.sample_rate.get(), sr.get(), "sample rate");

        let back = flac.read::<i16>().expect("read stereo FASTEST");
        assert_eq!(back.num_channels().get(), 2);
        assert_eq!(back.samples_per_channel(), stereo.samples_per_channel());

        fs::remove_file(&out_path).ok();
    }

    /// Content fidelity test for stereo: the two decoded channels keep their distinct content.
    #[test]
    fn test_roundtrip_stereo_content() {
        let sr = sample_rate!(44100);
        let stereo = make_stereo_audio(sr);

        let out_path = std::env::temp_dir().join("flac_rt_stereo_content.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &stereo, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();

        assert_eq!(back.num_channels().get(), 2);
        assert_eq!(back.samples_per_channel(), stereo.samples_per_channel());
        assert_eq!(back.sample_rate(), sr);

        // Channels differ (L=440Hz sine, R=880Hz sine)
        let back_iv: Vec<i16> = back.to_interleaved_vec().into_iter().collect();
        let ch0: Vec<i16> = back_iv.iter().step_by(2).copied().collect();
        let ch1: Vec<i16> = back_iv.iter().skip(1).step_by(2).copied().collect();
        assert_ne!(ch0, ch1, "stereo channels should carry distinct content");

        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_roundtrip_quad_channel() {
        use audio_samples::AudioSamples;

        let sr = sample_rate!(44100);
        let n = 1024usize;
        // 4 channels with distinct DC values for easy verification
        let mut flat: Vec<i16> = Vec::with_capacity(4 * n);
        for ch in 0..4usize {
            flat.extend(std::iter::repeat_n((ch as i16 + 1) * 1000, n));
        }
        let arr = ndarray::Array2::from_shape_vec((4, n), flat).unwrap();
        let audio: AudioSamples<'static, i16> = AudioSamples::new_multi_channel(arr, sr).unwrap();

        let out_path = std::env::temp_dir().join("flac_rt_quad.flac");
        {
            let file = File::create(&out_path).expect("create");
            // FASTEST for lossless constant-value recovery
            write_flac(BufWriter::new(file), &audio, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();

        assert_eq!(back.num_channels().get(), 4);
        assert_eq!(back.samples_per_channel().get(), n);

        // Each channel should have its distinct constant value (DC audio decoded exactly)
        let back_iv: Vec<i16> = back.to_interleaved_vec().into_iter().collect();
        for (ch, &got) in back_iv.iter().take(4).enumerate() {
            let expected = (ch as i16 + 1) * 1000;
            // first frame
            assert_eq!(got, expected, "channel {ch} first sample mismatch");
        }

        fs::remove_file(&out_path).ok();
    }

    // =========================================================================
    // D. Sample rate variety
    // =========================================================================

    fn roundtrip_sample_rate(hz: u32) {
        let sr = std::num::NonZeroU32::new(hz).unwrap();
        let sine = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.1), sr, 0.5).to_format::<i16>();

        let tag = format!("sr_{hz}");
        let out_path = std::env::temp_dir().join(format!("flac_rt_{tag}.flac"));
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();
        assert_eq!(back.sample_rate().get(), hz, "sample rate should be preserved");
        assert_eq!(back.samples_per_channel(), sine.samples_per_channel());
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_sample_rate_44100() {
        roundtrip_sample_rate(44100);
    }

    #[test]
    fn test_sample_rate_48000() {
        roundtrip_sample_rate(48000);
    }

    #[test]
    fn test_sample_rate_96000() {
        roundtrip_sample_rate(96000);
    }

    // =========================================================================
    // E. Edge cases
    // =========================================================================

    #[test]
    fn test_short_file_single_block() {
        // Just a handful of samples — less than one typical block
        let sr = sample_rate!(44100);
        let sine = sine_wave::<f32>(440.0, StdDuration::from_millis(10), sr, 0.5).to_format::<i16>();

        let out_path = std::env::temp_dir().join("flac_edge_short.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();
        assert_eq!(back.samples_per_channel(), sine.samples_per_channel());
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_long_file_five_seconds() {
        let sr = sample_rate!(44100);
        let expected_spc = 5 * 44100usize;
        let sine = sine_wave::<f32>(440.0, StdDuration::from_secs(5), sr, 0.5).to_format::<i16>();

        let out_path = std::env::temp_dir().join("flac_edge_long.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::FASTEST).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();
        assert_eq!(
            back.samples_per_channel().get(),
            expected_spc,
            "5s at 44100 should yield exactly 220500 samples/channel"
        );
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_silence_roundtrip() {
        use audio_samples::{AudioSamples, nzu};

        let sr = sample_rate!(44100);
        let silence: AudioSamples<'static, i16> = AudioSamples::zeros_mono(nzu!(4096), sr);

        let out_path = std::env::temp_dir().join("flac_edge_silence.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &silence, CompressionLevel::DEFAULT).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();

        assert_eq!(back.samples_per_channel().get(), 4096);
        // All samples must be zero
        let all_zero = back.to_interleaved_vec().iter().all(|&s| s == 0);
        assert!(all_zero, "silence should round-trip exactly");
        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_max_amplitude_i16() {
        use audio_samples::AudioSamples;
        use ndarray::Array1;

        let sr = sample_rate!(44100);
        // Alternating max/min i16 values
        let samples: Vec<i16> = (0..4096)
            .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
            .collect();
        let arr = Array1::from(samples.clone());
        let audio: AudioSamples<'static, i16> = AudioSamples::new_mono(arr, sr).unwrap();

        let out_path = std::env::temp_dir().join("flac_edge_maxamp.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &audio, CompressionLevel::DEFAULT).expect("write");
        }
        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).unwrap();
        let back = flac.read::<i16>().unwrap();

        let back_v: Vec<i16> = back.to_interleaved_vec().into_iter().collect();
        assert_eq!(back_v, samples, "max-amplitude i16 values should round-trip exactly");
        fs::remove_file(&out_path).ok();
    }

    // =========================================================================
    // F. Compression levels
    // =========================================================================

    #[test]
    fn test_compression_levels_produce_valid_files() {
        let sr = sample_rate!(44100);
        let sine = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.25), sr, 0.5).to_format::<i16>();

        for level in [
            CompressionLevel::FASTEST,
            CompressionLevel::FAST,
            CompressionLevel::DEFAULT,
            CompressionLevel::BEST,
        ] {
            let tag = format!("clvl_{}", level.level());
            let out_path = std::env::temp_dir().join(format!("flac_{tag}.flac"));
            {
                let file = File::create(&out_path).expect("create");
                write_flac(BufWriter::new(file), &sine, level).expect("write");
            }
            let flac = FlacFile::open_with_options(&out_path, OpenOptions::default())
                .unwrap_or_else(|e| panic!("level {:?}: open failed: {e}", level));
            // All levels must produce readable files with correct structure.
            let back = flac
                .read::<i16>()
                .unwrap_or_else(|e| panic!("level {:?}: read failed: {e}", level));
            assert_eq!(
                back.samples_per_channel(),
                sine.samples_per_channel(),
                "level {:?} sample count mismatch",
                level
            );
            assert_eq!(back.sample_rate(), sr, "level {:?} sample rate mismatch", level);
            assert_eq!(
                back.num_channels(),
                sine.num_channels(),
                "level {:?} channel count mismatch",
                level
            );

            fs::remove_file(&out_path).ok();
        }
    }

    #[test]
    fn test_best_compression_smaller_than_fastest() {
        let sr = sample_rate!(44100);
        // Use a longer, more compressible sine to get a meaningful size difference
        let sine = sine_wave::<f32>(440.0, StdDuration::from_secs(2), sr, 0.5).to_format::<i16>();

        let fast_path = std::env::temp_dir().join("flac_cmp_fastest.flac");
        let best_path = std::env::temp_dir().join("flac_cmp_best.flac");

        {
            let file = File::create(&fast_path).expect("create fast");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::FASTEST).expect("write fast");
        }
        {
            let file = File::create(&best_path).expect("create best");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::BEST).expect("write best");
        }

        let fast_size = fs::metadata(&fast_path).unwrap().len();
        let best_size = fs::metadata(&best_path).unwrap().len();

        assert!(
            fast_size >= best_size,
            "FASTEST ({fast_size} bytes) should not be smaller than BEST ({best_size} bytes)"
        );

        fs::remove_file(&fast_path).ok();
        fs::remove_file(&best_path).ok();
    }

    // =========================================================================
    // G. Reference decoding with symphonia
    //
    // symphonia, a dev-dependency, is an independent pure-Rust decoder. These
    // tests check that files from this encoder decode with it.
    // =========================================================================

    fn symphonia_decode_flac(path: &std::path::Path) -> (u32, u32, usize) {
        use symphonia::core::audio::GenericAudioBufferRef;
        use symphonia::core::codecs::CodecParameters;
        use symphonia::core::codecs::audio::AudioDecoderOptions;
        use symphonia::core::formats::probe::Hint;
        use symphonia::core::formats::{FormatOptions, TrackType};
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;

        let file = std::fs::File::open(path).expect("open for symphonia");
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let hint = Hint::new();
        let mut format = symphonia::default::get_probe()
            .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
            .expect("symphonia probe");

        let track = format
            .default_track(TrackType::Audio)
            .expect("default audio track")
            .clone();
        let audio_params = match track.codec_params.as_ref().expect("codec params") {
            CodecParameters::Audio(p) => p.clone(),
            _ => unreachable!("expected audio codec parameters for a FLAC track"),
        };
        let sr = audio_params.sample_rate.expect("sample rate");
        let ch = audio_params.channels.as_ref().expect("channels").count() as u32;

        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
            .expect("make decoder");

        // `next_packet` yields `None` at the end of the stream.
        let mut total_samples_decoded = 0usize;
        while let Ok(Some(packet)) = format.next_packet() {
            let decoded: GenericAudioBufferRef = decoder.decode(&packet).expect("decode packet");
            total_samples_decoded += decoded.frames();
        }

        (sr, ch, total_samples_decoded)
    }

    /// Write with this crate's encoder, decode with symphonia, verify metadata + sample count.
    #[test]
    fn test_symphonia_decodes_mono_sine() {
        let sr = sample_rate!(44100);
        let sine = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.1), sr, 0.5).to_format::<i16>();
        let expected_samples = sine.samples_per_channel().get();

        let out_path = std::env::temp_dir().join("flac_oracle_sym_mono.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::FASTEST).expect("write");
        }

        let (sym_sr, sym_ch, sym_samples) = symphonia_decode_flac(&out_path);
        assert_eq!(sym_sr, 44100, "symphonia sample rate");
        assert_eq!(sym_ch, 1, "symphonia channels");
        assert_eq!(sym_samples, expected_samples, "symphonia sample count");

        fs::remove_file(&out_path).ok();
    }

    /// Stereo file from this crate's encoder, decoded with symphonia.
    #[test]
    fn test_symphonia_decodes_stereo_sine() {
        let sr = sample_rate!(44100);
        let stereo = make_stereo_audio(sr);
        let expected_samples_per_ch = stereo.samples_per_channel().get();

        let out_path = std::env::temp_dir().join("flac_oracle_sym_stereo.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &stereo, CompressionLevel::FASTEST).expect("write");
        }

        let (sym_sr, sym_ch, sym_samples) = symphonia_decode_flac(&out_path);
        assert_eq!(sym_sr, 44100, "symphonia sample rate");
        assert_eq!(sym_ch, 2, "symphonia channels");
        // symphonia reports frames (samples per channel)
        assert_eq!(sym_samples, expected_samples_per_ch, "symphonia frame count");

        fs::remove_file(&out_path).ok();
    }

    // =========================================================================
    // H. Top-level API tests
    // =========================================================================

    #[test]
    fn test_lib_info_test_flac() {
        let info = crate::info("resources/test.flac").expect("lib::info");
        assert_eq!(info.file_type, FileType::FLAC);
        assert!(info.channels > 0);
        assert!(info.sample_rate.get() > 0);
        assert!(info.total_samples > 0);
    }

    #[test]
    fn test_lib_read_i16_test_flac() {
        let samples = crate::read::<_, i16>("resources/test.flac").expect("lib::read i16");
        assert!(samples.total_samples().get() > 0);
    }

    #[test]
    fn test_lib_read_f32_test_flac() {
        let samples = crate::read::<_, f32>("resources/test.flac").expect("lib::read f32");
        assert!(samples.total_samples().get() > 0);
    }

    #[test]
    fn test_lib_info_and_read_channels_consistent() {
        let info = crate::info("resources/test.flac").expect("lib::info");
        let samples = crate::read::<_, i16>("resources/test.flac").expect("lib::read");
        assert_eq!(samples.num_channels().get() as u16, info.channels);
        assert_eq!(samples.sample_rate().get(), info.sample_rate.get());
    }

    #[test]
    fn test_lib_write_read_roundtrip() {
        // write_flac at FASTEST; crate::write() uses CompressionLevel::DEFAULT.
        let sr = sample_rate!(44100);
        let sine = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.1), sr, 0.5).to_format::<i16>();

        let out_path = std::env::temp_dir().join("flac_lib_rw.flac");
        {
            let file = File::create(&out_path).expect("create");
            write_flac(BufWriter::new(file), &sine, CompressionLevel::FASTEST).expect("write");
        }

        let back = crate::read::<_, i16>(&out_path).expect("lib::read");
        assert_eq!(back.sample_rate(), sr);
        assert_eq!(back.num_channels(), sine.num_channels());
        assert_eq!(back.samples_per_channel(), sine.samples_per_channel());

        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_lib_open_test_flac() {
        let file = crate::open("resources/test.flac").expect("lib::open");
        assert!(!file.is_empty(), "opened file should have non-zero length");
    }

    // =========================================================================
    // I. Error cases
    // =========================================================================

    #[test]
    fn test_open_nonexistent_file_fails() {
        let result = FlacFile::open_with_options(Path::new("resources/does_not_exist.flac"), OpenOptions::default());
        assert!(result.is_err(), "opening nonexistent file should fail");
    }

    #[test]
    fn test_open_wav_as_flac_fails() {
        // Pass a WAV file to FlacFile — should fail because FLAC marker is missing
        let result = FlacFile::open_with_options(Path::new("resources/test.wav"), OpenOptions::default());
        assert!(result.is_err(), "opening a WAV as a FLAC should return an error");
    }

    #[test]
    fn test_lib_read_nonexistent_flac_fails() {
        let result = crate::read::<_, i16>("resources/no_such_file.flac");
        assert!(result.is_err(), "lib::read of nonexistent .flac should fail");
    }

    #[test]
    fn test_lib_read_unsupported_extension_fails() {
        let result = crate::read::<_, i16>("resources/audio.mp3");
        assert!(result.is_err(), "lib::read of .mp3 should fail with unsupported format");
    }

    // =========================================================================
    // J. Basic open, read and write
    // =========================================================================

    #[test]
    fn test_flac_open_read_i16() {
        let flac_path = Path::new("resources/test.flac");
        let flac_file =
            FlacFile::open_with_options(flac_path, OpenOptions::default()).expect("Failed to open test FLAC file");

        let audio = flac_file.read::<i16>().expect("Failed to read FLAC samples as i16");

        assert!(audio.num_channels().get() > 0, "expected non-zero channels");
        assert!(audio.sample_rate().get() > 0, "expected non-zero sample rate");
        assert!(audio.total_samples().get() > 0, "expected non-empty samples");
    }

    #[test]
    fn test_flac_roundtrip_i16_fastest() {
        let sr = sample_rate!(44100);
        let sine_f32 = sine_wave::<f32>(440.0, StdDuration::from_secs_f64(0.1), sr, 0.5);
        let sine_i16 = sine_f32.to_format::<i16>();

        let out_path = std::env::temp_dir().join("audio_samples_io_flac_roundtrip.flac");
        {
            let file = File::create(&out_path).expect("create");
            let writer = BufWriter::new(file);
            write_flac(writer, &sine_i16, CompressionLevel::FASTEST).expect("write_flac");
        }

        let flac = FlacFile::open_with_options(&out_path, OpenOptions::default()).expect("open roundtrip file");
        let read_back = flac.read::<i16>().expect("read i16");

        assert_eq!(read_back.sample_rate(), sr);
        assert_eq!(read_back.num_channels().get(), sine_i16.num_channels().get(),);
        assert_eq!(
            read_back.samples_per_channel().get(),
            sine_i16.samples_per_channel().get(),
        );

        fs::remove_file(&out_path).ok();
    }

    #[test]
    fn test_lib_flac_info_and_read() {
        let flac_path = Path::new("resources/test.flac");

        let info = crate::info(flac_path).expect("lib::info should succeed");
        assert_eq!(info.file_type, FileType::FLAC);
        assert!(info.channels > 0);

        let samples = crate::read::<_, i16>(flac_path).expect("lib::read should succeed");
        assert!(samples.total_samples().get() > 0);
        assert_eq!(samples.num_channels().get() as u16, info.channels);
    }
}
