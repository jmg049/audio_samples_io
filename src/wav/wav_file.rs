use audio_samples::{
    AudioData, AudioSamples, I24, SampleType,
    traits::{ConvertFrom, StandardSample},
};
use core::fmt::{Display, Formatter, Result as FmtResult};
use memmap2::MmapOptions;
use ndarray::{Array1, Array2, ShapeBuilder};
use non_empty_slice::NonEmptyVec;
use std::{
    any::TypeId,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    mem,
    num::NonZeroU32,
    ops::Range,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    MAX_MMAP_SIZE, MAX_WAV_SIZE,
    error::{AudioIOError, AudioIOResult, ErrorPosition},
    traits::{AudioFile, AudioFileMetadata, AudioFileRead, AudioFileWrite, AudioInfoMarker},
    types::{AudioDataSource, BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType},
    wav::{
        FormatCode,
        chunks::{ChunkDesc, ChunkID, DATA_CHUNK, FMT_CHUNK, RIFF_CHUNK, WAVE_CHUNK},
        data::DataChunk,
        error::WavError,
        fmt::FmtChunk,
    },
};

#[derive(Debug, Clone)]
pub struct WavFileInfo {
    pub available_chunks: Vec<ChunkID>,
    pub encoding: FormatCode,
}

impl Display for WavFileInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "WAV File Info:")?;
        writeln!(f, "Encoding: {}", self.encoding)?;
        writeln!(f, "Available Chunks: {:?}", self.available_chunks)?;
        Ok(())
    }
}

impl AudioInfoMarker for WavFileInfo {}

/// High-level WAV file representation
#[derive(Debug)]
pub struct WavFile<'a> {
    /// Data source (owned or memory-mapped)
    data_source: AudioDataSource<'a>,
    /// File path if loaded from file
    file_path: PathBuf,
    /// Chunk Storage
    chunks: Vec<ChunkDesc>,
    /// fmt chunk byte range (excludes 8-byte chunk header)
    fmt_range: Range<usize>,
    /// data chunk byte range (excludes 8-byte chunk header)
    data_range: Range<usize>,
    /// Validated sample type
    sample_type: ValidatedSampleType,
    total_samples: usize,
}

impl<'a> WavFile<'a> {
    /// Get the file path if available
    pub fn file_path(&self) -> Option<&Path> {
        Some(self.file_path.as_path())
    }

    /// Get the FMT chunk
    ///
    /// # Panics
    ///
    /// This function will panic if the FMT chunk is not valid.
    /// However, this should never happen as it will have been validated during file opening.
    pub fn fmt_chunk(&self) -> FmtChunk<'_> {
        FmtChunk::from_bytes(self.fmt_bytes()).expect("fmt chunk validated during open")
    }

    /// Get the DATA chunk
    pub fn data(&self) -> DataChunk<'_> {
        DataChunk::from_bytes(self.data_bytes())
    }

    #[inline]
    fn fmt_bytes(&self) -> &[u8] {
        &self.data_source[self.fmt_range.clone()]
    }

    #[inline]
    fn data_bytes(&self) -> &[u8] {
        &self.data_source[self.data_range.clone()]
    }

    pub fn fact(&'a self) -> Result<(), WavError> {
        todo!()
    }

    pub fn list(&'a self) -> Result<(), WavError> {
        todo!()
    }

    pub fn plst(&'a self) -> Result<(), WavError> {
        todo!()
    }

    pub fn cue(&'a self) -> Result<(), WavError> {
        todo!()
    }

    #[allow(dead_code)]
    fn chunk_bytes(&self, chunk: &ChunkDesc) -> &[u8] {
        // Return entire chunk including header (8 bytes) + logical data (no padding)
        &self.data_source[chunk.offset..chunk.offset + 8 + chunk.logical_size]
    }

    pub const fn total_samples(&self) -> usize {
        self.total_samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.fmt_chunk().sample_rate()
    }

    pub fn is_mono(&self) -> bool {
        self.fmt_chunk().channels() == 1
    }

    pub fn is_mulit_channel(&self) -> bool {
        self.fmt_chunk().channels() > 1
    }
}

impl<'a> AudioFileMetadata for WavFile<'a> {
    fn open_metadata<P: AsRef<Path>>(path: P) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        // For metadata-only operations, we can use the same implementation as regular open
        // but with memory mapping enabled by default for efficiency
        Self::open_with_options(path, OpenOptions::default())
    }

    fn base_info(&self) -> AudioIOResult<BaseAudioInfo> {
        let fmt_chunk = self.fmt_chunk();
        let (_, channels, sample_rate, byte_rate, block_align, bits_per_sample) =
            fmt_chunk.fmt_chunk();
        let sample_rate = match NonZeroU32::new(sample_rate) {
            Some(sr) => sr,
            None => {
                return Err(AudioIOError::corrupted_data_simple(
                    "Invalid sample rate in FMT chunk",
                    "Sample rate cannot be zero",
                ));
            }
        };
        let (total_samples, duration) = {
            let data_chunk = self.data();
            // safety: channels is non-zero as per WAV spec
            let total_frames = data_chunk.total_frames(self.sample_type, unsafe {
                NonZeroU32::new_unchecked(channels as u32)
            });
            // Duration is based on frames, not total samples
            let duration = Duration::from_secs_f64(total_frames as f64 / sample_rate.get() as f64);
            (self.total_samples(), duration)
        };
        let file_type = FileType::WAV;
        let sample_type = self.sample_type();
        let bytes_per_sample = (bits_per_sample as usize / 8) as u16;

        Ok(BaseAudioInfo::new(
            sample_rate,
            channels,
            bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            total_samples,
            duration,
            file_type,
            sample_type.into(),
        ))
    }

    #[allow(refining_impl_trait)]
    fn specific_info(&self) -> WavFileInfo {
        WavFileInfo {
            available_chunks: self.chunks.iter().map(|c| c.id).collect(),
            encoding: self.fmt_chunk().format_code(),
        }
    }

    fn file_type(&self) -> FileType {
        FileType::WAV
    }

    fn file_path(&self) -> &Path {
        self.file_path.as_path()
    }

    fn total_samples(&self) -> usize {
        self.total_samples
    }

    fn duration(&self) -> AudioIOResult<Duration> {
        let base_info = self.base_info()?;
        Ok(base_info.duration)
    }

    fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    fn num_channels(&self) -> u16 {
        self.fmt_chunk().channels()
    }
}

impl<'a> AudioFile for WavFile<'a> {
    fn open_with_options<P: AsRef<Path>>(fp: P, options: OpenOptions) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        let path = fp.as_ref().to_path_buf();
        let file = File::open(&path)?;

        let file_size = file.metadata()?.len();

        if file_size > MAX_WAV_SIZE {
            return Err(AudioIOError::corrupted_data_simple(
                "File too large",
                format!("File size {file_size} exceeds maximum {MAX_WAV_SIZE} bytes"),
            ));
        }

        let use_mmap = options.use_memory_map && file_size <= MAX_MMAP_SIZE;

        let audio_data_source: AudioDataSource<'a> = if use_mmap {
            AudioDataSource::MemoryMapped(unsafe { MmapOptions::new().map(&file)? })
        } else {
            // Fallback to buffered read for large files or when mmap is disabled
            let mut buf_reader = BufReader::new(file);
            let mut bytes = Vec::new();
            buf_reader.read_to_end(&mut bytes)?;
            AudioDataSource::Owned(bytes)
        };
        let bytes = audio_data_source.as_bytes();

        if bytes.len() < 12 {
            return Err(AudioIOError::corrupted_data(
                "File too small to be a valid WAV file",
                format!("File size: {}", audio_data_source.len()),
                ErrorPosition::new(0).with_description("start of file"),
            ));
        }

        // 1. Parse the RIFF header + File size + WAVE identifier
        // Assume the first 12 bytes are the RIFF header
        let riff = ChunkID::new(
            bytes[0..4]
                .try_into()
                .expect("Guaranteed to be at least 12 bytes now"),
        );

        if riff != RIFF_CHUNK {
            return Err(AudioIOError::corrupted_data(
                "Data does not start with RIFF header",
                format!("Found: {riff:?}"),
                ErrorPosition::new(0).with_description("RIFF header at file start"),
            ));
        }

        let file_size = u32::from_le_bytes(
            bytes[4..8]
                .try_into()
                .expect("Guaranteed to be at least 12 bytes now"),
        );

        if file_size as usize + 8 > bytes.len() {
            return Err(AudioIOError::corrupted_data(
                "File size in RIFF header does not match actual file size",
                format!(
                    "Header size: {}, Actual size: {}",
                    file_size + 8,
                    bytes.len()
                ),
                ErrorPosition::new(4).with_description("file size field in RIFF header"),
            ));
        }

        let wave = ChunkID::new(
            bytes[8..12]
                .try_into()
                .expect("Guaranteed to be at least 12 bytes now"),
        );

        if wave != WAVE_CHUNK {
            return Err(AudioIOError::corrupted_data(
                "Data does not contain WAVE identifier after RIFF header",
                format!("Found: {wave:?}"),
                ErrorPosition::new(8).with_description("WAVE identifier after RIFF header"),
            ));
        }

        let mut chunks: Vec<ChunkDesc> = Vec::new();
        chunks.push(ChunkDesc {
            id: riff,
            offset: 0,
            logical_size: file_size as usize,
            total_size: file_size as usize + 8, // RIFF header + data
        });
        chunks.push(ChunkDesc {
            id: wave,
            offset: 8,
            logical_size: 4,
            total_size: 4,
        });

        // 2. Iterate through the rest of the file to find chunks
        let mut offset = 12;

        while offset + 8 <= bytes.len() {
            let id = ChunkID::new(bytes[offset..offset + 4].try_into().map_err(|_| {
                AudioIOError::corrupted_data(
                    "Cannot read chunk ID",
                    "Insufficient data for chunk header",
                    ErrorPosition::new(offset).with_description("chunk ID bytes"),
                )
            })?);
            let size_bytes = bytes[offset + 4..offset + 8].try_into().map_err(|_| {
                AudioIOError::corrupted_data(
                    "Cannot read chunk size",
                    "Insufficient data for chunk header",
                    ErrorPosition::new(offset + 4).with_description("chunk size bytes"),
                )
            })?;
            let size = u32::from_le_bytes(size_bytes) as usize;
            let padded = size.checked_add(size & 1).ok_or_else(|| {
                AudioIOError::corrupted_data(
                    "Integer overflow in chunk size calculation",
                    format!("Chunk size: {size}"),
                    ErrorPosition::new(offset + 4).with_description("chunk size field"),
                )
            })?;
            let header_and_data_size = 8_usize.checked_add(padded).ok_or_else(|| {
                AudioIOError::corrupted_data(
                    "Integer overflow in chunk total size calculation",
                    format!("Header size: 8, Data size: {padded}"),
                    ErrorPosition::new(offset).with_description("chunk header"),
                )
            })?;
            let end = offset.checked_add(header_and_data_size).ok_or_else(|| {
                AudioIOError::corrupted_data(
                    "Integer overflow in chunk end position calculation",
                    format!("Offset: {offset}, Size: {header_and_data_size}"),
                    ErrorPosition::new(offset).with_description("chunk position"),
                )
            })?;

            if end > bytes.len() {
                return Err(AudioIOError::corrupted_data(
                    "Chunk extends beyond end of file",
                    format!("Chunk {id:?} at offset {offset}"),
                    ErrorPosition::new(offset).with_description(format!("chunk {id:?}")),
                ));
            }

            chunks.push(ChunkDesc {
                id,
                offset,
                logical_size: size, // Original chunk size without padding
                total_size: header_and_data_size, // Header + data + padding for file positioning
            });

            offset = end;
        }

        // 3. Ensure fmt and data chunks are present
        let fmt_chunk_desc = chunks.iter().find(|c| c.id == FMT_CHUNK);
        let data_chunk_desc = chunks.iter().find(|c| c.id == DATA_CHUNK);

        let (fmt_range, sample_type) = match fmt_chunk_desc {
            Some(fmt_chunk) => {
                let start = fmt_chunk.offset + 8; // skip 8-byte header
                let end = start + fmt_chunk.logical_size; // exclude padding if any
                let fmt_chunk = FmtChunk::from_bytes_validated(&bytes[start..end])
                    .map_err(AudioIOError::WavError)?;
                let sample_type = fmt_chunk.actual_sample_type()?;
                (start..end, sample_type)
            }
            None => {
                return Err(AudioIOError::corrupted_data(
                    "FMT chunk not found in WAV file",
                    format!(
                        "Found chunks: {:?}",
                        chunks.iter().map(|c| c.id).collect::<Vec<_>>()
                    ),
                    ErrorPosition::new(12).with_description("chunk data section"),
                ));
            }
        };

        let data_range = match data_chunk_desc {
            Some(data_chunk) => {
                let start = data_chunk.offset + 8; // skip 8-byte header
                let end = start + data_chunk.logical_size; // exclude padding byte
                start..end
            }
            None => {
                return Err(AudioIOError::corrupted_data(
                    "DATA chunk not found in WAV file",
                    format!(
                        "Found chunks: {:?}",
                        chunks.iter().map(|c| c.id).collect::<Vec<_>>()
                    ),
                    ErrorPosition::new(12).with_description("chunk data section"),
                ));
            }
        };

        let total_samples = {
            let data_chunk = DataChunk::from_bytes(&bytes[data_range.clone()]);
            data_chunk.total_samples(sample_type)
        };

        let wav_file = WavFile {
            data_source: audio_data_source,
            file_path: path,
            chunks,
            fmt_range,
            data_range,
            sample_type,
            total_samples,
        };

        Ok(wav_file)
    }

    fn len(&self) -> u64 {
        self.data_source.len() as u64
    }
}

/// Parse just the WAV header from a seekable reader, returning audio metadata and the byte offset
/// where the `data` chunk payload begins.
///
/// Unlike [`WavFile::open_with_options`] this never loads or maps the audio payload — it stops
/// scanning as soon as it has seen the `fmt ` and `data` chunk headers.  On a typical WAV file the
/// header fits in the first 8 KiB read by the `BufReader`, so this triggers at most one read
/// syscall regardless of file size.
#[inline]
pub fn parse_wav_header_streaming<R: Read + Seek>(
    reader: &mut R,
) -> AudioIOResult<(BaseAudioInfo, u64)> {
    use crate::wav::chunks::RIFF_CHUNK;

    // ---- RIFF + WAVE header (12 bytes) ----
    let mut buf12 = [0u8; 12];
    reader.read_exact(&mut buf12).map_err(AudioIOError::from)?;
    if &buf12[0..4] != RIFF_CHUNK.as_bytes() {
        return Err(AudioIOError::corrupted_data_simple(
            "Not a RIFF file",
            "First 4 bytes are not 'RIFF'",
        ));
    }
    if &buf12[8..12] != b"WAVE" {
        return Err(AudioIOError::corrupted_data_simple(
            "Not a WAV file",
            "Bytes 8-12 are not 'WAVE'",
        ));
    }

    // ---- scan sub-chunks until we find fmt  and data ----
    let mut fmt_buf = [0u8; 40]; // enough for base (16) or extensible (40) fmt chunk
    let mut fmt_size: usize = 0;
    let mut have_fmt = false;
    let mut data_byte_offset: Option<u64> = None;
    let mut data_byte_count: Option<usize> = None;

    let mut chunk_hdr = [0u8; 8];

    loop {
        match reader.read_exact(&mut chunk_hdr) {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(AudioIOError::from(e)),
            Ok(_) => {}
        }
        let id = &chunk_hdr[0..4];
        let size = u32::from_le_bytes(match chunk_hdr[4..8].try_into() {
            Ok(b) => b,
            Err(_) => unreachable!(
                "chunk header is 8 bytes long, 4..8 is a 4-byte slice, try_into will always succeed"
            ),
        }) as usize;
        let padded = size + (size & 1); // RIFF chunks are padded to even sizes

        if id == b"fmt " {
            if !(16..=40).contains(&size) {
                return Err(AudioIOError::corrupted_data_simple(
                    "Invalid fmt chunk size",
                    format!("Expected 16 or 40, got {size}"),
                ));
            }
            reader
                .read_exact(&mut fmt_buf[..size])
                .map_err(AudioIOError::from)?;
            if size & 1 != 0 {
                reader
                    .read_exact(&mut [0u8; 1])
                    .map_err(AudioIOError::from)?;
            }
            fmt_size = size;
            have_fmt = true;
        } else if id == b"data" {
            // Current stream position is now at the first byte of audio data.
            data_byte_offset = Some(reader.stream_position().map_err(AudioIOError::from)?);
            data_byte_count = Some(size);
            break; // we have everything we need
        } else {
            // Skip this chunk (including any padding byte).
            reader
                .seek(SeekFrom::Current(padded as i64))
                .map_err(AudioIOError::from)?;
        }
    }

    if !have_fmt {
        return Err(AudioIOError::corrupted_data_simple(
            "No fmt chunk found",
            "WAV file must contain a fmt  chunk before data",
        ));
    }
    let data_byte_offset = data_byte_offset.ok_or_else(|| {
        AudioIOError::corrupted_data_simple("No data chunk found", "WAV file has no data chunk")
    })?;

    let data_byte_count = data_byte_count.ok_or_else(|| {
        AudioIOError::corrupted_data_simple(
            "Data chunk size missing",
            "Could not determine size of audio data from data chunk header",
        )
    })?;

    // ---- parse fmt bytes ----
    let fmt_chunk =
        FmtChunk::from_bytes_validated(&fmt_buf[..fmt_size]).map_err(AudioIOError::WavError)?;
    let sample_type: ValidatedSampleType = fmt_chunk
        .actual_sample_type()
        .map_err(AudioIOError::WavError)?;

    let (_, channels, sample_rate, byte_rate, block_align, bits_per_sample) = fmt_chunk.fmt_chunk();
    let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
        AudioIOError::corrupted_data_simple("Invalid sample rate", "Sample rate cannot be zero")
    })?;

    let bytes_per_sample = fmt_chunk.bytes_per_sample();
    let total_samples = if bytes_per_sample > 0 {
        data_byte_count / bytes_per_sample as usize
    } else {
        0
    };
    let total_frames = if channels > 0 {
        total_samples / channels as usize
    } else {
        0
    };
    let duration = Duration::from_secs_f64(total_frames as f64 / sample_rate.get() as f64);

    let info = BaseAudioInfo::new(
        sample_rate,
        channels,
        bits_per_sample,
        bytes_per_sample,
        byte_rate,
        block_align,
        total_samples,
        duration,
        FileType::WAV,
        sample_type.into(),
    );

    Ok((info, data_byte_offset))
}

impl<'a> AudioFileRead<'a> for WavFile<'a> {
    /// Reads all samples from the audio file
    fn read<T>(&'a self) -> AudioIOResult<AudioSamples<'a, T>>
    where
        T: StandardSample + 'static,
    {
        let data_chunk = self.data();
        let fmt_chunk = self.fmt_chunk();

        let sample_type = self.sample_type;
        let sample_rate = fmt_chunk.sample_rate();
        // safety: sample_rate is guaranteed to be non-zero due to WAV spec validation during fmt chunk parsing
        let sample_rate = unsafe { NonZeroU32::new_unchecked(sample_rate) };
        let num_channels = fmt_chunk.channels() as u32;
        // safety: num_channels == 0 would have been rejected during fmt chunk parsing.
        let num_channels = unsafe {
            NonZeroU32::new_unchecked(num_channels) // WAV spec requires channels > 0
        };

        match sample_type {
            ValidatedSampleType::U8 => {
                read_typed_internal::<u8, T>(&data_chunk, num_channels, sample_rate)
            }
            ValidatedSampleType::I16 => {
                read_typed_internal::<i16, T>(&data_chunk, num_channels, sample_rate)
            }
            ValidatedSampleType::I24 => {
                read_typed_internal::<I24, T>(&data_chunk, num_channels, sample_rate)
            }
            ValidatedSampleType::I32 => {
                read_typed_internal::<i32, T>(&data_chunk, num_channels, sample_rate)
            }
            ValidatedSampleType::F32 => {
                read_typed_internal::<f32, T>(&data_chunk, num_channels, sample_rate)
            }
            ValidatedSampleType::F64 => {
                read_typed_internal::<f64, T>(&data_chunk, num_channels, sample_rate)
            }
        }
    }

    fn read_into<T>(&'a self, audio: &mut AudioSamples<'a, T>) -> AudioIOResult<()>
    where
        T: StandardSample + 'static,
    {
        let data_chunk = self.data();

        match self.sample_type {
            ValidatedSampleType::U8 => read_into_typed_internal::<u8, T>(&data_chunk, audio), // technicaly not part of the wav spec, but it does not disallow it either, so we support it
            ValidatedSampleType::I16 => read_into_typed_internal::<i16, T>(&data_chunk, audio),
            ValidatedSampleType::I24 => read_into_typed_internal::<I24, T>(&data_chunk, audio),
            ValidatedSampleType::I32 => read_into_typed_internal::<i32, T>(&data_chunk, audio),
            ValidatedSampleType::F32 => read_into_typed_internal::<f32, T>(&data_chunk, audio),
            ValidatedSampleType::F64 => read_into_typed_internal::<f64, T>(&data_chunk, audio),
        }
    }
}

/// Wrap interleaved WAV samples into an AudioSamples without a deinterleave copy.
///
/// WAV stores samples interleaved: [L0, R0, L1, R1, …].  Rather than physically rearranging
/// memory into planar layout, multi-channel audio is represented as an F-order (column-major)
/// Array2 with shape (channels, frames) and strides (1, channels), which matches the
/// interleaved memory layout exactly.
fn build_samples_from_interleaved_vec<'a, T>(
    interleaved_data: NonEmptyVec<T>,
    num_channels: NonZeroU32,
    sample_rate: NonZeroU32,
) -> AudioIOResult<AudioSamples<'a, T>>
where
    T: StandardSample + 'static,
{
    // SAFETY: sample_rate comes from validated WAV header which requires non-zero sample rate
    if num_channels.get() == 1 {
        // Mono: data is already in correct format
        AudioSamples::new_mono(Array1::from_vec(interleaved_data.into_vec()), sample_rate)
            .map_err(Into::into)
    } else {
        let total_samples = interleaved_data.len();
        let frames = total_samples.get() / num_channels.get() as usize;

        if frames == 0 {
            return Err(AudioIOError::corrupted_data_simple(
                "No frames in audio data",
                format!("total_samples={total_samples}, channels={num_channels}"),
            ));
        }

        // Fix C: wrap the interleaved Vec directly as an F-order (column-major) Array2.
        // An F-order array with logical shape (channels, frames) stores data as
        // [s[0,0], s[1,0], s[0,1], s[1,1], …] — exactly WAV's interleaved layout.
        // This avoids the deinterleave allocation and scatter-write entirely.
        let arr = Array2::from_shape_vec(
            (num_channels.get() as usize, frames).f(),
            interleaved_data.into_vec(), // Fix B: move, no copy
        )
        .map_err(|e| AudioIOError::corrupted_data_simple("Array shape error", e.to_string()))?;

        AudioSamples::new_multi_channel(arr, sample_rate).map_err(Into::into)
    }
}

fn read_into_typed_internal<'a, S, T>(
    data_chunk: &DataChunk<'a>,
    audio: &mut AudioSamples<'a, T>,
) -> AudioIOResult<()>
where
    S: StandardSample + 'static,
    T: StandardSample + ConvertFrom<S> + 'static,
{
    let bytes_per_sample = S::BITS as usize / 8;
    if !data_chunk.len().is_multiple_of(bytes_per_sample) {
        return Err(AudioIOError::corrupted_data_simple(
            "Data chunk size not aligned to sample size",
            format!(
                "Data chunk size {} not divisible by sample size {}",
                data_chunk.len(),
                bytes_per_sample
            ),
        ));
    }

    let converted = data_chunk.read_samples::<S, T>()?;
    let num_channels = audio.num_channels();

    if !converted
        .len()
        .get()
        .is_multiple_of(num_channels.get() as usize)
    {
        return Err(AudioIOError::corrupted_data_simple(
            "Channel alignment error",
            format!(
                "Sample count {} not divisible by channel count {}",
                converted.len(),
                num_channels,
            ),
        ));
    }

    if converted.len() != audio.total_samples() {
        return Err(AudioIOError::corrupted_data_simple(
            "Sample count mismatch",
            format!(
                "Converted sample count {} does not match target audio sample count {}",
                converted.len(),
                audio.total_samples(),
            ),
        ));
    }

    // For mono audio, data is already in correct format
    if audio.is_mono() {
        audio.replace_with_vec(&converted).map_err(|e| e.into())
    } else {
        // Multi-channel: deinterleave the converted data before replacing
        // Use optimized deinterleave from audio_samples
        let planar_data =
            audio_samples::simd_conversions::deinterleave_multi_vec(&converted, num_channels)
                .map_err(|e| {
                    AudioIOError::corrupted_data_simple("Deinterleave failed", e.to_string())
                })?;
        audio.replace_with_vec(&planar_data).map_err(|e| e.into())
    }
}

fn read_typed_internal<'a, S, T>(
    data_chunk: &DataChunk<'a>,
    num_channels: NonZeroU32,
    sample_rate: NonZeroU32,
) -> AudioIOResult<AudioSamples<'a, T>>
where
    S: StandardSample + 'static,
    T: StandardSample + ConvertFrom<S> + 'static,
{
    let bytes_per_sample = S::BYTES as usize;
    if !data_chunk.len().is_multiple_of(bytes_per_sample) {
        return Err(AudioIOError::corrupted_data_simple(
            "Data chunk size not aligned to sample size",
            format!(
                "Data chunk size {} not divisible by sample size {}",
                data_chunk.len(),
                bytes_per_sample
            ),
        ));
    }

    let converted = data_chunk.read_samples::<S, T>()?;

    if !converted
        .len()
        .get()
        .is_multiple_of(num_channels.get() as usize)
    {
        return Err(AudioIOError::corrupted_data_simple(
            "Channel alignment error",
            format!(
                "Sample count {} not divisible by channel count {}",
                converted.len(),
                num_channels,
            ),
        ));
    }

    build_samples_from_interleaved_vec(converted, num_channels, sample_rate)
}

// Helper functions for WAV writing

/// Maps SampleType to WAV FormatCode
///
/// Returns `None` if the SampleType is not supported for WAV writing.
/// This function validates that the sample type is valid before conversion.
const fn sample_type_to_format(sample_type: SampleType) -> Option<FormatCode> {
    match sample_type {
        SampleType::U8 | SampleType::I16 | SampleType::I24 | SampleType::I32 => {
            Some(FormatCode::Pcm)
        }
        SampleType::F32 | SampleType::F64 => Some(FormatCode::IeeeFloat),
        _ => None,
    }
}

/// Get SampleType from AudioSample type parameter
const fn get_sample_type<T>() -> SampleType
where
    T: StandardSample,
{
    T::SAMPLE_TYPE
}

/// Write 16-byte base FMT chunk
fn write_base_fmt<T, W>(writer: &mut W, channels: u16, sample_rate: u32) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let sample_type = get_sample_type::<T>();
    let format_code = sample_type_to_format(sample_type)
        .ok_or(AudioIOError::WavError(WavError::UnsupportedSampleType))?;
    let bits_per_sample = T::BITS as u16;
    let bytes_per_sample = T::BYTES as u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;

    // FMT chunk header
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // Chunk size (16 bytes)

    // FMT chunk data (16 bytes)
    writer.write_all(&format_code.as_u16().to_le_bytes())?; // Format code
    writer.write_all(&channels.to_le_bytes())?; // Channels
    writer.write_all(&sample_rate.to_le_bytes())?; // Sample rate
    writer.write_all(&byte_rate.to_le_bytes())?; // Byte rate
    writer.write_all(&block_align.to_le_bytes())?; // Block align
    writer.write_all(&bits_per_sample.to_le_bytes())?; // Bits per sample

    Ok(())
}

/// Determine if extensible format is needed
const fn needs_extensible_format<T>(channels: u16) -> bool
where
    T: StandardSample,
{
    // Use extensible format for more than 2 channels or non-standard bit depths.
    // 8-bit, 16-bit, and 32-bit PCM/float formats use base format for 1-2 channels
    // for maximum compatibility (e.g. soundfile/libsndfile doesn't support
    // WAVE_FORMAT_EXTENSIBLE with 8-bit PCM).
    channels > 2 || (T::BITS != 8 && T::BITS != 16 && T::BITS != 32)
}

/// Write 40-byte extensible FMT chunk
fn write_extensible_fmt<T, W>(writer: &mut W, channels: u16, sample_rate: u32) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let sample_type = get_sample_type::<T>();
    let format_code = sample_type_to_format(sample_type)
        .ok_or(AudioIOError::WavError(WavError::UnsupportedSampleType))?;
    let bits_per_sample = T::BITS as u16;
    let bytes_per_sample = T::BYTES as u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;

    // Windows standard speaker position channel masks
    // Based on WAVE_FORMAT_EXTENSIBLE specification
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
            // For >8 channels, saturate to all bits set
            // This is a reasonable fallback for non-standard channel configurations
            if channels < 32 {
                (1u32 << channels) - 1
            } else {
                0xFFFFFFFF
            }
        }
    };

    // FMT chunk header
    writer.write_all(b"fmt ")?;
    writer.write_all(&40u32.to_le_bytes())?; // Chunk size (40 bytes)

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
    writer.write_all(&channel_mask.to_le_bytes())?; // Channel mask

    // Sub-format GUID (16 bytes)
    // Standard KSDATAFORMAT_SUBTYPE_PCM:       {00000001-0000-0010-8000-00AA00389B71}
    // Standard KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: {00000003-0000-0010-8000-00AA00389B71}
    // In memory: Data1 (4 bytes LE), Data2 (2 bytes LE), Data3 (2 bytes LE), Data4 (8 bytes BE)
    let mut sub_format = [0u8; 16];
    // Data1: format_code as u32 LE (e.g. 1 for PCM, 3 for IEEE float)
    sub_format[0..4].copy_from_slice(&(u32::from(format_code.as_u16())).to_le_bytes());
    // Data2: 0x0000
    sub_format[4..6].copy_from_slice(&0u16.to_le_bytes());
    // Data3: 0x0010
    sub_format[6..8].copy_from_slice(&0x0010u16.to_le_bytes());
    // Data4: 0x8000-00AA00389B71 (big-endian)
    sub_format[8..16].copy_from_slice(&[0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]);
    writer.write_all(&sub_format)?;

    Ok(())
}

/// Build an interleaved byte buffer for WAV output and write it in one go.
/// Mono fast-path uses the underlying contiguous bytes view; multi-channel
/// uses optimized interleave functions for better cache locality.
fn write_audio_data_interleaved<T, W>(writer: &mut W, audio: &AudioSamples<T>) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let num_channels = audio.num_channels();

    // Mono data is already laid out correctly; respect I24 packing via AudioSamples::bytes
    if audio.is_mono() {
        let bytes = audio.bytes()?;
        writer.write_all(bytes.as_slice())?;
        return Ok(());
    }

    // Fast path: F-order (Fortran-contiguous) multi-channel arrays already have interleaved
    // layout in memory — [L0, R0, L1, R1, …] — matching WAV's on-disk format exactly.
    // On little-endian platforms the in-memory bytes need no reordering for any integer or
    // float type (except I24 which uses a non-standard 4-byte in-memory representation).
    #[cfg(target_endian = "little")]
    if TypeId::of::<T>() != TypeId::of::<I24>() {
        if let AudioData::Multi(ref m) = audio.data {
            let view = m.as_view();
            if !view.is_standard_layout() {
                if let Some(slice) = view.as_slice_memory_order() {
                    // Safety: T is a plain numeric type (StandardSample, not I24).
                    // Casting any &[T] to &[u8] is valid — bytes are always initialised.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            slice.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(slice),
                        )
                    };
                    writer.write_all(bytes)?;
                    return Ok(());
                }
            }
        }
    }

    let bytes_per_sample = if TypeId::of::<T>() == TypeId::of::<I24>() {
        3usize
    } else {
        mem::size_of::<T>()
    };

    // C-order (standard-layout) path: stream interleaved bytes via a small tile buffer.
    //
    // The naive approach (view.t().as_standard_layout()) allocates a 10 MB intermediate Vec
    // upfront, incurring ~2500 OS page faults (~2.5 ms) before any data moves.  Instead, we
    // reuse a single fixed-size tile buffer (~64 KB), fill it with interleaved samples one
    // tile at a time, and write each tile to the output — zero large allocations, sequential
    // reads within each channel's tile window, sequential writes.
    #[cfg(target_endian = "little")]
    if TypeId::of::<T>() != TypeId::of::<I24>() {
        if let AudioData::Multi(ref m) = audio.data {
            let view = m.as_view();
            if view.is_standard_layout() {
                use std::mem::MaybeUninit;

                let channels = view.shape()[0];
                let frames = view.shape()[1];
                let sample_size = mem::size_of::<T>();

                // Tile size: aim for ~64 KB of output data to stay in L1/L2 cache.
                const TARGET_TILE_BYTES: usize = 512 * 1024;
                let tile_frames = (TARGET_TILE_BYTES / (channels * sample_size)).max(1);

                // One contiguous slice per channel (rows are contiguous in C-order).
                let rows: Vec<&[T]> = (0..channels)
                    .map(|c| view.row(c).to_slice().expect("C-order row is contiguous"))
                    .collect();

                // Pre-allocate tile buffer once — avoids per-tile page faults.
                let tile_capacity = tile_frames * channels;

                // Allocate uninitialised buffer
                let mut tile: Vec<MaybeUninit<T>> = Vec::with_capacity(tile_capacity);

                // SAFETY: we will initialise every element before reading
                unsafe { tile.set_len(tile_capacity) };

                for tile_start in (0..frames).step_by(tile_frames) {
                    let tile_end = (tile_start + tile_frames).min(frames);
                    let actual_frames = tile_end - tile_start;
                    // Channels-outer loop: each channel's tile window is read sequentially,
                    // letting the HW prefetcher stream contiguous cache lines.  Writes land
                    // at stride=channels in the tile buffer (still within a small working set).
                    for c in 0..channels {
                        for (i, f) in (tile_start..tile_end).enumerate() {
                            // Safety: f < frames and c < channels, both in bounds.
                            tile[i * channels + c].write(rows[c][f]);
                        }
                    }
                    // Safety: T is a plain numeric type (not I24), LE platform; tile
                    // slice is fully initialised above.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            tile.as_ptr().cast::<u8>(),
                            actual_frames * channels * sample_size,
                        )
                    };
                    writer.write_all(bytes)?;
                }
                return Ok(());
            }
        }
    }

    // Fallback: materialise an interleaved Vec then serialise sample-by-sample.
    // Handles I24, big-endian platforms, and non-contiguous array layouts.
    let interleaved = audio.data.as_interleaved_vec();
    let total_samples = interleaved.len();

    const TARGET_CHUNK_BYTES: usize = 256 * 1024;
    let chunk_samples = TARGET_CHUNK_BYTES
        .checked_div(bytes_per_sample)
        .ok_or(AudioIOError::corrupted_data_simple(
            "Chunk size calculation overflow",
            "TARGET_CHUNK_BYTES / bytes_per_sample",
        ))?
        .max(num_channels.get() as usize);

    let mut buf = vec![0u8; chunk_samples * bytes_per_sample];

    let mut sample_start = 0;
    while sample_start < total_samples.get() {
        let remaining = total_samples.get() - sample_start;
        let samples_this_chunk = remaining.min(chunk_samples);
        let bytes_this_chunk = samples_this_chunk * bytes_per_sample;

        let mut write_idx = 0;
        for sample in interleaved
            .iter()
            .skip(sample_start)
            .take(samples_this_chunk)
        {
            let bytes = sample.to_le_bytes();
            let dst = &mut buf[write_idx..write_idx + bytes_per_sample];
            dst.copy_from_slice(bytes.as_ref());
            write_idx += bytes_per_sample;
        }

        debug_assert_eq!(write_idx, bytes_this_chunk);
        writer.write_all(&buf[..bytes_this_chunk])?;
        sample_start += samples_this_chunk;
    }

    Ok(())
}

// Write complete WAV file to a writer
pub(crate) fn write_wav<T, W>(writer: W, audio: &AudioSamples<T>) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let sample_rate = audio.sample_rate().get();
    let channels = audio.num_channels().get() as u16;
    let bytes_per_sample_disk = if TypeId::of::<T>() == TypeId::of::<I24>() {
        3usize
    } else {
        mem::size_of::<T>()
    };
    let data_size = audio
        .samples_per_channel()
        .get()
        .checked_mul(audio.num_channels().get() as usize)
        .and_then(|v| v.checked_mul(bytes_per_sample_disk))
        .ok_or_else(|| {
            AudioIOError::corrupted_data_simple(
                "Byte size overflow during header calculation",
                format!(
                    "channels={}, samples_per_channel={}, bytes_per_sample={}",
                    channels,
                    audio.samples_per_channel().get(),
                    bytes_per_sample_disk
                ),
            )
        })?;

    // Calculate padded data size (must be even)
    let padded_data_size = if data_size % 2 == 1 {
        data_size + 1
    } else {
        data_size
    };

    // Determine FMT chunk size
    let fmt_chunk_size = if needs_extensible_format::<T>(channels) {
        40
    } else {
        16
    };
    let fmt_total_size = 8 + fmt_chunk_size; // chunk header + data

    // Calculate total file size
    let file_size = 4 + fmt_total_size + 8 + padded_data_size; // WAVE + FMT + DATA

    // Always buffer writes to avoid caller-dependent performance cliffs
    let mut writer = BufWriter::new(writer);

    // Write RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&(file_size as u32).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // Write FMT chunk
    if needs_extensible_format::<T>(channels) {
        write_extensible_fmt::<T, _>(&mut writer, channels, sample_rate)?;
    } else {
        write_base_fmt::<T, _>(&mut writer, channels, sample_rate)?;
    }

    // Write DATA chunk header
    writer.write_all(b"data")?;
    writer.write_all(&(data_size as u32).to_le_bytes())?;

    // Write audio data (interleaved for multi-channel)
    write_audio_data_interleaved(&mut writer, audio)?;

    // Add padding byte if needed
    if data_size % 2 == 1 {
        writer.write_all(&[0])?;
    }

    writer.flush()?;
    Ok(())
}

impl<'a> AudioFileWrite for WavFile<'a> {
    fn write<P, T>(&mut self, out_fp: P) -> AudioIOResult<()>
    where
        P: AsRef<Path>,
        T: StandardSample + 'static,
    {
        // Read audio data as the target type T
        let audio = self.read::<T>()?;

        // Create output file with buffered writer
        let file = File::create(out_fp)?;
        let buf_writer = BufWriter::new(file);

        // Write WAV using the core function
        write_wav(buf_writer, &audio)?;

        Ok(())
    }
}

impl Display for WavFile<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "WAV File:")?;
        writeln!(f, "File Path: {:?}", self.file_path)?;
        writeln!(f, "Chunks:")?;
        for chunk in &self.chunks {
            writeln!(
                f,
                "  ID: {}, Offset: {}, Logical Size: {}, Total Size: {}",
                chunk.id, chunk.offset, chunk.logical_size, chunk.total_size
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod wav_tests {
    use audio_samples::sample_rate;
    use non_empty_slice::NonEmptySlice;

    use crate::wav::FormatCode;

    use super::*;

    #[allow(dead_code)]
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
    fn test_wav_open() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file = WavFile::open_with_options(wav_path, OpenOptions::default());
        assert!(wav_file.is_ok(), "Failed to open test WAV file");
    }

    #[test]
    fn test_wav_fmt_chunk() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file = WavFile::open_with_options(wav_path, OpenOptions::default())
            .expect("Failed to open test WAV file");
        let fmt_chunk = wav_file.fmt_chunk();
        assert_eq!(
            fmt_chunk.format_code(),
            FormatCode::Pcm,
            "Format code mismatch"
        );
        assert_eq!(fmt_chunk.sample_rate(), 44100, "Sample rate mismatch");
        assert_eq!(fmt_chunk.channels(), 2, "Channel count mismatch");
    }

    #[test]
    fn test_wav_data_chunk() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file = WavFile::open_with_options(wav_path, OpenOptions::default())
            .expect("Failed to open test WAV file");

        let audio = wav_file.read::<i16>();
        assert!(
            audio.is_ok(),
            "Failed to read audio samples from DATA chunk"
        );
    }

    #[test]
    fn test_wav_properties() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file = WavFile::open_with_options(wav_path, OpenOptions::default())
            .expect("Failed to open test WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(
            base_info.sample_rate,
            sample_rate!(44100),
            "Sample rate mismatch"
        );
        assert_eq!(base_info.channels, 2, "Channel count mismatch");
        assert_eq!(base_info.bits_per_sample, 16, "Bits per sample mismatch");

        let specific_info = wav_file.specific_info();
        assert_eq!(specific_info.encoding, FormatCode::Pcm, "Encoding mismatch");
        assert!(
            specific_info.available_chunks.contains(&FMT_CHUNK),
            "FMT chunk should be available"
        );
        assert!(
            specific_info.available_chunks.contains(&DATA_CHUNK),
            "DATA chunk should be available"
        );

        println!("Base Info: {base_info:#}");
    }

    #[test]
    fn test_wav_write_i16() {
        use audio_samples::{AudioTypeConversion, sine_wave};
        use std::fs;

        // Generate a sine wave
        let sample_rate = sample_rate!(44100);
        let frequency = 440.0;
        let duration = Duration::from_secs_f64(1.0); // 1 second
        let amplitude = 0.5;
        let sine_samples = sine_wave::<f32>(frequency, duration, sample_rate, amplitude);
        let sine_i16 = sine_samples.to_format::<i16>();

        // Write to file
        let output_path = std::env::temp_dir().join("test_write_i16.wav");
        write_wav(
            std::io::BufWriter::new(
                std::fs::File::create(&output_path).expect("Failed to create output file"),
            ),
            &sine_i16,
        )
        .expect("Failed to write WAV file");

        // Verify file was created and has reasonable size
        let metadata = fs::metadata(&output_path).expect("Failed to get file metadata");
        assert!(metadata.len() > 44, "WAV file too small"); // At least header size

        // Read back and verify
        let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
            .expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(44100));
        assert_eq!(base_info.channels, 1);
        assert_eq!(base_info.bits_per_sample, 16);

        let read_samples = wav_file.read::<i16>().expect("Failed to read samples");
        assert_eq!(read_samples.total_samples(), sine_i16.total_samples());

        // // Clean up
        // fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_f32() {
        use audio_samples::sine_wave;
        use std::fs;

        // Generate a sine wave
        let sample_rate = sample_rate!(48000);
        let frequency = 1000.0;
        let duration = Duration::from_secs_f64(0.5); // 0.5 seconds
        let amplitude = 0.8;
        let sine_samples = sine_wave::<f32>(frequency, duration, sample_rate, amplitude);

        // Write to file
        let output_path = std::env::temp_dir().join("test_write_f32.wav");
        write_wav(
            std::io::BufWriter::new(
                std::fs::File::create(&output_path).expect("Failed to create output file"),
            ),
            &sine_samples,
        )
        .expect("Failed to write WAV file");

        // Read back and verify
        let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
            .expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(48000));
        assert_eq!(base_info.channels, 1);
        assert_eq!(base_info.bits_per_sample, 32);

        let read_samples = wav_file.read::<f32>().expect("Failed to read samples");
        assert_eq!(read_samples.total_samples(), sine_samples.total_samples());

        // Verify format is IEEE Float
        let fmt_chunk = wav_file.fmt_chunk();
        assert_eq!(fmt_chunk.format_code(), FormatCode::IeeeFloat);

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_read_i24_roundtrip() {
        use audio_samples::sine_wave;
        use std::fs;

        let sample_rate = sample_rate!(48_000);
        let duration = Duration::from_millis(20);
        let audio = sine_wave::<I24>(440.0, duration, sample_rate, 0.5);

        let output_path = std::env::temp_dir().join(format!(
            "test_read_i24_roundtrip_{}.wav",
            std::process::id()
        ));
        write_wav(
            std::io::BufWriter::new(
                std::fs::File::create(&output_path).expect("Failed to create output file"),
            ),
            &audio,
        )
        .expect("Failed to write I24 WAV file");

        let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
            .expect("Failed to reopen written WAV file");
        let read_samples = wav_file.read::<I24>().expect("Failed to read I24 samples");

        assert_eq!(
            read_samples.total_samples(),
            audio.total_samples(),
            "Roundtrip sample count mismatch"
        );
        assert_eq!(read_samples.num_channels(), audio.num_channels());
        assert_eq!(read_samples.sample_rate(), audio.sample_rate());

        let original = audio.to_interleaved_vec();
        let roundtrip = read_samples.to_interleaved_vec();
        assert_eq!(original, roundtrip, "I24 roundtrip data mismatch");

        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_stereo() {
        use audio_samples::sine_wave;
        use std::fs;

        // Generate stereo sine waves (left: 440Hz, right: 880Hz)
        let sample_rate = sample_rate!(44100);
        let duration = Duration::from_secs_f64(0.25);
        let left = sine_wave::<f32>(440.0, duration, sample_rate, 0.6);
        let right = sine_wave::<f32>(880.0, duration, sample_rate, 0.4);

        // Combine into stereo
        let stereo =
            audio_samples::AudioEditing::stack(NonEmptySlice::new(&[left, right]).unwrap())
                .expect("Failed to create stereo");

        // Write to file
        let output_path = std::env::temp_dir().join("test_write_stereo.wav");
        write_wav(
            std::io::BufWriter::new(
                std::fs::File::create(&output_path).expect("Failed to create output file"),
            ),
            &stereo,
        )
        .expect("Failed to write stereo WAV file");

        // Read back and verify
        let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
            .expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(44100));
        assert_eq!(base_info.channels, 2);
        assert_eq!(base_info.bits_per_sample, 32);

        let read_samples = wav_file.read::<f32>().expect("Failed to read samples");
        assert_eq!(read_samples.total_samples(), stereo.total_samples());
        assert_eq!(read_samples.num_channels().get(), 2);

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_type_conversion() {
        use audio_samples::{AudioTypeConversion, sine_wave};
        use std::fs;

        // Generate f32 sine wave
        let sample_rate = sample_rate!(44100);
        let sine_f32 = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.1), sample_rate, 0.7);

        // Write as i16 (should convert)
        let output_path = std::env::temp_dir().join("test_conversion.wav");
        let sine_i16 = sine_f32.to_format::<i16>();
        write_wav(
            std::io::BufWriter::new(
                std::fs::File::create(&output_path).expect("Failed to create output file"),
            ),
            &sine_i16,
        )
        .expect("Failed to write converted WAV file");

        // Verify it's written as i16 PCM
        let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
            .expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.bits_per_sample, 16);

        let fmt_chunk = wav_file.fmt_chunk();
        assert_eq!(fmt_chunk.format_code(), FormatCode::Pcm);

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_audiofilewrite_trait() {
        use audio_samples::sine_wave;
        use std::fs;

        // Create a test WAV file first
        let sample_rate = sample_rate!(22050);
        let sine_samples = sine_wave::<i16>(330.0, Duration::from_secs_f64(0.2), sample_rate, 0.5);
        let input_path = std::env::temp_dir().join("test_input.wav");
        write_wav(
            std::io::BufWriter::new(
                std::fs::File::create(&input_path).expect("Failed to create input file"),
            ),
            &sine_samples,
        )
        .expect("Failed to write input WAV file");

        // Open the WAV file and use the trait method to write as f32
        let mut wav_file = WavFile::open_with_options(&input_path, OpenOptions::default())
            .expect("Failed to open input WAV file");

        let output_path = std::env::temp_dir().join("test_trait_output.wav");
        wav_file
            .write::<_, f32>(&output_path)
            .expect("Failed to write using trait method");

        // Verify the output is f32
        let output_wav = WavFile::open_with_options(&output_path, OpenOptions::default())
            .expect("Failed to open output WAV file");

        let base_info = output_wav.base_info().expect("Failed to get base info");
        assert_eq!(base_info.bits_per_sample, 32);
        assert_eq!(base_info.sample_rate, sample_rate!(22050));

        let fmt_chunk = output_wav.fmt_chunk();
        assert_eq!(fmt_chunk.format_code(), FormatCode::IeeeFloat);

        // Clean up
        fs::remove_file(&input_path).ok();
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_read_roundtrip_validation() {
        use audio_samples::{AudioTypeConversion, sine_wave};
        use std::fs;

        // Test multiple sample types with comprehensive validation
        let sample_rate = sample_rate!(44100);
        let duration = Duration::from_secs_f64(0.5);
        let base_sine = sine_wave::<f32>(440.0, duration, sample_rate, 0.5);

        // Test cases: (type_name, bits_per_sample, format_code)
        let test_cases = [
            ("i16", 16, FormatCode::Pcm),
            ("i32", 32, FormatCode::Pcm),
            ("f32", 32, FormatCode::IeeeFloat),
        ];

        for (type_name, expected_bits, expected_format) in test_cases.iter() {
            let output_path = std::env::temp_dir().join(format!("test_roundtrip_{type_name}.wav"));

            match *type_name {
                "i16" => {
                    let samples = base_sine.to_format::<i16>();
                    write_wav(
                        std::io::BufWriter::new(
                            std::fs::File::create(&output_path)
                                .expect("Failed to create output file"),
                        ),
                        &samples,
                    )
                    .expect("Failed to write WAV file");

                    // Validate WAV structure
                    let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
                        .expect("Failed to open WAV file");
                    let base_info = wav_file.base_info().expect("Failed to get WAV base info");
                    let fmt_chunk = wav_file.fmt_chunk();

                    assert_eq!(base_info.sample_rate, sample_rate!(44100));
                    assert_eq!(base_info.bits_per_sample, *expected_bits);
                    assert_eq!(fmt_chunk.format_code(), *expected_format);

                    // Read back and verify data integrity
                    let read_samples = wav_file.read::<i16>().expect("Failed to read WAV samples");
                    let read_bytes = read_samples
                        .bytes()
                        .expect("Failed to get bytes from read samples");
                    let written_bytes = samples
                        .bytes()
                        .expect("Failed to get bytes from written samples");
                    assert_eq!(read_bytes.as_slice(), written_bytes.as_slice());
                }
                "i32" => {
                    let samples = base_sine.to_format::<i32>();
                    write_wav(
                        std::io::BufWriter::new(
                            std::fs::File::create(&output_path)
                                .expect("Failed to create output file"),
                        ),
                        &samples,
                    )
                    .expect("Failed to write WAV file");

                    // Validate WAV structure
                    let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
                        .expect("Failed to open WAV file");
                    let base_info = wav_file.base_info().expect("Failed to get WAV base info");
                    let fmt_chunk = wav_file.fmt_chunk();

                    assert_eq!(base_info.sample_rate, sample_rate!(44100));
                    assert_eq!(base_info.bits_per_sample, *expected_bits);
                    assert_eq!(fmt_chunk.format_code(), *expected_format);

                    // Read back and verify data integrity
                    let read_samples = wav_file.read::<i32>().expect("Failed to read WAV samples");
                    let read_bytes = read_samples
                        .bytes()
                        .expect("Failed to get bytes from read samples");
                    let written_bytes = samples
                        .bytes()
                        .expect("Failed to get bytes from written samples");
                    assert_eq!(read_bytes.as_slice(), written_bytes.as_slice());
                }
                "f32" => {
                    write_wav(
                        std::io::BufWriter::new(
                            std::fs::File::create(&output_path)
                                .expect("Failed to create output file"),
                        ),
                        &base_sine,
                    )
                    .expect("Failed to write WAV file");

                    // Validate WAV structure
                    let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
                        .expect("Failed to open WAV file");
                    let base_info = wav_file.base_info().expect("Failed to get WAV base info");
                    let fmt_chunk = wav_file.fmt_chunk();

                    assert_eq!(base_info.sample_rate, sample_rate!(44100));
                    assert_eq!(base_info.bits_per_sample, *expected_bits);
                    assert_eq!(fmt_chunk.format_code(), *expected_format);

                    // Read back and verify data integrity (with small tolerance for f32)
                    let read_samples = wav_file.read::<f32>().expect("Failed to read WAV samples");
                    let orig_bytes = base_sine
                        .bytes()
                        .expect("Failed to get bytes from original samples");
                    let read_bytes = read_samples
                        .bytes()
                        .expect("Failed to get bytes from read samples");

                    let orig_f32: &[f32] = bytemuck::cast_slice(orig_bytes.as_slice());
                    let read_f32: &[f32] = bytemuck::cast_slice(read_bytes.as_slice());

                    for (orig, read) in orig_f32.iter().zip(read_f32.iter()) {
                        assert!(
                            (orig - read).abs() < 1e-6,
                            "f32 samples should be nearly identical"
                        );
                    }
                }
                _ => unreachable!(),
            }

            // Verify file is readable by external tools (basic structure check)
            let file_bytes = std::fs::read(&output_path).expect("Failed to read output file");
            assert!(
                file_bytes.len() > 44,
                "WAV file should have proper header + data"
            );
            assert_eq!(&file_bytes[0..4], b"RIFF");
            assert_eq!(&file_bytes[8..12], b"WAVE");

            // Find and validate FMT and DATA chunks exist
            let mut has_fmt = false;
            let mut has_data = false;
            let mut pos = 12;

            while pos + 8 <= file_bytes.len() {
                let chunk_id = &file_bytes[pos..pos + 4];
                let chunk_size = u32::from_le_bytes([
                    file_bytes[pos + 4],
                    file_bytes[pos + 5],
                    file_bytes[pos + 6],
                    file_bytes[pos + 7],
                ]) as usize;

                if chunk_id == b"fmt " {
                    has_fmt = true;
                    assert!(chunk_size >= 16, "FMT chunk should be at least 16 bytes");
                } else if chunk_id == b"data" {
                    has_data = true;
                    assert!(chunk_size > 0, "DATA chunk should contain audio data");
                }

                pos += 8 + chunk_size + (chunk_size % 2); // Add padding if odd size
            }

            assert!(has_fmt, "WAV file should have FMT chunk");
            assert!(has_data, "WAV file should have DATA chunk");

            // Clean up
            fs::remove_file(&output_path).ok();
        }
    }
}
