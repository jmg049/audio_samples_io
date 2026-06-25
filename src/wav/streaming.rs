//! Streaming WAV file reader for memory-efficient audio processing.
//!
//! This module provides `StreamedWavFile`, a streaming reader that parses WAV headers
//! on construction but reads audio data on-demand, enabling processing of large files
//! without loading them entirely into memory.

use std::{
    io::SeekFrom,
    num::{NonZeroU32, NonZeroUsize},
    path::{Path, PathBuf},
    time::Duration,
};

use audio_samples::{AudioSamples, CastFrom, CastInto, ConvertFrom, ConvertTo, I24, nzu, traits::StandardSample};
use non_empty_slice::NonEmptyVec;

use crate::{
    ReadSeek,
    error::{AudioIOError, AudioIOResult, ErrorPosition},
    traits::{AudioFileMetadata, AudioInfoMarker, AudioStreamRead, AudioStreamReader},
    types::{BaseAudioInfo, FileType, ValidatedSampleType},
    wav::{
        FormatCode,
        chunks::{
            BW64_CHUNK, ChunkDesc, ChunkID, DATA_CHUNK, DS64_CHUNK, FMT_CHUNK, RF64_CHUNK, RIFF_CHUNK, WAVE_CHUNK,
        },
        ds64::Ds64,
        fmt::FmtChunk,
        wav_file::WavFileInfo,
    },
};

/// A streaming WAV file reader that reads audio data on-demand.
///
/// Unlike `WavFile` which loads or memory-maps the entire file, `StreamedWavFile`
/// only parses headers at construction and reads audio frames incrementally.
/// This is ideal for:
/// - Processing files larger than available memory
/// - Real-time streaming applications
/// - Network sources implementing `Read + Seek`
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::wav::StreamedWavFile;
/// use audio_samples_io::traits::AudioFileMetadata;
/// use audio_samples::{AudioSamples, nzu};
/// use std::fs::File;
/// use std::io::BufReader;
/// use std::num::NonZeroU32;
///
/// let file = BufReader::new(File::open("audio.wav")?);
/// let mut streamed = StreamedWavFile::new(file)?;
///
/// // Read 1024 frames at a time
/// let channels = NonZeroU32::new(streamed.num_channels() as u32).ok_or_else(|| audio_samples_io::error::AudioIOError::UnsupportedFormat("channels must be non-zero".to_string()))?;
/// let sample_rate = NonZeroU32::new(streamed.sample_rate()).ok_or_else(|| audio_samples_io::error::AudioIOError::UnsupportedFormat("sample_rate must be non-zero".to_string()))?;
/// let mut buffer = AudioSamples::<f32>::zeros_multi(channels, nzu!(1024), sample_rate);
/// while streamed.remaining_frames() > 0 {
///     let frames_read = streamed.read_frames_into(&mut buffer, nzu!(1024))?;
///     // Process buffer...
/// }
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[derive(Debug)]
pub struct StreamedWavFile<R>
where
    R: ReadSeek,
{
    /// The underlying reader
    reader: R,
    /// File path (if opened from path, otherwise synthetic)
    file_path: PathBuf,
    /// Discovered chunks (for metadata)
    chunks: Vec<ChunkDesc>,
    /// Cached format code
    format_code: FormatCode,
    /// Sample rate in Hz
    pub sample_rate: NonZeroU32,
    /// Number of channels
    pub channels: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Bytes per sample
    pub bytes_per_sample: u16,
    /// Byte rate
    pub byte_rate: u32,
    /// Block align (bytes per frame)
    pub block_align: u16,
    /// Validated sample type
    pub sample_type: ValidatedSampleType,
    /// Total number of samples (all channels)
    pub total_samples: usize,
    /// Total number of frames
    pub total_frames: usize,
    /// Byte offset where audio data starts (absolute file position)
    data_offset: u64,
    /// Current frame position (0-indexed)
    current_frame: usize,
    /// Reusable byte buffer for reading
    byte_buffer: Vec<u8>,
}

impl<R> StreamedWavFile<R>
where
    R: ReadSeek,
{
    /// Create a new streaming WAV reader from any `Read + Seek` source.
    ///
    /// Parses the WAV header to extract format information and locates the
    /// data chunk, but does not read any audio samples.
    ///
    /// # Arguments
    ///
    /// * `reader` - Any type implementing `Read + Seek` (e.g., `BufReader<File>`)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source is not a valid WAV file
    /// - Required chunks (fmt, data) are missing
    /// - The format is unsupported
    pub fn new(reader: R) -> AudioIOResult<Self> {
        Self::new_with_path(reader, PathBuf::from("<stream>"))
    }

    /// Create a new streaming WAV reader with an associated path.
    ///
    /// The path is used for error messages and metadata; the reader
    /// is the actual data source.
    pub fn new_with_path(mut reader: R, file_path: PathBuf) -> AudioIOResult<Self> {
        // Read enough bytes for RIFF header + reasonable chunk scanning
        // We'll read in chunks to handle headers of various sizes
        let mut header_buf = vec![0u8; 4096];
        let bytes_read = reader.read(&mut header_buf)?;
        header_buf.truncate(bytes_read);

        if header_buf.len() < 12 {
            return Err(AudioIOError::corrupted_data(
                "File too small to be a valid WAV file",
                format!("Read {} bytes", header_buf.len()),
                ErrorPosition::new(0).with_description("start of file"),
            ));
        }

        // Parse RIFF header
        let riff_bytes: [u8; 4] = header_buf.get(0..4).and_then(|s| s.try_into().ok()).ok_or_else(|| {
            AudioIOError::corrupted_data(
                "Cannot read RIFF header",
                format!("Read {} bytes", header_buf.len()),
                ErrorPosition::new(0).with_description("RIFF header at file start"),
            )
        })?;
        let riff = ChunkID::new(&riff_bytes);
        let is_rf64 = riff == RF64_CHUNK || riff == BW64_CHUNK;

        if riff != RIFF_CHUNK && !is_rf64 {
            return Err(AudioIOError::corrupted_data(
                "Data does not start with RIFF header",
                format!("Found: {riff:?}"),
                ErrorPosition::new(0).with_description("RIFF/RF64/BW64 header at file start"),
            ));
        }

        let riff_size_bytes: [u8; 4] = header_buf.get(4..8).and_then(|s| s.try_into().ok()).ok_or_else(|| {
            AudioIOError::corrupted_data(
                "Cannot read RIFF chunk size",
                format!("Read {} bytes", header_buf.len()),
                ErrorPosition::new(4).with_description("RIFF chunk size"),
            )
        })?;
        let riff_size = u32::from_le_bytes(riff_size_bytes);

        let wave_bytes: [u8; 4] = header_buf.get(8..12).and_then(|s| s.try_into().ok()).ok_or_else(|| {
            AudioIOError::corrupted_data(
                "Cannot read WAVE identifier",
                format!("Read {} bytes", header_buf.len()),
                ErrorPosition::new(8).with_description("WAVE identifier"),
            )
        })?;
        let wave = ChunkID::new(&wave_bytes);

        if wave != WAVE_CHUNK {
            return Err(AudioIOError::corrupted_data(
                "Data does not contain WAVE identifier",
                format!("Found: {wave:?}"),
                ErrorPosition::new(8).with_description("WAVE identifier"),
            ));
        }

        // Scan for chunks
        let mut chunks: Vec<ChunkDesc> = Vec::new();
        chunks.push(ChunkDesc {
            id: riff,
            offset: 0,
            logical_size: riff_size as usize,
            // Saturating: a streaming/unfinalised WAV stores 0xFFFFFFFF here,
            // which overflows a 32-bit usize (wasm) on `+ 8`.
            total_size: (riff_size as usize).saturating_add(8),
        });
        chunks.push(ChunkDesc {
            id: wave,
            offset: 8,
            logical_size: 4,
            total_size: 4,
        });

        let mut fmt_chunk_data: Option<Vec<u8>> = None;
        let mut data_chunk_desc: Option<ChunkDesc> = None;
        // RF64/BW64: true 64-bit sizes from the mandatory ds64 chunk.
        let mut ds64: Option<Ds64> = None;
        let mut offset = 12usize;

        // Parse chunks from buffer, seeking for more data if needed
        loop {
            // Ensure we have enough data for chunk header
            while offset + 8 > header_buf.len() {
                let current_len = header_buf.len();
                header_buf.resize(current_len + 4096, 0);
                let additional = reader.read(&mut header_buf[current_len..])?;
                if additional == 0 {
                    // EOF reached
                    header_buf.truncate(current_len);
                    break;
                }
                header_buf.truncate(current_len + additional);
            }

            if offset + 8 > header_buf.len() {
                break; // No more chunks
            }

            let id = ChunkID::new(header_buf[offset..offset + 4].try_into().map_err(|_| {
                AudioIOError::corrupted_data("Cannot read chunk ID", "Insufficient data", ErrorPosition::new(offset))
            })?);

            let declared_size_32 = u32::from_le_bytes(header_buf[offset + 4..offset + 8].try_into().map_err(|_| {
                AudioIOError::corrupted_data(
                    "Cannot read chunk size",
                    "Insufficient data",
                    ErrorPosition::new(offset + 4),
                )
            })?);

            // In an RF64/BW64 file, a chunk declaring 0xFFFFFFFF stores its true
            // 64-bit size in the ds64 chunk (the data chunk always; others via table).
            let size = match &ds64 {
                Some(ds) => usize::try_from(ds.resolve(id, declared_size_32)).unwrap_or(usize::MAX),
                None => declared_size_32 as usize,
            };

            // Saturating: a 0xFFFFFFFF placeholder size overflows 32-bit usize.
            let padded = size.saturating_add(size & 1);
            let header_and_data_size = padded.saturating_add(8);

            chunks.push(ChunkDesc {
                id,
                offset,
                logical_size: size,
                total_size: header_and_data_size,
            });

            // Handle ds64 chunk (RF64/BW64) - need its content before any chunk
            // that declares the 0xFFFFFFFF size placeholder can be resolved
            if is_rf64 && id == DS64_CHUNK {
                let body_start = offset + 8;
                let body_end = body_start + size;

                while body_end > header_buf.len() {
                    let current_len = header_buf.len();
                    header_buf.resize(current_len + 4096, 0);
                    let additional = reader.read(&mut header_buf[current_len..])?;
                    if additional == 0 {
                        return Err(AudioIOError::corrupted_data(
                            "Unexpected EOF reading ds64 chunk",
                            format!("Need {} bytes, have {}", body_end, header_buf.len()),
                            ErrorPosition::new(body_start),
                        ));
                    }
                    header_buf.truncate(current_len + additional);
                }

                ds64 = Some(Ds64::from_bytes(&header_buf[body_start..body_end])?);
            }

            // Handle fmt chunk - need to read its content
            if id == FMT_CHUNK {
                let fmt_start = offset + 8;
                let fmt_end = fmt_start + size;

                // Ensure we have the fmt data in buffer
                while fmt_end > header_buf.len() {
                    let current_len = header_buf.len();
                    header_buf.resize(current_len + 4096, 0);
                    let additional = reader.read(&mut header_buf[current_len..])?;
                    if additional == 0 {
                        return Err(AudioIOError::corrupted_data(
                            "Unexpected EOF reading fmt chunk",
                            format!("Need {} bytes, have {}", fmt_end, header_buf.len()),
                            ErrorPosition::new(fmt_start),
                        ));
                    }
                    header_buf.truncate(current_len + additional);
                }

                fmt_chunk_data = Some(header_buf[fmt_start..fmt_end].to_vec());
            }

            // Handle data chunk - just record its location
            if id == DATA_CHUNK {
                data_chunk_desc = Some(ChunkDesc {
                    id,
                    offset,
                    logical_size: size,
                    total_size: header_and_data_size,
                });
                // Don't read data chunk content - that's the point of streaming!
                break;
            }

            offset = offset.saturating_add(header_and_data_size);
        }

        // Validate we found required chunks
        let fmt_bytes = fmt_chunk_data.ok_or_else(|| {
            AudioIOError::corrupted_data(
                "FMT chunk not found in WAV file",
                format!("Found chunks: {:?}", chunks.iter().map(|c| c.id).collect::<Vec<_>>()),
                ErrorPosition::new(12),
            )
        })?;

        let data_desc = data_chunk_desc.ok_or_else(|| {
            AudioIOError::corrupted_data(
                "DATA chunk not found in WAV file",
                format!("Found chunks: {:?}", chunks.iter().map(|c| c.id).collect::<Vec<_>>()),
                ErrorPosition::new(12),
            )
        })?;

        // Parse fmt chunk
        let fmt_chunk = FmtChunk::from_bytes_validated(&fmt_bytes).map_err(AudioIOError::WavError)?;
        let sample_type = fmt_chunk.actual_sample_type()?;

        let (format_code, channels, sample_rate, byte_rate, block_align, bits_per_sample) = fmt_chunk.fmt_chunk();
        let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
            AudioIOError::corrupted_data_simple("Invalid sample rate in FMT chunk", "Sample rate cannot be zero")
        })?;
        let bytes_per_sample = bits_per_sample / 8;

        // Calculate frame info
        let data_offset = (data_desc.offset as u64).saturating_add(8);
        // A streaming/unfinalised WAV stores 0xFFFFFFFF as the data-chunk size
        // placeholder; the real size is whatever bytes remain in the stream.
        // Also clamp a declared size that overruns the available stream.
        let stream_len = reader.seek(SeekFrom::End(0))?;
        let avail = stream_len.saturating_sub(data_offset);
        let declared = data_desc.logical_size as u64;
        let data_size = if declared == u32::MAX as u64 || declared > avail {
            avail
        } else {
            declared
        };
        let total_samples = data_size as usize / sample_type.bytes_per_sample();
        let total_frames = total_samples / channels as usize;

        // Seek to start of audio data
        reader.seek(SeekFrom::Start(data_offset))?;

        Ok(StreamedWavFile {
            reader,
            file_path,
            chunks,
            format_code,
            sample_rate,
            channels,
            bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            sample_type,
            total_samples,
            total_frames,
            data_offset,
            current_frame: 0,
            byte_buffer: Vec::new(),
        })
    }

    /// Get the current frame position.
    #[inline]
    pub const fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Get the number of remaining frames from current position.
    #[inline]
    pub const fn remaining_frames(&self) -> usize {
        self.total_frames.saturating_sub(self.current_frame)
    }

    /// Get the total number of frames in the file.
    #[inline]
    pub const fn total_frames(&self) -> usize {
        self.total_frames
    }

    /// Get the sample rate.
    #[inline]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate.get()
    }

    /// Get the bytes per frame (block_align).
    #[inline]
    pub const fn bytes_per_frame(&self) -> usize {
        self.block_align as usize
    }

    /// Get the number of channels.
    #[inline]
    pub const fn num_channels(&self) -> u16 {
        self.channels
    }

    /// Seek to a specific frame position.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame index to seek to (0-indexed)
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is beyond the end of the file or seek fails.
    pub fn seek_to_frame(&mut self, frame: usize) -> AudioIOResult<()> {
        if frame > self.total_frames {
            return Err(AudioIOError::SeekError(format!(
                "Frame {} is beyond end of file (total frames: {})",
                frame, self.total_frames
            )));
        }

        let byte_offset = frame as u64 * self.block_align as u64;
        self.reader.seek(SeekFrom::Start(self.data_offset + byte_offset))?;
        self.current_frame = frame;
        Ok(())
    }

    /// Reset to the beginning of the audio data.
    pub fn reset(&mut self) -> AudioIOResult<()> {
        self.seek_to_frame(0)
    }

    /// Read frames into a pre-allocated `AudioSamples` buffer.
    ///
    /// This is the primary zero-allocation read method. After initial buffer
    /// allocation, repeated calls reuse the same memory.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Pre-allocated `AudioSamples` to fill with frame data
    /// * `frame_count` - Maximum number of frames to read
    ///
    /// # Returns
    ///
    /// The actual number of frames read (may be less at end of file).
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or data is corrupted.
    pub fn read_frames_into<T>(
        &mut self,
        buffer: &mut AudioSamples<'_, T>,
        frame_count: NonZeroUsize,
    ) -> AudioIOResult<usize>
    where
        T: StandardSample + 'static,
    {
        let frames_available = self.remaining_frames();
        let frames_to_read = frame_count.get().min(frames_available);

        if frames_to_read == 0 {
            return Ok(0);
        }

        let bytes_to_read = frames_to_read * self.bytes_per_frame();

        // Resize byte buffer if needed (only grows, never shrinks during iteration)
        if self.byte_buffer.len() < bytes_to_read {
            self.byte_buffer.resize(bytes_to_read, 0);
        }

        // Read bytes from source
        let bytes_read = self.reader.read(&mut self.byte_buffer[..bytes_to_read])?;
        let frames_read = bytes_read / self.bytes_per_frame();

        if frames_read == 0 {
            return Ok(0);
        }

        let actual_bytes = frames_read * self.bytes_per_frame();

        let mut converted_buf = Vec::new();
        self.convert_bytes_to_samples::<T>(&self.byte_buffer[..actual_bytes], &mut converted_buf);
        // safety: We have already verified that converted_buf.len() == total samples to read
        let converted = unsafe { NonEmptyVec::new_unchecked(converted_buf) };
        // Deinterleave and replace buffer contents
        // safety: channels is guaranteed > 0 from parsing, and converted length matches frames read
        let num_channels = unsafe { NonZeroU32::new_unchecked(self.channels as u32) };

        if buffer.is_mono() {
            buffer.replace_with_vec(&converted)?;
        } else {
            let planar = audio_samples::simd_conversions::deinterleave_multi_vec(&converted, num_channels)
                .map_err(|e| AudioIOError::corrupted_data_simple("Deinterleave failed", e.to_string()))?;
            buffer.replace_with_vec(&planar)?;
        }

        self.current_frame += frames_read;
        Ok(frames_read)
    }

    /// Convert raw bytes to samples of type T, writing into `out` (cleared first).
    fn convert_bytes_to_samples<T>(&self, bytes: &[u8], out: &mut Vec<T>)
    where
        T: StandardSample + 'static,
    {
        out.clear();
        match self.sample_type {
            ValidatedSampleType::U8 => out.extend(bytes.iter().map(|&b| T::convert_from(b))),
            ValidatedSampleType::I16 => out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|c| T::convert_from(i16::from_le_bytes([c[0], c[1]]))),
            ),
            ValidatedSampleType::I24 => out.extend(
                bytes
                    .chunks_exact(3)
                    .map(|c| T::convert_from(I24::from_le_bytes([c[0], c[1], c[2]]))),
            ),
            ValidatedSampleType::I32 => out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| T::convert_from(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
            ),
            ValidatedSampleType::F32 => out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| T::convert_from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
            ),
            ValidatedSampleType::F64 => out.extend(
                bytes
                    .chunks_exact(8)
                    .map(|c| T::convert_from(f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))),
            ),
        }
    }

    /// Create a frame iterator over this streamed file.
    ///
    /// Returns frames one at a time, reusing an internal buffer.
    ///
    /// # Panics
    ///
    /// Does not panic since the sample rate is guaranteed to be non-zero during parsing.
    pub fn frames<T>(&mut self) -> StreamedFrameIter<'_, R, T>
    where
        T: StandardSample + ConvertTo<T> + ConvertFrom<T> + 'static,
        f64: CastInto<T> + CastFrom<T> + ConvertTo<T> + ConvertFrom<T>,
    {
        let sample_rate = self.sample_rate;
        let frame_buffer = if self.channels == 1 {
            AudioSamples::zeros_mono(audio_samples::nzu!(1), sample_rate)
        } else {
            // safe: channels is guaranteed > 0
            AudioSamples::zeros_multi(
                unsafe { NonZeroU32::new_unchecked(self.channels as u32) },
                audio_samples::nzu!(1),
                sample_rate,
            )
        };
        StreamedFrameIter {
            source: self,
            frame_buffer,
        }
    }

    /// Create a windowed iterator over this streamed file.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of frames per window
    /// * `hop_size` - Number of frames to advance between windows
    ///
    /// # Panics
    ///
    /// Does not panic since the sample rate is guaranteed to be non-zero during parsing.
    pub fn windows<T>(&mut self, window_size: NonZeroUsize, hop_size: NonZeroUsize) -> StreamedWindowIter<'_, R, T>
    where
        T: StandardSample + ConvertTo<T> + ConvertFrom<T> + 'static,
        f64: CastInto<T> + CastFrom<T> + ConvertTo<T> + ConvertFrom<T>,
    {
        let sample_rate = self.sample_rate;
        let window_buffer = if self.channels == 1 {
            AudioSamples::zeros_mono(window_size, sample_rate)
        } else {
            AudioSamples::zeros_multi(
                unsafe { NonZeroU32::new_unchecked(self.channels as u32) },
                window_size,
                sample_rate,
            )
        };
        StreamedWindowIter {
            source: self,
            window_buffer,
            window_size,
            hop_size,
            first_window: true,
        }
    }

    /// Create a sample iterator over this streamed file.
    ///
    /// Yields individual samples in interleaved channel order (L, R, L, R, … for stereo),
    /// converting from the file's native sample type to `T` on the fly. Reads from the
    /// source in chunks to amortise I/O cost.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use audio_samples_io::wav::StreamedWavFile;
    /// use std::{fs::File, io::BufReader};
    ///
    /// let mut reader = StreamedWavFile::new(BufReader::new(File::open("audio.wav")?))?;
    /// let peak: f32 = reader.samples::<f32>()
    ///     .flatten()
    ///     .map(f32::abs)
    ///     .fold(0.0_f32, f32::max);
    /// # Ok::<(), audio_samples_io::error::AudioIOError>(())
    /// ```
    pub fn samples<T>(&mut self) -> StreamedSampleIter<'_, R, T>
    where
        T: StandardSample + Copy + 'static,
    {
        let byte_buf = vec![0u8; 512 * self.bytes_per_frame()];
        StreamedSampleIter {
            source: self,
            byte_buf,
            sample_buf: Vec::new(),
            pos: 0,
        }
    }
}

// Implement AudioFileMetadata for StreamedWavFile
impl<R> AudioFileMetadata for StreamedWavFile<R>
where
    R: ReadSeek,
{
    fn open_metadata<P: AsRef<Path>>(_path: P) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        // This doesn't make sense for StreamedWavFile since we need a reader
        Err(AudioIOError::corrupted_data_simple(
            "StreamedWavFile requires a reader",
            "Use StreamedWavFile::new() instead",
        ))
    }

    fn base_info(&self) -> AudioIOResult<BaseAudioInfo> {
        let duration = Duration::from_secs_f64(self.total_frames as f64 / self.sample_rate.get() as f64);
        Ok(BaseAudioInfo::new(
            self.sample_rate,
            self.channels,
            self.bits_per_sample,
            self.bytes_per_sample,
            self.byte_rate,
            self.block_align,
            self.total_samples,
            duration,
            FileType::WAV,
            self.sample_type.into(),
        ))
    }

    fn specific_info(&self) -> impl AudioInfoMarker {
        WavFileInfo {
            available_chunks: self.chunks.iter().map(|c| c.id).collect(),
            encoding: self.format_code,
            fact_num_samples: None,
            info_metadata: None,
        }
    }

    fn file_type(&self) -> FileType {
        FileType::WAV
    }

    fn file_path(&self) -> &Path {
        &self.file_path
    }

    fn total_samples(&self) -> usize {
        self.total_samples
    }

    fn duration(&self) -> AudioIOResult<Duration> {
        Ok(Duration::from_secs_f64(
            self.total_frames as f64 / self.sample_rate.get() as f64,
        ))
    }

    fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    fn num_channels(&self) -> u16 {
        self.channels
    }
}

// Implement AudioStreamReader for StreamedWavFile (object-safe streaming trait)
impl<R> AudioStreamReader for StreamedWavFile<R>
where
    R: ReadSeek,
{
    #[inline]
    fn current_frame(&self) -> usize {
        self.current_frame
    }

    #[inline]
    fn remaining_frames(&self) -> usize {
        self.total_frames.saturating_sub(self.current_frame)
    }

    #[inline]
    fn total_frames(&self) -> usize {
        self.total_frames
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        self.sample_rate.get()
    }

    #[inline]
    fn bytes_per_frame(&self) -> usize {
        self.block_align as usize
    }

    #[inline]
    fn num_channels(&self) -> u16 {
        self.channels
    }

    fn seek_to_frame(&mut self, frame: usize) -> AudioIOResult<()> {
        StreamedWavFile::seek_to_frame(self, frame)
    }

    fn reset(&mut self) -> AudioIOResult<()> {
        StreamedWavFile::reset(self)
    }
}

// Implement AudioStreamRead for StreamedWavFile (generic streaming read trait)
impl<R> AudioStreamRead for StreamedWavFile<R>
where
    R: ReadSeek,
{
    fn read_frames_into<T>(
        &mut self,
        buffer: &mut AudioSamples<'_, T>,
        frame_count: NonZeroUsize,
    ) -> AudioIOResult<usize>
    where
        T: StandardSample + 'static,
    {
        StreamedWavFile::read_frames_into(self, buffer, frame_count)
    }
}

/// Iterator over individual frames from a streamed WAV file.
///
/// Each call to `next()` reads one frame from the source and returns
/// a reference to the internal buffer containing that frame's samples.
pub struct StreamedFrameIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + 'static,
{
    source: &'a mut StreamedWavFile<R>,
    frame_buffer: AudioSamples<'static, T>,
}

impl<'a, R, T> Iterator for StreamedFrameIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + 'static,
{
    type Item = AudioIOResult<AudioSamples<'static, T>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.source.remaining_frames() == 0 {
            return None;
        }

        match self.source.read_frames_into(&mut self.frame_buffer, nzu!(1)) {
            Ok(0) => None,
            Ok(_) => Some(Ok(self.frame_buffer.clone())),
            Err(e) => Some(Err(e)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.source.remaining_frames();
        (remaining, Some(remaining))
    }
}

impl<'a, R, T> ExactSizeIterator for StreamedFrameIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + 'static,
{
}

/// Iterator over windows of frames from a streamed WAV file.
///
/// Supports overlapping windows via configurable hop size.
pub struct StreamedWindowIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + 'static,
{
    source: &'a mut StreamedWavFile<R>,
    window_buffer: AudioSamples<'static, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
    first_window: bool,
}

impl<'a, R, T> Iterator for StreamedWindowIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + 'static,
{
    type Item = AudioIOResult<AudioSamples<'static, T>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.source.remaining_frames() == 0 {
            return None;
        }

        // For overlapping windows after the first, we need to seek back
        if !self.first_window && self.hop_size < self.window_size {
            let overlap = self.window_size.get() - self.hop_size.get();
            let new_frame = self.source.current_frame.saturating_sub(overlap);
            if let Err(e) = self.source.seek_to_frame(new_frame) {
                return Some(Err(e));
            }
        }
        self.first_window = false;

        match self.source.read_frames_into(&mut self.window_buffer, self.window_size) {
            Ok(0) => None,
            Ok(_) => Some(Ok(self.window_buffer.clone())),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Iterator over individual samples from a streamed WAV file.
///
/// Samples are yielded in interleaved channel order. Internally reads from the
/// source in chunks to amortise I/O overhead. Use [`StreamedWavFile::samples`]
/// to construct this iterator.
pub struct StreamedSampleIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + Copy + 'static,
{
    source: &'a mut StreamedWavFile<R>,
    byte_buf: Vec<u8>,
    sample_buf: Vec<T>,
    pos: usize,
}

impl<'a, R, T> Iterator for StreamedSampleIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + Copy + 'static,
{
    type Item = AudioIOResult<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.sample_buf.len() {
            let s = self.sample_buf[self.pos];
            self.pos += 1;
            return Some(Ok(s));
        }

        if self.source.remaining_frames() == 0 {
            return None;
        }

        let bpf = self.source.bytes_per_frame();
        let frames_to_read = (self.byte_buf.len() / bpf).min(self.source.remaining_frames());
        let bytes_to_read = frames_to_read * bpf;

        let bytes_read = match self.source.reader.read(&mut self.byte_buf[..bytes_to_read]) {
            Err(e) => return Some(Err(AudioIOError::from(e))),
            Ok(0) => return None,
            Ok(n) => n,
        };

        let actual_bytes = (bytes_read / bpf) * bpf;
        let frames_read = actual_bytes / bpf;

        self.source
            .convert_bytes_to_samples::<T>(&self.byte_buf[..actual_bytes], &mut self.sample_buf);
        self.source.current_frame += frames_read;
        self.pos = 1;
        Some(Ok(self.sample_buf[0]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let buffered = self.sample_buf.len() - self.pos;
        let from_file = self.source.remaining_frames() * self.source.channels as usize;
        (buffered + from_file, Some(buffered + from_file))
    }
}

impl<'a, R, T> ExactSizeIterator for StreamedSampleIter<'a, R, T>
where
    R: ReadSeek,
    T: StandardSample + Copy + 'static,
{
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use audio_samples::{nzu, sample_rate};

    use super::*;

    #[test]
    fn test_streamed_wav_file_open() {
        let file = BufReader::new(File::open("resources/test.wav").expect("Test file not found"));
        let streamed = StreamedWavFile::new(file);
        assert!(streamed.is_ok(), "Failed to open streamed WAV file");

        let streamed = streamed.expect("Failed to open streamed WAV file");
        assert!(streamed.total_frames() > 0);
        assert_eq!(streamed.current_frame(), 0);
    }

    #[test]
    fn test_streamed_wav_metadata() {
        let file = BufReader::new(File::open("resources/test.wav").expect("Test file not found"));
        let streamed = StreamedWavFile::new(file).expect("Failed to open");

        let base_info = streamed.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(44100));
        assert_eq!(base_info.channels, 2);
    }

    #[test]
    fn test_streamed_read_frames() {
        let file = BufReader::new(File::open("resources/test.wav").expect("Test file not found"));
        let mut streamed = StreamedWavFile::new(file).expect("Failed to open");

        let channels = streamed.num_channels() as usize;
        let sample_rate = NonZeroU32::new(streamed.sample_rate()).expect("sample rate is non-zero");

        let mut buffer = if channels == 1 {
            AudioSamples::<f32>::zeros_mono(nzu!(1024), sample_rate)
        } else {
            // safety: channels is guaranteed > 0
            let channels = unsafe { NonZeroU32::new_unchecked(channels as u32) };
            AudioSamples::<f32>::zeros_multi(channels, nzu!(1024), sample_rate)
        };
        let frames_read = streamed.read_frames_into(&mut buffer, nzu!(1024)).expect("Read failed");

        assert!(frames_read > 0);
        assert_eq!(streamed.current_frame(), frames_read);
    }

    #[test]
    fn test_streamed_seek() {
        let file = BufReader::new(File::open("resources/test.wav").expect("Test file not found"));
        let mut streamed = StreamedWavFile::new(file).expect("Failed to open");

        let mid_frame = streamed.total_frames() / 2;
        streamed.seek_to_frame(mid_frame).expect("Seek failed");
        assert_eq!(streamed.current_frame(), mid_frame);

        streamed.reset().expect("Reset failed");
        assert_eq!(streamed.current_frame(), 0);
    }

    #[test]
    fn test_streamed_frame_iterator() {
        let file = BufReader::new(File::open("resources/test.wav").expect("Test file not found"));
        let mut streamed = StreamedWavFile::new(file).expect("Failed to open");

        let total = streamed.total_frames();
        let mut count = 0;

        for frame_result in streamed.frames::<f32>() {
            let _frame = frame_result.expect("Frame read failed");
            count += 1;
            if count >= 100 {
                break; // Don't iterate entire file in test
            }
        }

        assert_eq!(count, 100.min(total));
    }
}
