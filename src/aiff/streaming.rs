//! Streaming AIFF/AIFF-C file reader for memory-efficient audio processing.
//!
//! `StreamedAiffFile` parses the `COMM` chunk (and locates `SSND`) at construction
//! but reads sound data on demand, mirroring [`crate::wav::StreamedWavFile`]. See
//! [`crate::aiff::aiff_file`] for the format details (chunk layout, compression
//! codes, the AIFF 8-bit signed convention).

use std::{
    io::SeekFrom,
    num::{NonZeroU32, NonZeroUsize},
    path::{Path, PathBuf},
    time::Duration,
};

use audio_samples::{AudioSamples, traits::StandardSample};
use non_empty_slice::NonEmptyVec;

use crate::{
    ReadSeek,
    aiff::aiff_file::{AiffFileInfo, SoundEncoding, decode_sample_be, resolve_encoding},
    aiff::extended::decode_extended,
    error::{AudioIOError, AudioIOResult, ErrorPosition},
    traits::{AudioFileMetadata, AudioInfoMarker, AudioStreamRead, AudioStreamReader},
    types::{BaseAudioInfo, FileType, ValidatedSampleType},
};

fn be_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(bytes[at..at + 4].try_into().expect("4-byte slice"))
}

fn be_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_be_bytes(bytes[at..at + 2].try_into().expect("2-byte slice"))
}

/// A streaming AIFF/AIFF-C reader that reads sound data on-demand.
///
/// Unlike [`AiffFile`](crate::aiff::AiffFile), which loads or memory-maps the entire
/// file, `StreamedAiffFile` only parses the `COMM`/`SSND` headers at construction and
/// reads frames incrementally — suitable for files larger than available memory,
/// real-time playback, or any `Read` source, `Seek`able or not (see
/// [`new_forward_only`](Self::new_forward_only)).
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::aiff::StreamedAiffFile;
/// use audio_samples_io::traits::AudioStreamReader;
/// use audio_samples::{AudioSamples, nzu};
/// use std::fs::File;
/// use std::io::BufReader;
/// use std::num::NonZeroU32;
///
/// let file = BufReader::new(File::open("audio.aiff")?);
/// let mut streamed = StreamedAiffFile::new(file)?;
///
/// let channels = NonZeroU32::new(streamed.num_channels() as u32).unwrap();
/// let sample_rate = NonZeroU32::new(streamed.sample_rate()).unwrap();
/// let mut buffer = AudioSamples::<f32>::zeros_multi(channels, nzu!(1024), sample_rate);
/// while streamed.remaining_frames() > 0 {
///     let frames_read = streamed.read_frames_into(&mut buffer, nzu!(1024))?;
///     // Process buffer...
/// }
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[derive(Debug)]
pub struct StreamedAiffFile<R> {
    reader: R,
    file_path: PathBuf,
    form_type: [u8; 4],
    compression: [u8; 4],
    available_chunks: Vec<[u8; 4]>,
    channels: u16,
    bits_per_sample: u16,
    sample_rate: NonZeroU32,
    sample_type: ValidatedSampleType,
    encoding: SoundEncoding,
    total_frames: usize,
    current_frame: usize,
    /// Absolute byte offset where the `SSND` sample data begins. Only
    /// meaningful when `R: Seek`; [`new_forward_only`](Self::new_forward_only)
    /// never seeks and leaves it `0`.
    sound_data_offset: u64,
    byte_buffer: Vec<u8>,
    /// Sound-data bytes already read into the header-scan buffer past
    /// `SSND`'s 8-byte offset/blockSize fields, while probing for more chunk
    /// headers. [`new`](Self::new) discards these (it seeks back to the exact
    /// data start instead); [`new_forward_only`](Self::new_forward_only) can't
    /// seek, so it keeps them here and [`read_frames_into`](Self::read_frames_into)
    /// drains this before pulling anything more from `reader`.
    leftover: Vec<u8>,
}

/// Header fields shared by both [`StreamedAiffFile::new`] (`Read + Seek`) and
/// [`StreamedAiffFile::new_forward_only`] (`Read` only).
struct AiffHeaderParse {
    form_type: [u8; 4],
    compression: [u8; 4],
    available_chunks: Vec<[u8; 4]>,
    channels: u16,
    num_frames: usize,
    bits_per_sample: u16,
    sample_rate: NonZeroU32,
    sample_type: ValidatedSampleType,
    encoding: SoundEncoding,
    /// Absolute byte offset (from stream start) where SSND's sample data begins.
    sound_data_offset: u64,
    /// Declared sample-data size from SSND's chunk size minus its
    /// offset/blockSize fields and internal data offset, if computable.
    declared_sound_size: Option<usize>,
    /// Sound-data bytes already buffered past `sound_data_offset` while
    /// scanning for SSND — see the field doc on [`StreamedAiffFile::leftover`].
    overshoot: Vec<u8>,
}

/// Parse the FORM/COMM header and locate (but do not read) SSND's sample data.
/// Shared by both the `Read + Seek` and `Read`-only construction paths.
fn parse_aiff_header<R: std::io::Read>(reader: &mut R) -> AudioIOResult<AiffHeaderParse> {
    let mut header_buf = vec![0u8; 4096];
    let bytes_read = reader.read(&mut header_buf)?;
    header_buf.truncate(bytes_read);

    if header_buf.len() < 12 {
        return Err(AudioIOError::corrupted_data(
            "File too small to be a valid AIFF file",
            format!("Read {} bytes", header_buf.len()),
            ErrorPosition::new(0).with_description("start of file"),
        ));
    }
    if &header_buf[0..4] != b"FORM" {
        return Err(AudioIOError::corrupted_data(
            "Data does not start with FORM header",
            format!("Found: {:?}", &header_buf[0..4]),
            ErrorPosition::new(0).with_description("FORM header at file start"),
        ));
    }
    let form_type: [u8; 4] = header_buf[8..12].try_into().expect("4-byte slice");
    let is_aifc = &form_type == b"AIFC";
    if &form_type != b"AIFF" && !is_aifc {
        return Err(AudioIOError::corrupted_data(
            "FORM type is not AIFF or AIFC",
            format!("Found: {form_type:?}"),
            ErrorPosition::new(8).with_description("FORM type"),
        ));
    }

    let mut available_chunks = Vec::new();
    let mut comm: Option<(u16, usize, u16, f64, [u8; 4])> = None;
    let mut sound_data_offset: Option<u64> = None;
    let mut declared_sound_size: Option<usize> = None;

    let mut offset = 12usize;
    loop {
        while offset + 8 > header_buf.len() {
            let current_len = header_buf.len();
            header_buf.resize(current_len + 4096, 0);
            let additional = reader.read(&mut header_buf[current_len..])?;
            if additional == 0 {
                header_buf.truncate(current_len);
                break;
            }
            header_buf.truncate(current_len + additional);
        }

        if offset + 8 > header_buf.len() {
            break; // no more chunks
        }

        let id: [u8; 4] = header_buf[offset..offset + 4].try_into().expect("4-byte slice");
        let declared = be_u32(&header_buf, offset + 4) as usize;
        let body_start = offset + 8;
        available_chunks.push(id);

        if &id == b"COMM" {
            let comm_end = body_start + declared.max(18);
            while comm_end > header_buf.len() {
                let current_len = header_buf.len();
                header_buf.resize(current_len + 4096, 0);
                let additional = reader.read(&mut header_buf[current_len..])?;
                if additional == 0 {
                    return Err(AudioIOError::corrupted_data(
                        "Unexpected EOF reading COMM chunk",
                        format!("Need {} bytes, have {}", comm_end, header_buf.len()),
                        ErrorPosition::new(body_start),
                    ));
                }
                header_buf.truncate(current_len + additional);
            }

            if declared < 18 {
                return Err(AudioIOError::corrupted_data(
                    "COMM chunk too small",
                    format!("Expected at least 18 bytes, found {declared}"),
                    ErrorPosition::new(body_start).with_description("COMM chunk body"),
                ));
            }
            let channels = be_u16(&header_buf, body_start);
            let num_frames = be_u32(&header_buf, body_start + 2) as usize;
            let bits = be_u16(&header_buf, body_start + 6);
            let rate_bytes: [u8; 10] = header_buf[body_start + 8..body_start + 18]
                .try_into()
                .expect("10-byte slice");
            let rate = decode_extended(&rate_bytes);
            let compression: [u8; 4] = if is_aifc && declared >= 22 {
                header_buf[body_start + 18..body_start + 22]
                    .try_into()
                    .expect("4-byte slice")
            } else {
                *b"NONE"
            };
            comm = Some((channels, num_frames, bits, rate, compression));
        } else if &id == b"SSND" {
            let ssnd_header_end = body_start + 8;
            while ssnd_header_end > header_buf.len() {
                let current_len = header_buf.len();
                header_buf.resize(current_len + 4096, 0);
                let additional = reader.read(&mut header_buf[current_len..])?;
                if additional == 0 {
                    return Err(AudioIOError::corrupted_data(
                        "Unexpected EOF reading SSND chunk",
                        format!("Need {ssnd_header_end} bytes, have {}", header_buf.len()),
                        ErrorPosition::new(body_start),
                    ));
                }
                header_buf.truncate(current_len + additional);
            }
            if declared < 8 {
                return Err(AudioIOError::corrupted_data(
                    "SSND chunk too small",
                    format!("Expected at least 8 bytes, found {declared}"),
                    ErrorPosition::new(body_start).with_description("SSND chunk body"),
                ));
            }
            // SSND body: offset (u32) + blockSize (u32) + sound data.
            let data_offset = be_u32(&header_buf, body_start) as usize;
            sound_data_offset = Some((body_start + 8 + data_offset) as u64);
            declared_sound_size = Some(declared.saturating_sub(8).saturating_sub(data_offset));
            // Don't read the sound data itself — that's the point of streaming.
            break;
        }

        let padded = declared + (declared & 1); // IFF chunks are word-aligned
        offset = body_start + padded;
    }

    let (channels, num_frames, bits_per_sample, rate, compression) = comm.ok_or_else(|| {
        AudioIOError::corrupted_data(
            "COMM chunk not found in AIFF file",
            format!("Found chunks: {available_chunks:?}"),
            ErrorPosition::new(12).with_description("chunk data section"),
        )
    })?;
    let sound_data_offset = sound_data_offset.ok_or_else(|| {
        AudioIOError::corrupted_data(
            "SSND chunk not found in AIFF file",
            format!("Found chunks: {available_chunks:?}"),
            ErrorPosition::new(12).with_description("chunk data section"),
        )
    })?;

    if channels == 0 {
        return Err(AudioIOError::corrupted_data_simple(
            "Invalid channel count",
            "COMM declares zero channels",
        ));
    }
    let sample_rate = NonZeroU32::new(rate.round() as u32).ok_or_else(|| {
        AudioIOError::corrupted_data_simple("Invalid sample rate", format!("COMM declares {rate} Hz"))
    })?;

    let (encoding, sample_type) = resolve_encoding(&compression, bits_per_sample)?;

    let overshoot = header_buf
        .get(sound_data_offset as usize..)
        .map(<[u8]>::to_vec)
        .unwrap_or_default();

    Ok(AiffHeaderParse {
        form_type,
        compression,
        available_chunks,
        channels,
        num_frames,
        bits_per_sample,
        sample_rate,
        sample_type,
        encoding,
        sound_data_offset,
        declared_sound_size,
        overshoot,
    })
}

impl<R> StreamedAiffFile<R>
where
    R: ReadSeek,
{
    /// Create a new streaming AIFF reader from any `Read + Seek` source.
    ///
    /// Parses `COMM` and locates `SSND`, but does not read any sound data. See
    /// [`new_forward_only`](Self::new_forward_only) for sources that can't seek.
    ///
    /// # Errors
    ///
    /// Returns an error if the source is not a valid AIFF/AIFF-C stream, the
    /// `COMM`/`SSND` chunks are missing, or the compression code is unsupported.
    pub fn new(reader: R) -> AudioIOResult<Self> {
        Self::new_with_path(reader, PathBuf::from("<stream>"))
    }

    /// Create a new streaming AIFF reader with an associated path.
    ///
    /// The path is used for error messages and metadata; the reader is the actual
    /// data source.
    pub fn new_with_path(mut reader: R, file_path: PathBuf) -> AudioIOResult<Self> {
        let parsed = parse_aiff_header(&mut reader)?;

        // Clamp the frame count to the bytes actually present, tolerating a
        // declared SSND size that overruns the stream (e.g. an unfinalised or
        // truncated file), mirroring `StreamedWavFile`.
        let bytes_per_sample = parsed.sample_type.bytes_per_sample().get();
        let frame_bytes = parsed.channels as usize * bytes_per_sample;
        let stream_len = reader.seek(SeekFrom::End(0))?;
        let avail = stream_len.saturating_sub(parsed.sound_data_offset) as usize;
        let declared_size = parsed.declared_sound_size.unwrap_or(avail);
        let sound_size = declared_size.min(avail);
        let frames_in_sound = sound_size / frame_bytes.max(1);
        let total_frames = parsed.num_frames.min(frames_in_sound);

        // Seek to start of sound data — the header scan above may have
        // buffered (and discarded, via `parsed.overshoot`) bytes past it.
        reader.seek(SeekFrom::Start(parsed.sound_data_offset))?;

        Ok(StreamedAiffFile {
            reader,
            file_path,
            form_type: parsed.form_type,
            compression: parsed.compression,
            available_chunks: parsed.available_chunks,
            channels: parsed.channels,
            bits_per_sample: parsed.bits_per_sample,
            sample_rate: parsed.sample_rate,
            sample_type: parsed.sample_type,
            encoding: parsed.encoding,
            total_frames,
            current_frame: 0,
            sound_data_offset: parsed.sound_data_offset,
            byte_buffer: Vec::new(),
            leftover: Vec::new(),
        })
    }

    /// Seek to a specific frame position (0-indexed).
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is beyond the end of the stream or seek fails.
    pub fn seek_to_frame(&mut self, frame: usize) -> AudioIOResult<()> {
        if frame > self.total_frames {
            return Err(AudioIOError::SeekError(format!(
                "Frame {} is beyond end of file (total frames: {})",
                frame, self.total_frames
            )));
        }

        let byte_offset = frame as u64 * self.bytes_per_frame() as u64;
        self.reader
            .seek(SeekFrom::Start(self.sound_data_offset + byte_offset))?;
        self.current_frame = frame;
        Ok(())
    }

    /// Reset to the beginning of the sound data.
    pub fn reset(&mut self) -> AudioIOResult<()> {
        self.seek_to_frame(0)
    }
}

impl<R: std::io::Read> StreamedAiffFile<R> {
    /// Create a streaming AIFF reader from any `Read` source that need not
    /// support `Seek` (a network transport, for instance).
    ///
    /// Unlike [`new`](Self::new), this can't seek to the end of the stream to
    /// clamp an oversized or truncated declared `SSND` size — the declared
    /// size is trusted as-is, relying on the underlying `Read` reaching real
    /// EOF to end the stream if it's wrong.
    /// [`seek_to_frame`](StreamedAiffFile::seek_to_frame)/[`reset`](StreamedAiffFile::reset)
    /// (and so the [`AudioStreamReader`]/[`AudioStreamRead`] trait impls) are
    /// unavailable — they require `R: Seek`.
    pub fn new_forward_only(reader: R) -> AudioIOResult<Self> {
        Self::new_forward_only_with_path(reader, PathBuf::from("<stream>"))
    }

    /// [`new_forward_only`](Self::new_forward_only) with an associated path
    /// for error messages and metadata.
    pub fn new_forward_only_with_path(mut reader: R, file_path: PathBuf) -> AudioIOResult<Self> {
        let parsed = parse_aiff_header(&mut reader)?;

        let bytes_per_sample = parsed.sample_type.bytes_per_sample().get();
        let frame_bytes = (parsed.channels as usize * bytes_per_sample).max(1);
        let total_frames = match parsed.declared_sound_size {
            Some(declared) => parsed.num_frames.min(declared / frame_bytes),
            None => parsed.num_frames,
        };

        Ok(StreamedAiffFile {
            reader,
            file_path,
            form_type: parsed.form_type,
            compression: parsed.compression,
            available_chunks: parsed.available_chunks,
            channels: parsed.channels,
            bits_per_sample: parsed.bits_per_sample,
            sample_rate: parsed.sample_rate,
            sample_type: parsed.sample_type,
            encoding: parsed.encoding,
            total_frames,
            current_frame: 0,
            sound_data_offset: 0,
            byte_buffer: Vec::new(),
            leftover: parsed.overshoot,
        })
    }

    /// Get the current frame position.
    #[inline]
    pub const fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Get the number of remaining frames from the current position.
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

    /// Get the bytes per frame (block align).
    #[inline]
    pub fn bytes_per_frame(&self) -> usize {
        self.channels as usize * self.sample_type.bytes_per_sample().get()
    }

    /// Get the number of channels.
    #[inline]
    pub const fn num_channels(&self) -> u16 {
        self.channels
    }

    /// The native sample type of the sound data.
    pub const fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    /// Fill `buf` from `leftover` first, then the underlying reader.
    fn read_bytes(&mut self, buf: &mut [u8]) -> AudioIOResult<usize> {
        let mut total = 0;
        if !self.leftover.is_empty() {
            let take = self.leftover.len().min(buf.len());
            buf[..take].copy_from_slice(&self.leftover[..take]);
            self.leftover.drain(..take);
            total += take;
        }
        if total < buf.len() {
            total += self.reader.read(&mut buf[total..])?;
        }
        Ok(total)
    }

    /// Read frames into a pre-allocated `AudioSamples` buffer.
    ///
    /// Returns the actual number of frames read (may be less at end of stream).
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

        let channels = self.channels as usize;
        let bps = self.sample_type.bytes_per_sample().get();
        let frame_bytes = channels * bps;
        let bytes_to_read = frames_to_read * frame_bytes;

        if self.byte_buffer.len() < bytes_to_read {
            self.byte_buffer.resize(bytes_to_read, 0);
        }

        let mut byte_buffer = std::mem::take(&mut self.byte_buffer);
        let bytes_read = self.read_bytes(&mut byte_buffer[..bytes_to_read]);
        self.byte_buffer = byte_buffer;
        let bytes_read = bytes_read?;
        let frames_read = bytes_read / frame_bytes;

        if frames_read == 0 {
            return Ok(0);
        }

        // Decode directly into planar (channel-major) order — SSND is
        // interleaved, so this both converts and deinterleaves in one pass.
        let mut planar: Vec<T> = Vec::with_capacity(channels * frames_read);
        for ch in 0..channels {
            for frame in 0..frames_read {
                let at = (frame * channels + ch) * bps;
                let sample = &self.byte_buffer[at..at + bps];
                planar.push(decode_sample_be(sample, self.encoding, self.sample_type));
            }
        }

        // SAFETY: planar.len() == channels * frames_read, and both are > 0.
        let converted = unsafe { NonEmptyVec::new_unchecked(planar) };
        buffer.replace_with_vec(&converted)?;

        self.current_frame += frames_read;
        Ok(frames_read)
    }
}

impl<R: std::io::Read> AudioFileMetadata for StreamedAiffFile<R> {
    fn open_metadata<P: AsRef<Path>>(_path: P) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        Err(AudioIOError::corrupted_data_simple(
            "StreamedAiffFile requires a reader",
            "Use StreamedAiffFile::new() instead",
        ))
    }

    fn base_info(&self) -> AudioIOResult<BaseAudioInfo> {
        let bytes_per_sample = self.sample_type.bytes_per_sample().get() as u16;
        let block_align = self.channels * bytes_per_sample;
        let byte_rate = self.sample_rate.get() * block_align as u32;
        let duration = Duration::from_secs_f64(self.total_frames as f64 / self.sample_rate.get() as f64);

        Ok(BaseAudioInfo::new(
            self.sample_rate,
            self.channels,
            self.bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            self.total_frames * self.channels as usize,
            duration,
            FileType::AIFF,
            self.sample_type.into(),
        ))
    }

    fn specific_info(&self) -> impl AudioInfoMarker {
        AiffFileInfo {
            form_type: self.form_type,
            compression: self.compression,
            available_chunks: self.available_chunks.clone(),
        }
    }

    fn file_type(&self) -> FileType {
        FileType::AIFF
    }

    fn file_path(&self) -> &Path {
        &self.file_path
    }

    fn total_samples(&self) -> usize {
        self.total_frames * self.channels as usize
    }

    fn duration(&self) -> AudioIOResult<Duration> {
        Ok(Duration::from_secs_f64(
            self.total_frames as f64 / self.sample_rate.get() as f64,
        ))
    }

    fn samples_per_channel(&self) -> AudioIOResult<usize> {
        Ok(self.total_frames)
    }

    fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    fn num_channels(&self) -> u16 {
        self.channels
    }
}

impl<R> AudioStreamReader for StreamedAiffFile<R>
where
    R: ReadSeek,
{
    #[inline]
    fn current_frame(&self) -> usize {
        StreamedAiffFile::current_frame(self)
    }

    #[inline]
    fn remaining_frames(&self) -> usize {
        StreamedAiffFile::remaining_frames(self)
    }

    #[inline]
    fn total_frames(&self) -> usize {
        StreamedAiffFile::total_frames(self)
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        StreamedAiffFile::sample_rate(self)
    }

    #[inline]
    fn bytes_per_frame(&self) -> usize {
        StreamedAiffFile::bytes_per_frame(self)
    }

    #[inline]
    fn num_channels(&self) -> u16 {
        StreamedAiffFile::num_channels(self)
    }

    fn seek_to_frame(&mut self, frame: usize) -> AudioIOResult<()> {
        StreamedAiffFile::seek_to_frame(self, frame)
    }

    fn reset(&mut self) -> AudioIOResult<()> {
        StreamedAiffFile::reset(self)
    }
}

impl<R> AudioStreamRead for StreamedAiffFile<R>
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
        StreamedAiffFile::read_frames_into(self, buffer, frame_count)
    }
}
