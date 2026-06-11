//! Streaming FLAC reader for memory-efficient on-demand frame decoding.
//!
//! This module provides `StreamedFlacFile`, a streaming reader that parses FLAC
//! metadata on construction but decodes audio frames on demand, enabling
//! processing of large files without loading them entirely into memory.

use std::{
    io::SeekFrom,
    num::{NonZeroU32, NonZeroUsize},
    path::{Path, PathBuf},
    time::Duration,
};

use audio_samples::{AudioSamples, ConvertFrom, ConvertTo, I24, traits::StandardSample};
use non_empty_slice::NonEmptyVec;

use crate::{
    ReadSeek,
    error::{AudioIOError, AudioIOResult},
    flac::{
        FlacFileInfo,
        constants::FLAC_MARKER,
        error::FlacError,
        frame::decode_frame_into_scratch,
        metadata::{MetadataBlockType, StreamInfo},
    },
    traits::{AudioFileMetadata, AudioStreamRead, AudioStreamReader},
    types::{BaseAudioInfo, FileType, ValidatedSampleType},
};

/// A streaming FLAC reader that decodes frames on demand.
///
/// Unlike `FlacFile` which loads everything into memory before decoding,
/// `StreamedFlacFile` decodes one FLAC frame at a time as you call
/// `read_frames_into`, keeping only a small sliding window of raw bytes
/// plus the most recently decoded block in memory.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::flac::StreamedFlacFile;
/// use audio_samples_io::traits::{AudioFileMetadata, AudioStreamRead, AudioStreamReader};
/// use audio_samples::{AudioSamples, nzu};
/// use std::{fs::File, io::BufReader, num::NonZeroU32};
///
/// let file = BufReader::new(File::open("audio.flac")?);
/// let mut streamed = StreamedFlacFile::new(file)?;
///
/// let sample_rate = NonZeroU32::new(streamed.sample_rate()).unwrap();
/// let channels = NonZeroU32::new(streamed.num_channels() as u32).unwrap();
/// let mut buffer = AudioSamples::<f32>::zeros_multi(channels, nzu!(4096), sample_rate);
///
/// while streamed.remaining_frames() > 0 {
///     let frames_read = streamed.read_frames_into(&mut buffer, nzu!(4096))?;
///     // Process buffer…
///     if frames_read == 0 { break; }
/// }
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
pub struct StreamedFlacFile<R> {
    reader: R,
    file_path: PathBuf,
    stream_info: StreamInfo,
    validated_sample_type: ValidatedSampleType,
    /// Byte offset from file start where audio frames begin.
    audio_data_start: u64,
    /// Per-channel sample count (= stream_info.total_samples).
    total_frames: usize,
    /// Per-channel samples consumed so far.
    current_frame: usize,
    /// Sliding window of raw compressed bytes.
    read_buf: Vec<u8>,
    buf_start: usize,
    buf_end: usize,
    reader_exhausted: bool,
    /// Per-channel decoded samples not yet consumed.
    pending: Vec<Vec<i32>>,
    /// Index into pending[ch] of the next unconsumed sample.
    pending_start: usize,
    /// Per-channel decode scratch (reused across frames to avoid allocation).
    scratch: Vec<Vec<i32>>,
}

// ─── Constructors ────────────────────────────────────────────────────────────

impl<R: ReadSeek> StreamedFlacFile<R> {
    /// Create a new streaming FLAC reader from any `Read + Seek` source.
    ///
    /// The file path is set to `"<stream>"` for error messages.
    pub fn new(reader: R) -> AudioIOResult<Self> {
        Self::new_with_path(reader, PathBuf::from("<stream>"))
    }

    /// Get the number of channels in the stream.
    #[inline]
    pub const fn num_channels(&self) -> u16 {
        self.stream_info.channels as u16
    }

    /// Create a new streaming FLAC reader with an associated path.
    pub fn new_with_path(mut reader: R, file_path: PathBuf) -> AudioIOResult<Self> {
        // Read and validate the 4-byte FLAC marker.
        let mut marker = [0u8; 4];
        reader.read_exact(&mut marker).map_err(|_| {
            AudioIOError::corrupted_data_simple(
                "File too small to be a valid FLAC file",
                "Could not read 4-byte FLAC marker",
            )
        })?;
        if marker != FLAC_MARKER {
            return Err(AudioIOError::FlacError(FlacError::invalid_marker(marker)));
        }

        // Streaming-parse metadata blocks until the is_last flag is set.
        let mut stream_info: Option<StreamInfo> = None;
        let mut is_last = false;

        while !is_last {
            // Each metadata block header is exactly 4 bytes:
            //   byte 0:   [is_last (1 bit) | block_type (7 bits)]
            //   bytes 1-3: block data length (big-endian 24-bit)
            let mut header = [0u8; 4];
            reader.read_exact(&mut header).map_err(|_| {
                AudioIOError::corrupted_data_simple(
                    "Truncated FLAC metadata block header",
                    "Could not read 4-byte metadata block header",
                )
            })?;

            is_last = (header[0] & 0x80) != 0;
            let block_type_byte = header[0] & 0x7F;
            let block_size = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;

            let block_type = MetadataBlockType::from_byte(block_type_byte);

            // Read the block body.
            let mut body = vec![0u8; block_size];
            reader.read_exact(&mut body).map_err(|_| {
                AudioIOError::corrupted_data_simple(
                    "Truncated FLAC metadata block body",
                    format!(
                        "Block type {:?} claimed {} bytes but stream ended early",
                        block_type, block_size
                    ),
                )
            })?;

            // Parse STREAMINFO (block type 0).
            if block_type == MetadataBlockType::StreamInfo {
                stream_info = Some(StreamInfo::from_bytes(&body).map_err(AudioIOError::FlacError)?);
            }
        }

        let stream_info = stream_info.ok_or(AudioIOError::FlacError(FlacError::MissingStreamInfo))?;

        // Record where audio data starts (current reader position after all metadata).
        let audio_data_start = reader.stream_position().map_err(AudioIOError::from)?;

        // Determine validated sample type from bits_per_sample.
        let validated_sample_type = match stream_info.bits_per_sample {
            1..=16 => ValidatedSampleType::I16,
            17..=24 => ValidatedSampleType::I24,
            25..=32 => ValidatedSampleType::I32,
            bits => {
                return Err(AudioIOError::FlacError(FlacError::InvalidBitsPerSample { bits }));
            },
        };

        let num_channels = stream_info.channels as usize;
        let total_frames = stream_info.total_samples as usize;

        // Choose read buffer capacity: generous enough for one typical frame,
        // but fall back to 64 KiB when max_frame_size is unknown (0).
        let buf_cap = if stream_info.max_frame_size == 0 {
            65536
        } else {
            65536_usize.max(stream_info.max_frame_size as usize * 4)
        };

        let block_size_hint = stream_info.max_block_size as usize;
        let scratch: Vec<Vec<i32>> = (0..num_channels).map(|_| Vec::with_capacity(block_size_hint)).collect();
        let pending: Vec<Vec<i32>> = (0..num_channels).map(|_| Vec::new()).collect();

        Ok(StreamedFlacFile {
            reader,
            file_path,
            stream_info,
            validated_sample_type,
            audio_data_start,
            total_frames,
            current_frame: 0,
            read_buf: Vec::with_capacity(buf_cap),
            buf_start: 0,
            buf_end: 0,
            reader_exhausted: false,
            pending,
            pending_start: 0,
            scratch,
        })
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

impl<R: ReadSeek> StreamedFlacFile<R> {
    /// Compact remaining data to the front of `read_buf`, grow if needed,
    /// then read more bytes from `self.reader`.
    fn refill_buf(&mut self) -> AudioIOResult<()> {
        // Compact: move unread bytes to the front.
        if self.buf_start > 0 {
            self.read_buf.copy_within(self.buf_start..self.buf_end, 0);
            self.buf_end -= self.buf_start;
            self.buf_start = 0;
        }

        // Grow the buffer if it's full.
        if self.buf_end == self.read_buf.capacity() {
            let new_cap = (self.read_buf.capacity() * 2).max(65536);
            self.read_buf.reserve(new_cap - self.read_buf.capacity());
        }

        // Extend the buffer to its capacity before reading.
        let cap = self.read_buf.capacity();
        self.read_buf.resize(cap, 0);

        // Read more bytes.
        let n = self
            .reader
            .read(&mut self.read_buf[self.buf_end..])
            .map_err(AudioIOError::from)?;

        if n == 0 {
            self.reader_exhausted = true;
        }
        self.buf_end += n;
        Ok(())
    }

    /// Decode the next FLAC frame from the internal buffer.
    ///
    /// Returns `Ok(true)` when a frame was successfully decoded and placed into
    /// `self.pending`.  Returns `Ok(false)` at end-of-stream.
    fn decode_next_frame(&mut self) -> AudioIOResult<bool> {
        loop {
            // Ensure we have at least 2 bytes to check the sync code.
            if self.buf_end - self.buf_start < 2 && !self.reader_exhausted {
                self.refill_buf()?;
            }

            if self.buf_end == self.buf_start {
                // Truly EOF.
                return Ok(false);
            }

            let data = &self.read_buf[self.buf_start..self.buf_end];

            if data.len() < 2 {
                // Not enough bytes even after refill.
                return Ok(false);
            }

            // Check for FLAC frame sync: 0xFF followed by byte where top 6 bits = 0b111110
            if data[0] != 0xFF || (data[1] & 0xFC) != 0xF8 {
                // Not a sync word — advance by one byte and keep searching.
                self.buf_start += 1;
                continue;
            }

            // Attempt to decode a frame.
            match decode_frame_into_scratch(
                data,
                self.stream_info.sample_rate,
                self.stream_info.bits_per_sample,
                &mut self.scratch,
            ) {
                Ok(bytes_consumed) => {
                    self.buf_start += bytes_consumed;

                    // Move scratch into pending.
                    let num_channels = self.stream_info.channels as usize;
                    for ch in 0..num_channels {
                        self.pending[ch].clear();
                        self.pending[ch].extend_from_slice(&self.scratch[ch]);
                    }
                    self.pending_start = 0;

                    return Ok(true);
                },
                Err(FlacError::UnexpectedEof) if !self.reader_exhausted => {
                    // We ran out of buffered bytes mid-frame — fetch more and retry.
                    self.refill_buf()?;
                    // Loop back to retry at same buf_start position.
                },
                Err(FlacError::UnexpectedEof) => {
                    // EOF and still not enough bytes — stream is truncated.
                    return Ok(false);
                },
                Err(FlacError::InvalidFrameSync { .. }) => {
                    // Bad sync at this position — skip one byte and keep searching.
                    self.buf_start += 1;
                },
                Err(other) => {
                    return Err(AudioIOError::FlacError(other));
                },
            }
        }
    }

    /// Number of decoded samples available in `pending` but not yet consumed.
    #[inline]
    fn pending_available(&self) -> usize {
        self.pending
            .first()
            .map(|ch| ch.len().saturating_sub(self.pending_start))
            .unwrap_or(0)
    }
}

// ─── AudioFileMetadata ────────────────────────────────────────────────────────

impl<R: ReadSeek> AudioFileMetadata for StreamedFlacFile<R> {
    fn open_metadata<P: AsRef<Path>>(_path: P) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        Err(AudioIOError::corrupted_data_simple(
            "StreamedFlacFile requires a reader",
            "Use StreamedFlacFile::new() instead",
        ))
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
            self.validated_sample_type.into(),
        ))
    }

    #[allow(refining_impl_trait)]
    fn specific_info(&self) -> FlacFileInfo {
        let si = &self.stream_info;
        FlacFileInfo {
            // We don't track metadata blocks during streaming parse.
            metadata_blocks: vec![],
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
        self.total_frames * self.stream_info.channels as usize
    }

    fn duration(&self) -> AudioIOResult<Duration> {
        self.base_info().map(|info| info.duration)
    }

    fn sample_type(&self) -> ValidatedSampleType {
        self.validated_sample_type
    }

    fn num_channels(&self) -> u16 {
        self.stream_info.channels as u16
    }
}

// ─── AudioStreamReader ────────────────────────────────────────────────────────

impl<R: ReadSeek> AudioStreamReader for StreamedFlacFile<R> {
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
        self.stream_info.sample_rate
    }

    /// Nominal uncompressed bytes per inter-channel sample (frame).
    #[inline]
    fn bytes_per_frame(&self) -> usize {
        self.stream_info.channels as usize * self.stream_info.bits_per_sample.div_ceil(8) as usize
    }

    #[inline]
    fn num_channels(&self) -> u16 {
        self.stream_info.channels as u16
    }

    fn seek_to_frame(&mut self, frame: usize) -> AudioIOResult<()> {
        if frame > self.total_frames {
            return Err(AudioIOError::SeekError(format!(
                "Frame {} is beyond end of stream (total frames: {})",
                frame, self.total_frames
            )));
        }

        // Seek the underlying reader back to the start of audio data.
        self.reader
            .seek(SeekFrom::Start(self.audio_data_start))
            .map_err(AudioIOError::from)?;

        // Reset all internal state.
        self.buf_start = 0;
        self.buf_end = 0;
        self.reader_exhausted = false;
        self.pending_start = 0;
        self.current_frame = 0;
        for ch in &mut self.pending {
            ch.clear();
        }

        // Decode-and-discard frames until we reach the target position.
        while self.current_frame < frame {
            let available = self.pending_available();
            if available > 0 {
                let advance = available.min(frame - self.current_frame);
                self.pending_start += advance;
                self.current_frame += advance;
            } else {
                // Need to decode another frame.
                match self.decode_next_frame()? {
                    true => {
                        // pending now has a freshly decoded block; loop will drain it.
                    },
                    false => {
                        // EOF before reaching target frame — stop here.
                        break;
                    },
                }
            }
        }

        Ok(())
    }

    fn reset(&mut self) -> AudioIOResult<()> {
        self.seek_to_frame(0)
    }
}

// ─── AudioStreamRead ──────────────────────────────────────────────────────────

impl<R: ReadSeek> AudioStreamRead for StreamedFlacFile<R> {
    fn read_frames_into<T>(
        &mut self,
        buffer: &mut AudioSamples<'_, T>,
        frame_count: NonZeroUsize,
    ) -> AudioIOResult<usize>
    where
        T: StandardSample + ConvertTo<T> + ConvertFrom<T> + 'static,
    {
        let frames_to_read = frame_count.get().min(self.remaining_frames());
        if frames_to_read == 0 {
            return Ok(0);
        }

        let num_channels = self.stream_info.channels as usize;
        // One Vec<T> per channel, capacity = frames_to_read.
        let mut out: Vec<Vec<T>> = (0..num_channels).map(|_| Vec::with_capacity(frames_to_read)).collect();

        // Inline i32 → T conversion closure.
        // The match on `bits` is resolved at monomorphisation time.
        let bits = self.stream_info.bits_per_sample;
        let convert = move |s: i32| -> T {
            match bits {
                1..=8 => T::convert_from(((s) << (16 - bits as u32)) as i16),
                9..=16 => T::convert_from(s as i16),
                17..=24 => T::convert_from(I24::wrapping_from_i32(s)),
                _ => T::convert_from(s),
            }
        };

        // Fill `out` from pending decoded samples and fresh decoded frames.
        loop {
            let filled = out.first().map(|v| v.len()).unwrap_or(0);
            if filled >= frames_to_read {
                break;
            }

            let available = self.pending_available();
            if available > 0 {
                let take = available.min(frames_to_read - filled);
                let start = self.pending_start;
                let end = start + take;
                for (pending_ch, out_ch) in self.pending.iter().zip(out.iter_mut()) {
                    for &s in &pending_ch[start..end] {
                        out_ch.push(convert(s));
                    }
                }
                self.pending_start += take;
            } else {
                // Decode another frame.
                match self.decode_next_frame()? {
                    true => {
                        // pending refreshed; loop will drain it.
                    },
                    false => {
                        // EOF — stop filling.
                        break;
                    },
                }
            }
        }

        let actual_frames = out.first().map(|v| v.len()).unwrap_or(0);
        if actual_frames == 0 {
            return Ok(0);
        }

        self.current_frame += actual_frames;

        // Build a flat planar Vec<T>: [ch0[0..N], ch1[0..N], …].
        let mut flat: Vec<T> = Vec::with_capacity(num_channels * actual_frames);
        for ch_data in out {
            flat.extend(ch_data);
        }

        let non_empty = NonEmptyVec::try_from(flat).map_err(|_| {
            AudioIOError::corrupted_data_simple("Empty decoded output", "No samples after FLAC frame decode")
        })?;
        buffer.replace_with_vec(&non_empty)?;

        Ok(actual_frames)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader, num::NonZeroUsize, time::Duration};

    use audio_samples::{AudioSamples, nzu, sample_rate, sine_wave};

    use super::*;

    fn open_test_flac() -> StreamedFlacFile<BufReader<File>> {
        let file = BufReader::new(File::open("resources/test.flac").expect("test.flac"));
        StreamedFlacFile::new(file).expect("open StreamedFlacFile")
    }

    fn make_buf(s: &StreamedFlacFile<BufReader<File>>, frames: usize) -> AudioSamples<'static, f32> {
        let sr = NonZeroU32::new(s.sample_rate()).unwrap();
        if s.num_channels() == 1 {
            AudioSamples::<f32>::zeros_mono(NonZeroUsize::new(frames).unwrap(), sr)
        } else {
            let ch = NonZeroU32::new(s.num_channels() as u32).unwrap();
            AudioSamples::<f32>::zeros_multi(ch, NonZeroUsize::new(frames).unwrap(), sr)
        }
    }

    #[test]
    fn test_streamed_flac_metadata() {
        let s = open_test_flac();
        assert!(s.sample_rate() > 0);
        assert!(s.total_frames() > 0);
        assert_eq!(s.current_frame(), 0);
        assert_eq!(s.remaining_frames(), s.total_frames());
        assert!(s.num_channels() > 0);
        assert!(s.bytes_per_frame() > 0);
    }

    #[test]
    fn test_streamed_flac_base_info() {
        let s = open_test_flac();
        let info = s.base_info().expect("base_info");
        assert_eq!(info.file_type, FileType::FLAC);
        assert!(info.sample_rate.get() > 0);
        assert!(info.channels > 0);
        assert!(info.total_samples > 0);
    }

    #[test]
    fn test_streamed_flac_read_frames_advances_position() {
        let mut s = open_test_flac();
        let mut buf = make_buf(&s, 512);
        let before = s.current_frame();
        let read = s.read_frames_into(&mut buf, nzu!(512)).expect("read");
        assert!(read > 0);
        assert_eq!(s.current_frame(), before + read);
        assert_eq!(s.remaining_frames(), s.total_frames() - read);
    }

    #[test]
    fn test_streamed_flac_read_all_frames() {
        let mut s = open_test_flac();
        let total = s.total_frames();
        let mut buf = make_buf(&s, 1024);

        let mut frames_read = 0;
        while s.remaining_frames() > 0 {
            let n = s.read_frames_into(&mut buf, nzu!(1024)).expect("read");
            if n == 0 {
                break;
            }
            frames_read += n;
        }
        assert_eq!(frames_read, total);
        assert_eq!(s.current_frame(), total);
    }

    #[test]
    fn test_streamed_flac_reset() {
        let mut s = open_test_flac();
        let total = s.total_frames();
        let mut buf = make_buf(&s, 256);
        s.read_frames_into(&mut buf, nzu!(256)).expect("read");
        assert!(s.current_frame() > 0);

        s.reset().expect("reset");
        assert_eq!(s.current_frame(), 0);
        assert_eq!(s.remaining_frames(), total);
    }

    #[test]
    fn test_streamed_flac_seek_to_frame() {
        let mut s = open_test_flac();
        let total = s.total_frames();
        let target = total / 4;
        s.seek_to_frame(target).expect("seek_to_frame");
        assert_eq!(s.current_frame(), target);
        assert_eq!(s.remaining_frames(), total - target);
    }

    #[test]
    fn test_streamed_flac_matches_bulk_read() {
        use std::io::BufWriter;

        use audio_samples::AudioTypeConversion;

        use crate::flac::{CompressionLevel, FlacFile, write_flac};
        use crate::traits::{AudioFile, AudioFileRead};

        let sr = sample_rate!(44100);
        let sine = sine_wave::<f32>(440.0, Duration::from_millis(200), sr, 0.5).to_format::<i16>();

        // Write a test FLAC to a temp file
        let path = std::env::temp_dir().join("streamed_flac_cmp.flac");
        {
            let f = File::create(&path).expect("create");
            write_flac(BufWriter::new(f), &sine, CompressionLevel::FASTEST).expect("write");
        }

        // Bulk read via FlacFile
        let flac = FlacFile::open_with_options(&path, crate::types::OpenOptions::default()).unwrap();
        let bulk = flac.read::<i16>().unwrap();

        // Streaming read via StreamedFlacFile (sine_wave is mono)
        let file = BufReader::new(File::open(&path).expect("open"));
        let mut streamed = StreamedFlacFile::new(file).expect("new");
        let sr2 = NonZeroU32::new(streamed.sample_rate()).unwrap();
        let mut buf = AudioSamples::<i16>::zeros_mono(nzu!(1024), sr2);

        let mut frames_read = 0usize;

        while streamed.remaining_frames() > 0 {
            let n = streamed.read_frames_into(&mut buf, nzu!(1024)).expect("read");
            if n == 0 {
                break;
            }
            frames_read += n;
        }

        assert_eq!(
            bulk.samples_per_channel().get(),
            frames_read,
            "streamed and bulk should read the same number of frames"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_lib_open_streamed_flac() {
        let s = crate::open_streamed_flac("resources/test.flac").expect("open_streamed_flac");
        assert!(s.total_frames() > 0);
        assert_eq!(s.current_frame(), 0);
    }

    #[test]
    fn test_lib_open_streamed_dyn_flac() {
        let s = crate::open_streamed_dyn("resources/test.flac").expect("open_streamed_dyn");
        assert!(s.total_frames() > 0);
        assert_eq!(s.current_frame(), 0);
    }
}
