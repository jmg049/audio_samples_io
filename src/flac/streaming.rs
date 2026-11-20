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

use crate::{
    ReadSeek,
    error::{AudioIOError, AudioIOResult},
    flac::{
        FlacFileInfo,
        constants::FLAC_MARKER,
        error::FlacError,
        frame::{decode_frame_into_scratch, parse_frame_position},
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
    /// Capacity `read_buf` gets at its first use.
    read_buf_capacity: usize,
    buf_start: usize,
    buf_end: usize,
    reader_exhausted: bool,
    /// Per-channel decoded samples not yet consumed.
    pending: Vec<Vec<i32>>,
    /// Index into pending[ch] of the next unconsumed sample.
    pending_start: usize,
    /// Per-channel decode scratch (reused across frames to avoid allocation).
    scratch: Vec<Vec<i32>>,
    /// File offset of `read_buf[0]`.
    buf_file_offset: u64,
    /// First sample of the next frame to decode.
    next_frame_sample: u64,
    /// First sample of the frame in `pending`.
    pending_first_sample: u64,
    /// `(first sample, file offset)` of every frame decoded so far, ascending.
    frame_index: Vec<(u64, u64)>,
    /// Whether `frame_index` is kept.
    index_frames: bool,
}

// ─── Constructors ────────────────────────────────────────────────────────────

impl<R: std::io::Read> StreamedFlacFile<R> {
    /// Create a new streaming FLAC reader from any `Read` source.
    ///
    /// The file path is set to `"<stream>"` for error messages. Note that
    /// [`seek_to_frame`](AudioStreamReader::seek_to_frame)/[`reset`](AudioStreamReader::reset)
    /// (and therefore the [`AudioStreamReader`]/[`AudioStreamRead`] trait
    /// impls) additionally require `R: Seek` — a plain `Read` source (e.g. a
    /// network transport) can still be constructed and decoded via the
    /// inherent [`read_frames_into`](Self::read_frames_into), just not seek.
    pub fn new(reader: R) -> AudioIOResult<Self> {
        Self::new_with_path(reader, PathBuf::from("<stream>"))
    }

    /// Get the number of channels in the stream.
    #[inline]
    pub const fn num_channels(&self) -> u16 {
        self.stream_info.channels as u16
    }

    /// Get the sample rate in Hz.
    #[inline]
    pub const fn sample_rate(&self) -> u32 {
        self.stream_info.sample_rate
    }

    /// Get the current frame position.
    #[inline]
    pub const fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Get the total number of frames in the stream.
    #[inline]
    pub const fn total_frames(&self) -> usize {
        self.total_frames
    }

    /// Get the number of remaining frames from the current position.
    #[inline]
    pub const fn remaining_frames(&self) -> usize {
        self.total_frames.saturating_sub(self.current_frame)
    }

    /// Nominal uncompressed bytes per inter-channel sample (frame).
    #[inline]
    pub const fn bytes_per_frame(&self) -> usize {
        self.stream_info.channels as usize * self.stream_info.bits_per_sample.div_ceil(8) as usize
    }

    /// Create a new streaming FLAC reader with an associated path.
    pub fn new_with_path(mut reader: R, file_path: PathBuf) -> AudioIOResult<Self> {
        // Tracked manually (rather than via `Seek::stream_position`) so this
        // constructor works for any `Read` source, not just `Read + Seek`.
        let mut bytes_read: u64 = 0;

        // Read and validate the 4-byte FLAC marker.
        let mut marker = [0u8; 4];
        reader.read_exact(&mut marker).map_err(|_| {
            AudioIOError::corrupted_data_simple(
                "File too small to be a valid FLAC file",
                "Could not read 4-byte FLAC marker",
            )
        })?;
        bytes_read += 4;
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
            bytes_read += 4;

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
            bytes_read += block_size as u64;

            // Parse STREAMINFO (block type 0).
            if block_type == MetadataBlockType::StreamInfo {
                stream_info = Some(StreamInfo::from_bytes(&body).map_err(AudioIOError::FlacError)?);
            }
        }

        let stream_info = stream_info.ok_or(AudioIOError::FlacError(FlacError::MissingStreamInfo))?;

        // Audio data starts exactly where the metadata-block loop stopped
        // reading — unlike WAV/AIFF's growing header buffer, this parser
        // never reads past the boundary, so no overshoot to account for.
        let audio_data_start = bytes_read;

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

        // Choose read buffer capacity: at least two frames, so that a frame found in the buffer
        // is whole; 64 KiB when max_frame_size is unknown (0).
        let buf_cap = if stream_info.max_frame_size == 0 {
            65536
        } else {
            32768_usize.max(stream_info.max_frame_size as usize * 2)
        };

        // Allocated when the first frame is decoded: a reader may only want the metadata.
        let scratch: Vec<Vec<i32>> = (0..num_channels).map(|_| Vec::new()).collect();
        let pending: Vec<Vec<i32>> = (0..num_channels).map(|_| Vec::new()).collect();

        Ok(StreamedFlacFile {
            reader,
            file_path,
            stream_info,
            validated_sample_type,
            audio_data_start,
            total_frames,
            current_frame: 0,
            read_buf: Vec::new(),
            read_buf_capacity: buf_cap,
            buf_start: 0,
            buf_end: 0,
            reader_exhausted: false,
            pending,
            pending_start: 0,
            scratch,
            buf_file_offset: audio_data_start,
            next_frame_sample: 0,
            pending_first_sample: 0,
            frame_index: Vec::new(),
            index_frames: true,
        })
    }
}

/// Read until `buf` is full or the source ends; returns the bytes read.
fn read_up_to<R: std::io::Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {},
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

impl<R: std::io::Read> StreamedFlacFile<R> {
    /// Compact remaining data to the front of `read_buf`, grow if needed,
    /// then read more bytes from `self.reader`.
    fn refill_buf(&mut self) -> AudioIOResult<()> {
        // Compact: move unread bytes to the front.
        if self.buf_start > 0 {
            self.read_buf.copy_within(self.buf_start..self.buf_end, 0);
            self.buf_end -= self.buf_start;
            self.buf_file_offset += self.buf_start as u64;
            self.buf_start = 0;
        }

        if self.read_buf.capacity() == 0 {
            self.read_buf.reserve_exact(self.read_buf_capacity);
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

    /// Decode the next FLAC frame into `self.pending`.
    ///
    /// Returns `Ok(true)` when a frame was decoded and `Ok(false)` at end-of-stream.
    fn decode_next_frame(&mut self) -> AudioIOResult<bool> {
        if !self.decode_next_into_scratch()? {
            return Ok(false);
        }
        std::mem::swap(&mut self.pending, &mut self.scratch);
        self.pending_start = 0;
        self.pending_first_sample = self.next_frame_sample - self.pending[0].len() as u64;
        Ok(true)
    }

    /// Decode the next FLAC frame from the internal buffer into `self.scratch`.
    ///
    /// Returns `Ok(true)` when a frame was successfully decoded and `Ok(false)` at
    /// end-of-stream.
    fn decode_next_into_scratch(&mut self) -> AudioIOResult<bool> {
        loop {
            // Refill before a frame might be cut off: decoding half a frame and retrying costs
            // more than moving a few bytes. Without a known frame size, at least the sync code.
            let wanted = (self.stream_info.max_frame_size as usize).max(2);
            if self.buf_end - self.buf_start < wanted && !self.reader_exhausted {
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
                    let frame_offset = self.buf_file_offset + self.buf_start as u64;
                    if self.index_frames && self.frame_index.last().is_none_or(|&(_, off)| off < frame_offset) {
                        self.frame_index.push((self.next_frame_sample, frame_offset));
                    }
                    self.buf_start += bytes_consumed;
                    self.next_frame_sample += self.scratch[0].len() as u64;

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

    /// Do not record frame positions, for a reader that will only read straight through.
    pub fn without_seek_index(mut self) -> Self {
        self.index_frames = false;
        self
    }

    /// The stream's parameters and where its frames start, for decoding the file as a whole.
    #[cfg(feature = "numpy")]
    pub(crate) const fn stream_parameters(&self) -> (&StreamInfo, u64) {
        (&self.stream_info, self.audio_data_start)
    }

    /// Fill `out`, planar with `out.len() / channels` samples per channel, from the decoded
    /// stream. Returns the samples per channel read, which is less than asked at the end of the
    /// stream. Channel `c` starts at `out[c * per_channel]` whatever the count read.
    pub fn read_planar_into<T>(&mut self, out: &mut [T]) -> AudioIOResult<usize>
    where
        T: StandardSample + 'static,
    {
        let num_channels = self.stream_info.channels as usize;
        let per_channel = out.len() / num_channels;
        let bits = self.stream_info.bits_per_sample;
        let convert = move |s: i32| -> T {
            match bits {
                1..=8 => T::convert_from(((s) << (16 - bits as u32)) as i16),
                9..=16 => T::convert_from(s as i16),
                17..=24 => T::convert_from(I24::wrapping_from_i32(s)),
                _ => T::convert_from(s),
            }
        };

        let mut filled = 0usize;
        while filled < per_channel {
            let available = self.pending_available();
            if available > 0 {
                let take = available.min(per_channel - filled);
                let start = self.pending_start;
                for (ch, pending_ch) in self.pending.iter().enumerate() {
                    let dst = &mut out[ch * per_channel + filled..ch * per_channel + filled + take];
                    for (d, &s) in dst.iter_mut().zip(&pending_ch[start..start + take]) {
                        *d = convert(s);
                    }
                }
                self.pending_start += take;
                filled += take;
            } else {
                if !self.decode_next_into_scratch()? {
                    break;
                }
                let block = self.scratch[0].len();
                if block <= per_channel - filled {
                    // A whole frame fits: convert it where it was decoded.
                    for (ch, samples) in self.scratch.iter().enumerate() {
                        let dst = &mut out[ch * per_channel + filled..ch * per_channel + filled + block];
                        for (d, &s) in dst.iter_mut().zip(samples) {
                            *d = convert(s);
                        }
                    }
                    filled += block;
                    for ch in &mut self.pending {
                        ch.clear();
                    }
                    self.pending_start = 0;
                } else {
                    std::mem::swap(&mut self.pending, &mut self.scratch);
                    self.pending_start = 0;
                    self.pending_first_sample = self.next_frame_sample - block as u64;
                }
            }
        }
        self.current_frame += filled;
        Ok(filled)
    }

    /// Number of decoded samples available in `pending` but not yet consumed.
    #[inline]
    fn pending_available(&self) -> usize {
        self.pending
            .first()
            .map(|ch| ch.len().saturating_sub(self.pending_start))
            .unwrap_or(0)
    }

    /// Decode frames into a pre-allocated `AudioSamples` buffer.
    ///
    /// This is the primary zero-allocation-after-warmup read method (only the
    /// per-call `out`/`flat` staging vectors allocate). Available for any
    /// `Read` source — the [`AudioStreamRead`] trait impl below (`R: Read +
    /// Seek`) just delegates to this. The buffer must have the file's channel count
    /// and sample rate, and keeps its layout (see [`AudioStreamRead::read_frames_into`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer's channel count or sample rate differs from the
    /// file's (nothing is decoded), or if decoding fails.
    pub fn read_frames_into<T>(
        &mut self,
        buffer: &mut AudioSamples<'_, T>,
        frame_count: NonZeroUsize,
    ) -> AudioIOResult<usize>
    where
        T: StandardSample + ConvertTo<T> + ConvertFrom<T> + 'static,
    {
        let num_channels = self.stream_info.channels as usize;
        crate::traits::check_read_buffer(buffer, num_channels, self.stream_info.sample_rate)?;
        let frames_to_read = frame_count.get().min(self.remaining_frames());
        if frames_to_read == 0 {
            return Ok(0);
        }

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

        crate::traits::fill_planar(buffer, flat, num_channels)?;

        Ok(actual_frames)
    }
}

impl<R: ReadSeek> StreamedFlacFile<R> {
    /// Continue decoding from the frame at `offset`, which starts at `first_sample`.
    fn restart_at(&mut self, first_sample: u64, offset: u64) -> AudioIOResult<()> {
        self.reader.seek(SeekFrom::Start(offset)).map_err(AudioIOError::from)?;
        self.buf_file_offset = offset;
        self.buf_start = 0;
        self.buf_end = 0;
        self.reader_exhausted = false;
        self.pending_start = 0;
        self.next_frame_sample = first_sample;
        self.pending_first_sample = first_sample;
        self.current_frame = first_sample as usize;
        for ch in &mut self.pending {
            ch.clear();
        }
        Ok(())
    }

    /// Drop decoded samples until the next read returns sample `frame`.
    fn discard_until(&mut self, frame: usize) -> AudioIOResult<()> {
        while self.current_frame < frame {
            let available = self.pending_available();
            if available > 0 {
                let advance = available.min(frame - self.current_frame);
                self.pending_start += advance;
                self.current_frame += advance;
            } else if !self.decode_next_frame()? {
                // EOF before reaching the target frame.
                break;
            }
        }
        Ok(())
    }

    /// Bisect the file for a frame at or just before sample `frame`, starting from the known
    /// frame `lo`. Frames are found by their sync code and validated by their CRC-8.
    fn bisect_for_frame(&mut self, frame: u64, mut lo: (u64, u64)) -> AudioIOResult<(u64, u64)> {
        let channels = self.stream_info.channels as usize;
        let bits = self.stream_info.bits_per_sample;
        let block = u64::from(self.stream_info.max_block_size.max(1));
        let min_block = u32::from(self.stream_info.min_block_size);
        let probe_len = 16384usize.max(self.stream_info.max_frame_size as usize * 2);
        let mut probe = vec![0u8; probe_len];
        let mut hi = self.reader.seek(SeekFrom::End(0)).map_err(AudioIOError::from)?;
        let mut hi_sample = self.total_frames as u64;

        for _ in 0..48 {
            if frame.saturating_sub(lo.0) <= 2 * block || hi <= lo.1 + 1 {
                break;
            }
            // Interpolate within the bracket, clear of its ends.
            let span_samples = hi_sample.saturating_sub(lo.0).max(1);
            let estimate = (u128::from(hi - lo.1) * u128::from(frame - lo.0) / u128::from(span_samples)) as u64;
            let at = lo.1 + estimate.min(hi - lo.1 - 1);

            self.reader.seek(SeekFrom::Start(at)).map_err(AudioIOError::from)?;
            let n = read_up_to(&mut self.reader, &mut probe).map_err(AudioIOError::from)?;
            let found = (0..n.saturating_sub(1)).find_map(|i| {
                if probe[i] != 0xFF || (probe[i + 1] & 0xFC) != 0xF8 {
                    return None;
                }
                parse_frame_position(&probe[i..n], min_block, channels, bits).map(|(sample, _)| (sample, at + i as u64))
            });
            match found {
                Some((sample, offset)) if sample <= frame && sample > lo.0 => lo = (sample, offset),
                Some((sample, _)) if sample > frame => {
                    hi = at;
                    hi_sample = sample;
                },
                // Nothing usable from here on: the wanted frame is before this probe.
                _ => {
                    hi = at;
                },
            }
        }
        Ok(lo)
    }
}

// ─── AudioFileMetadata ────────────────────────────────────────────────────────

impl<R: std::io::Read> AudioFileMetadata for StreamedFlacFile<R> {
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
        StreamedFlacFile::current_frame(self)
    }

    #[inline]
    fn remaining_frames(&self) -> usize {
        StreamedFlacFile::remaining_frames(self)
    }

    #[inline]
    fn total_frames(&self) -> usize {
        StreamedFlacFile::total_frames(self)
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        StreamedFlacFile::sample_rate(self)
    }

    #[inline]
    fn bytes_per_frame(&self) -> usize {
        StreamedFlacFile::bytes_per_frame(self)
    }

    #[inline]
    fn num_channels(&self) -> u16 {
        StreamedFlacFile::num_channels(self)
    }

    fn seek_to_frame(&mut self, frame: usize) -> AudioIOResult<()> {
        if frame > self.total_frames {
            return Err(AudioIOError::SeekError(format!(
                "Frame {} is beyond end of stream (total frames: {})",
                frame, self.total_frames
            )));
        }

        let target = frame as u64;
        let block = u64::from(self.stream_info.max_block_size.max(1));

        // Inside the block already decoded: no I/O.
        let pending_len = self.pending.first().map_or(0, Vec::len) as u64;
        if pending_len > 0 && (self.pending_first_sample..=self.pending_first_sample + pending_len).contains(&target) {
            self.pending_start = (target - self.pending_first_sample) as usize;
            self.current_frame = frame;
            return Ok(());
        }

        // A short way ahead: keep decoding from here.
        if frame >= self.current_frame && target - self.current_frame as u64 <= 4 * block {
            return self.discard_until(frame);
        }

        // The nearest frame decoded so far at or before the target...
        let known = match self.frame_index.partition_point(|&(sample, _)| sample <= target) {
            0 => (0, self.audio_data_start),
            i => self.frame_index[i - 1],
        };
        // ... or, when that is far away, one found by bisecting the file.
        let start = if target - known.0 > 8 * block && self.stream_info.total_samples > 0 {
            self.bisect_for_frame(target, known)?
        } else {
            known
        };

        self.restart_at(start.0, start.1)?;
        match self.discard_until(frame) {
            // A false sync that passed the CRC-8: decode from the known frame instead.
            Err(_) if start != known => {
                self.restart_at(known.0, known.1)?;
                self.discard_until(frame)
            },
            other => other,
        }
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
        StreamedFlacFile::read_frames_into(self, buffer, frame_count)
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

    /// Wraps any `Read` but deliberately does *not* implement `Seek` — proves
    /// at the type level that construction and `read_frames_into` genuinely
    /// don't require it (a network transport, for instance, could use this).
    struct ReadOnly<R>(R);

    impl<R: std::io::Read> std::io::Read for ReadOnly<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(buf)
        }
    }

    #[test]
    fn test_streamed_flac_works_over_read_only_source() {
        let bytes = std::fs::read("resources/test.flac").expect("read test.flac");
        let expected_total = open_test_flac().total_frames();

        let mut s = StreamedFlacFile::new(ReadOnly(std::io::Cursor::new(bytes))).expect("new over Read-only source");
        assert_eq!(s.total_frames(), expected_total);

        let sr = NonZeroU32::new(s.sample_rate()).unwrap();
        let mut buf = if s.num_channels() == 1 {
            AudioSamples::<f32>::zeros_mono(NonZeroUsize::new(1024).unwrap(), sr)
        } else {
            let ch = NonZeroU32::new(s.num_channels() as u32).unwrap();
            AudioSamples::<f32>::zeros_multi(ch, NonZeroUsize::new(1024).unwrap(), sr)
        };

        let mut frames_read = 0;
        loop {
            let n = s.read_frames_into(&mut buf, nzu!(1024)).expect("read");
            if n == 0 {
                break;
            }
            frames_read += n;
        }
        assert_eq!(
            frames_read, expected_total,
            "Read-only decode must reach the same total as a Read+Seek source"
        );
    }
}
