//! Streaming FLAC file writer for memory-efficient incremental encoding.
//!
//! This module provides [`StreamedFlacWriter`], a streaming writer that writes the FLAC
//! marker and a placeholder STREAMINFO block on construction, then encodes audio one
//! block at a time as frames are supplied. This enables writing large files without
//! buffering the whole stream in memory.
//!
//! # Why streaming FLAC is different from streaming WAV
//!
//! Unlike WAV, FLAC is a *block-based* codec: samples are grouped into fixed-size blocks
//! and each block is entropy-coded into a self-contained frame. The writer therefore
//! buffers incoming samples until a full block (`block_size`) is available, encodes that
//! block with the same routine used by the bulk [`write_flac`](super::write_flac) path,
//! and flushes the encoded frame. Any trailing partial block is encoded on
//! [`finalize`](AudioStreamWriter::finalize).
//!
//! The total sample count and block sizes in STREAMINFO are unknown up front, so the
//! writer requires a [`WriteSeek`] destination and back-patches the STREAMINFO block on
//! finalize. The MD5 signature is left zeroed (the "unknown" sentinel), exactly as the
//! bulk encoder does.

use std::io::{SeekFrom, Write};

use audio_samples::{AudioSamples, SampleType, traits::StandardSample};

use crate::{
    WriteSeek,
    error::{AudioIOError, AudioIOResult},
    traits::{AudioStreamWrite, AudioStreamWriter},
    types::ValidatedSampleType,
};

use super::{
    CompressionLevel, constants::FLAC_MARKER, flac_file::audio_to_planar_i32,
    frame::encode_frame_from_channels, metadata::StreamInfo,
};

/// Encoder parameters derived once from a [`CompressionLevel`], reused for every frame.
#[derive(Debug, Clone, Copy)]
struct FlacEncodeParams {
    max_lpc_order: usize,
    qlp_precision: u8,
    min_partition_order: u8,
    max_partition_order: u8,
    try_mid_side: bool,
    exhaustive_rice: bool,
}

impl FlacEncodeParams {
    const fn from_level(level: CompressionLevel) -> Self {
        let (min_partition_order, max_partition_order) = level.rice_partition_order_range();
        Self {
            max_lpc_order: level.max_lpc_order() as usize,
            qlp_precision: level.qlp_precision(),
            min_partition_order,
            max_partition_order,
            try_mid_side: level.try_mid_side(),
            exhaustive_rice: level.exhaustive_rice_search(),
        }
    }
}

/// A streaming FLAC writer that encodes audio incrementally, block by block.
///
/// Unlike [`write_flac`](super::write_flac), which requires the whole signal in memory,
/// `StreamedFlacWriter` writes the FLAC marker and a placeholder STREAMINFO on
/// construction and encodes frames as they are provided. This suits:
/// - Writing files larger than available memory
/// - Real-time recording / capture pipelines
/// - Streaming to seekable network destinations
///
/// # Bit depth
///
/// The output bit depth is fixed at construction from the configured sample type, matching
/// [`write_flac`]: 16-bit for `i16`, 24-bit for every other supported type. Each call to
/// [`write_frames`](AudioStreamWrite::write_frames) must supply samples that map to the
/// same bit depth.
///
/// # Finalization
///
/// Call [`finalize`](AudioStreamWriter::finalize) when done to encode the trailing partial
/// block and back-patch STREAMINFO with the final sample count. If the writer is dropped
/// without an explicit `finalize`, it finalizes on a best-effort basis (errors are
/// swallowed, since [`Drop`] cannot report them); prefer calling `finalize` yourself so
/// I/O errors surface.
#[derive(Debug)]
pub struct StreamedFlacWriter<W>
where
    W: WriteSeek,
{
    /// The underlying seekable writer.
    writer: W,
    /// Number of channels.
    channels: u16,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// FLAC output bit depth (16 or 24).
    bits_per_sample: u8,
    /// Encoding block size (samples per channel per full frame).
    block_size: u32,
    /// Encoder parameters derived from the compression level.
    params: FlacEncodeParams,
    /// Per-channel buffer of i32 samples not yet emitted as a full block.
    accum: Vec<Vec<i32>>,
    /// Next frame number (FLAC fixed-blocksize frame counter).
    frame_number: u64,
    /// Total frames (samples per channel) accepted so far.
    frames_written: usize,
    /// File offset of the 34-byte STREAMINFO body (for back-patching on finalize).
    streaminfo_offset: u64,
    /// Whether finalize() has run.
    finalized: bool,
}

impl<W> StreamedFlacWriter<W>
where
    W: WriteSeek,
{
    /// Create a streaming FLAC writer for 16-bit output.
    pub fn new_i16(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new(
            writer,
            channels,
            sample_rate,
            ValidatedSampleType::I16,
            CompressionLevel::default(),
        )
    }

    /// Create a streaming FLAC writer for 24-bit output from 24-bit PCM input.
    pub fn new_i24(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new(
            writer,
            channels,
            sample_rate,
            ValidatedSampleType::I24,
            CompressionLevel::default(),
        )
    }

    /// Create a streaming FLAC writer for 24-bit output from 32-bit PCM input.
    pub fn new_i32(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new(
            writer,
            channels,
            sample_rate,
            ValidatedSampleType::I32,
            CompressionLevel::default(),
        )
    }

    /// Create a streaming FLAC writer for 24-bit output from 32-bit float input.
    pub fn new_f32(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new(
            writer,
            channels,
            sample_rate,
            ValidatedSampleType::F32,
            CompressionLevel::default(),
        )
    }

    /// Create a streaming FLAC writer for 24-bit output from 64-bit float input.
    pub fn new_f64(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<Self> {
        Self::new(
            writer,
            channels,
            sample_rate,
            ValidatedSampleType::F64,
            CompressionLevel::default(),
        )
    }

    /// Create a streaming FLAC writer with an explicit sample type and compression level.
    ///
    /// The bit depth is derived from `sample_type` exactly as [`write_flac`](super::write_flac):
    /// 16-bit for [`ValidatedSampleType::I16`], 24-bit otherwise.
    pub fn new(
        mut writer: W,
        channels: u16,
        sample_rate: u32,
        sample_type: ValidatedSampleType,
        level: CompressionLevel,
    ) -> AudioIOResult<Self> {
        if channels == 0 || channels > 8 {
            return Err(AudioIOError::corrupted_data_simple(
                "Invalid channel count for FLAC",
                format!("FLAC supports 1-8 channels, got {channels}"),
            ));
        }

        let bits_per_sample = flac_bits_for(sample_type);
        let block_size = level.block_size();

        // Write the FLAC marker and a placeholder STREAMINFO block. total_samples and the
        // final block sizes are back-patched in finalize() once they are known.
        writer.write_all(&FLAC_MARKER)?;
        let stream_info = StreamInfo {
            min_block_size: block_size as u16,
            max_block_size: block_size as u16,
            min_frame_size: 0,
            max_frame_size: 0,
            sample_rate,
            channels: channels as u8,
            bits_per_sample,
            total_samples: 0,
            md5_signature: [0; 16],
        };
        let streaminfo_bytes = stream_info.to_bytes();
        writer.write_all(&[0x80])?; // Last metadata block, type 0 (STREAMINFO)
        writer.write_all(&[(streaminfo_bytes.len() >> 16) as u8])?;
        writer.write_all(&[(streaminfo_bytes.len() >> 8) as u8])?;
        writer.write_all(&[streaminfo_bytes.len() as u8])?;
        let streaminfo_offset = writer.stream_position()?;
        writer.write_all(&streaminfo_bytes)?;

        let accum = (0..channels as usize).map(|_| Vec::new()).collect();

        Ok(Self {
            writer,
            channels,
            sample_rate,
            bits_per_sample,
            block_size,
            params: FlacEncodeParams::from_level(level),
            accum,
            frame_number: 0,
            frames_written: 0,
            streaminfo_offset,
            finalized: false,
        })
    }
}

/// FLAC output bit depth for a configured sample type, matching `write_flac`'s rule.
const fn flac_bits_for(sample_type: ValidatedSampleType) -> u8 {
    match sample_type {
        ValidatedSampleType::I16 => 16,
        _ => 24,
    }
}

/// Encode `accum[..][start..start + len]` as one FLAC frame and write it to `writer`.
///
/// Takes `writer` and `accum` as separate borrows so the caller can pass disjoint fields
/// of `self`.
fn encode_and_write_block<W: Write>(
    writer: &mut W,
    accum: &[Vec<i32>],
    start: usize,
    len: usize,
    bits_per_sample: u8,
    sample_rate: u32,
    frame_number: u64,
    params: &FlacEncodeParams,
) -> AudioIOResult<()> {
    let ch_slices: Vec<&[i32]> = accum.iter().map(|ch| &ch[start..start + len]).collect();
    let frame_bytes = encode_frame_from_channels(
        &ch_slices,
        bits_per_sample,
        sample_rate,
        frame_number,
        params.max_lpc_order,
        params.qlp_precision,
        params.min_partition_order,
        params.max_partition_order,
        params.try_mid_side,
        params.exhaustive_rice,
    )
    .map_err(AudioIOError::FlacError)?;
    writer.write_all(&frame_bytes)?;
    Ok(())
}

impl<W> AudioStreamWriter for StreamedFlacWriter<W>
where
    W: WriteSeek,
{
    fn flush(&mut self) -> AudioIOResult<()> {
        // Flush already-encoded frame bytes; the partial block stays buffered until
        // finalize() so the stream remains a valid fixed-blocksize FLAC stream.
        self.writer.flush()?;
        Ok(())
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        if self.finalized {
            return Ok(()); // Idempotent
        }

        // Encode any buffered trailing (partial) block as the final frame.
        let remaining = self.accum.first().map(|c| c.len()).unwrap_or(0);
        if remaining > 0 {
            encode_and_write_block(
                &mut self.writer,
                &self.accum,
                0,
                remaining,
                self.bits_per_sample,
                self.sample_rate,
                self.frame_number,
                &self.params,
            )?;
            self.frame_number += 1;
            for ch in self.accum.iter_mut() {
                ch.clear();
            }
        }

        // Back-patch STREAMINFO with the final sample count and block sizes. For a stream
        // shorter than one block, clamp the reported block size to the sample count, just
        // as the bulk encoder does.
        let total = self.frames_written as u64;
        let final_block = if self.frames_written == 0 {
            self.block_size
        } else {
            self.block_size.min(self.frames_written as u32)
        };
        let stream_info = StreamInfo {
            min_block_size: final_block as u16,
            max_block_size: final_block as u16,
            min_frame_size: 0,
            max_frame_size: 0,
            sample_rate: self.sample_rate,
            channels: self.channels as u8,
            bits_per_sample: self.bits_per_sample,
            total_samples: total,
            md5_signature: [0; 16],
        };
        let streaminfo_bytes = stream_info.to_bytes();

        let end = self.writer.stream_position()?;
        self.writer.seek(SeekFrom::Start(self.streaminfo_offset))?;
        self.writer.write_all(&streaminfo_bytes)?;
        self.writer.seek(SeekFrom::Start(end))?;
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

impl<W> AudioStreamWrite for StreamedFlacWriter<W>
where
    W: WriteSeek,
{
    fn write_frames<T>(&mut self, samples: &AudioSamples<'_, T>) -> AudioIOResult<usize>
    where
        T: StandardSample + 'static,
    {
        if self.finalized {
            return Err(AudioIOError::corrupted_data_simple(
                "Cannot write to finalized stream",
                "Call write_frames before finalize()",
            ));
        }

        // Validate channel count.
        let input_channels = samples.num_channels();
        if input_channels.get() != self.channels as u32 {
            return Err(AudioIOError::corrupted_data_simple(
                "Channel count mismatch",
                format!(
                    "Writer configured for {} channels, got {} channels",
                    self.channels, input_channels
                ),
            ));
        }

        // Validate bit depth: every chunk must map to the writer's configured depth.
        let chunk_bits: u8 = match T::SAMPLE_TYPE {
            SampleType::I16 => 16,
            _ => 24,
        };
        if chunk_bits != self.bits_per_sample {
            return Err(AudioIOError::corrupted_data_simple(
                "Sample bit-depth mismatch",
                format!(
                    "Writer configured for {}-bit FLAC, but input maps to {}-bit",
                    self.bits_per_sample, chunk_bits
                ),
            ));
        }

        let frames_per_channel = samples.samples_per_channel().get();
        let planar = audio_to_planar_i32(samples)?;
        for (ch, data) in planar.iter().enumerate() {
            self.accum[ch].extend_from_slice(data);
        }

        // Emit every complete block now; keep the remainder buffered.
        let bs = self.block_size as usize;
        let available = self.accum.first().map(|c| c.len()).unwrap_or(0);
        if bs > 0 && available >= bs {
            let full_blocks = available / bs;
            for b in 0..full_blocks {
                encode_and_write_block(
                    &mut self.writer,
                    &self.accum,
                    b * bs,
                    bs,
                    self.bits_per_sample,
                    self.sample_rate,
                    self.frame_number,
                    &self.params,
                )?;
                self.frame_number += 1;
            }
            let consumed = full_blocks * bs;
            for ch in self.accum.iter_mut() {
                ch.drain(0..consumed);
            }
        }

        self.frames_written += frames_per_channel;
        Ok(frames_per_channel)
    }
}

/// Best-effort finalization on drop.
///
/// Unlike the WAV writer, FLAC buffers a partial block in memory, so dropping without
/// finalizing would lose those samples *and* leave STREAMINFO's sample count at zero. We
/// therefore finalize on drop, swallowing errors (a destructor cannot report them) and
/// warning in debug builds. Prefer calling `finalize()` explicitly so I/O errors surface.
impl<W> Drop for StreamedFlacWriter<W>
where
    W: WriteSeek,
{
    fn drop(&mut self) {
        if !self.finalized {
            #[cfg(debug_assertions)]
            if self.frames_written > 0 {
                eprintln!(
                    "Warning: StreamedFlacWriter dropped without calling finalize(); \
                     finalizing now. Call finalize() explicitly to surface I/O errors."
                );
            }
            let _ = self.finalize();
        }
    }
}
