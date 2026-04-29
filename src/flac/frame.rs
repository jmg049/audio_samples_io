//! FLAC frame encoding and decoding.
//!
//! A FLAC frame consists of:
//! - Frame header (sync code, parameters, CRC-8)
//! - One subframe per channel
//! - Padding to byte boundary
//! - Frame footer (CRC-16)

use crate::flac::ChannelAssignment;
use crate::flac::bitstream::{BitReader, BitWriter};
use crate::flac::constants::{
    BITS_PER_SAMPLE_TABLE, BLOCK_SIZE_TABLE, FRAME_SYNC_CODE, SAMPLE_RATE_TABLE,
    bits_per_sample_to_code, block_size_to_code, sample_rate_to_code,
};
use crate::flac::crc::{Crc8, Crc16};
use crate::flac::error::FlacError;
use crate::flac::subframe::{Subframe, decode_subframe, decode_subframe_into};
#[cfg(not(feature = "flac-parallel"))]
use crate::flac::subframe::encode_subframe_into;
#[cfg(feature = "flac-parallel")]
use crate::flac::subframe::{EncodedSubframe, encode_subframe};

/// Decoded frame header.
#[derive(Debug, Clone, Copy)]
pub struct FrameHeader {
    /// Blocking strategy (false = fixed-blocksize, true = variable-blocksize)
    pub variable_blocksize: bool,
    /// Block size in samples
    pub block_size: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Channel assignment
    pub channel_assignment: ChannelAssignment,
    /// Bits per sample (from STREAMINFO if 0)
    pub bits_per_sample: u8,
    /// Frame or sample number (depending on blocking strategy)
    pub frame_or_sample_number: u64,
}

/// A decoded FLAC frame.
#[derive(Debug)]
pub struct Frame {
    /// Frame header
    pub header: FrameHeader,
    /// Subframes (one per channel)
    pub subframes: Vec<Subframe>,
}

impl Frame {
    /// Consume the frame and return decoded samples per channel.
    ///
    /// For stereo decorrelation modes (left-side, right-side, mid-side),
    /// this undoes the decorrelation and returns independent channel data.
    ///
    /// Returns a Vec of channels, where each inner Vec contains the samples
    /// for that channel as i32 (FLAC's internal representation).
    pub fn into_channel_samples(self) -> Vec<Vec<i32>> {
        let block_size = self.subframes.first().map(|s| s.samples.len()).unwrap_or(0);

        match self.header.channel_assignment {
            ChannelAssignment::LeftSide => {
                // Left-side: subframe 0 = left, subframe 1 = left - right (side)
                // Restore: right = left - side
                let mut iter = self.subframes.into_iter();
                let left = iter.next().map(|s| s.samples).unwrap_or_default();
                let side = iter.next().map(|s| s.samples).unwrap_or_default();

                let right: Vec<i32> = left.iter().zip(side.iter()).map(|(&l, &s)| l - s).collect();
                vec![left, right]
            }
            ChannelAssignment::RightSide => {
                // Right-side: subframe 0 = left - right (side), subframe 1 = right
                // Restore: left = side + right
                let mut iter = self.subframes.into_iter();
                let side = iter.next().map(|s| s.samples).unwrap_or_default();
                let right = iter.next().map(|s| s.samples).unwrap_or_default();

                let left: Vec<i32> = side
                    .iter()
                    .zip(right.iter())
                    .map(|(&s, &r)| s + r)
                    .collect();
                vec![left, right]
            }
            ChannelAssignment::MidSide => {
                // Mid-side: subframe 0 = mid = (left + right) >> 1, subframe 1 = side = left - right
                // Restore: left = mid + (side + 1) >> 1, right = mid - (side >> 1)
                // More precisely: left = mid + (side >> 1) + (side & 1), right = left - side
                let mut iter = self.subframes.into_iter();
                let mid = iter.next().map(|s| s.samples).unwrap_or_default();
                let side = iter.next().map(|s| s.samples).unwrap_or_default();

                let mut left = Vec::with_capacity(block_size);
                let mut right = Vec::with_capacity(block_size);

                for (&m, &s) in mid.iter().zip(side.iter()) {
                    let l = m + (s >> 1) + (s & 1);
                    let r = l - s;
                    left.push(l);
                    right.push(r);
                }

                vec![left, right]
            }
            ChannelAssignment::Independent(_) => {
                // Independent channels: return as-is
                self.subframes.into_iter().map(|sf| sf.samples).collect()
            }
        }
    }

    /// Get the number of samples per channel in this frame.
    pub fn block_size(&self) -> usize {
        self.subframes.first().map(|s| s.samples.len()).unwrap_or(0)
    }

    /// Get the number of channels in this frame.
    pub const fn num_channels(&self) -> usize {
        self.subframes.len()
    }
}

/// Decode a FLAC frame from the bitstream.
///
/// # Arguments
/// * `data` - Byte slice containing the frame
/// * `streaminfo_sample_rate` - Sample rate from STREAMINFO (fallback)
/// * `streaminfo_bits_per_sample` - Bits per sample from STREAMINFO (fallback)
/// * `streaminfo_channels` - Channel count from STREAMINFO (fallback)
///
/// # Returns
/// Tuple of (decoded frame, bytes consumed)
pub fn decode_frame(
    data: &[u8],
    streaminfo_sample_rate: u32,
    streaminfo_bits_per_sample: u8,
    _streaminfo_channels: u8,
) -> Result<(Frame, usize), FlacError> {
    if data.len() < 6 {
        return Err(FlacError::UnexpectedEof);
    }

    let mut reader = BitReader::new(data);

    // Sync code (14 bits) + reserved (1 bit) + blocking strategy (1 bit)
    let sync = reader.read_bits(14)? as u16;
    if sync != FRAME_SYNC_CODE {
        return Err(FlacError::invalid_frame_sync(sync));
    }

    let reserved = reader.read_bit()?;
    if reserved {
        return Err(FlacError::InvalidFrameSync { found: 0xFFFF });
    }

    let variable_blocksize = reader.read_bit()?;

    // Block size code (4 bits) + sample rate code (4 bits)
    let block_size_code = reader.read_bits(4)? as u8;
    let sample_rate_code = reader.read_bits(4)? as u8;

    // Channel assignment (4 bits) + sample size code (3 bits) + reserved (1 bit)
    let channel_code = reader.read_bits(4)? as u8;
    let sample_size_code = reader.read_bits(3)? as u8;
    let reserved2 = reader.read_bit()?;
    if reserved2 {
        return Err(FlacError::ReservedBitsPerSampleCode);
    }

    // Parse channel assignment
    let channel_assignment = ChannelAssignment::from_code(channel_code)
        .ok_or(FlacError::InvalidChannelAssignment(channel_code))?;

    // Parse sample size
    let bits_per_sample = if sample_size_code == 0 {
        streaminfo_bits_per_sample
    } else if sample_size_code == 0b011 {
        return Err(FlacError::ReservedBitsPerSampleCode);
    } else {
        BITS_PER_SAMPLE_TABLE[sample_size_code as usize]
    };

    // Read frame/sample number (UTF-8 coded)
    let frame_or_sample_number = reader.read_utf8_coded()?;
    // Add UTF-8 bytes to CRC (we need the raw bytes)
    // For now, we'll recalculate from the current position
    // This is a simplification; proper implementation would track bytes

    // Parse block size (may need extra bytes)
    let block_size = match block_size_code {
        0b0000 => return Err(FlacError::ReservedBlockSizeCode),
        0b0110 => reader.read_bits(8)? + 1,
        0b0111 => reader.read_bits(16)? + 1,
        _ => BLOCK_SIZE_TABLE[block_size_code as usize],
    };

    // Parse sample rate (may need extra bytes)
    let sample_rate = match sample_rate_code {
        0b0000 => streaminfo_sample_rate,
        0b1100 => reader.read_bits(8)? * 1000,
        0b1101 => reader.read_bits(16)?,
        0b1110 => reader.read_bits(16)? * 10,
        0b1111 => return Err(FlacError::ReservedSampleRateCode),
        _ => SAMPLE_RATE_TABLE[sample_rate_code as usize],
    };

    // Verify CRC-8 over all header bytes preceding this byte
    let crc8_header_len = reader.byte_position();
    let crc8_expected = reader.read_bits(8)? as u8;
    let crc8_computed = Crc8::compute(&data[..crc8_header_len]);
    if crc8_computed != crc8_expected {
        return Err(FlacError::FrameHeaderCrcMismatch { expected: crc8_expected, computed: crc8_computed });
    }

    let header = FrameHeader {
        variable_blocksize,
        block_size,
        sample_rate,
        channel_assignment,
        bits_per_sample,
        frame_or_sample_number,
    };

    // Decode subframes
    let num_channels = channel_assignment.channels() as usize;
    let mut subframes = Vec::with_capacity(num_channels);

    for ch in 0..num_channels {
        // Adjust bits per sample for stereo decorrelation
        let subframe_bps = match (channel_assignment, ch) {
            (ChannelAssignment::MidSide, 1)|(ChannelAssignment::RightSide, 0)| (ChannelAssignment::LeftSide, 1) => bits_per_sample + 1,
            _ => bits_per_sample,
        };

        let subframe = decode_subframe(&mut reader, block_size as usize, subframe_bps)?;
        subframes.push(subframe);
    }

    // Align to byte boundary (single frame-level padding before CRC-16)
    reader.align_to_byte();

    // Verify CRC-16 over all frame bytes preceding the footer
    let crc16_frame_len = reader.byte_position();
    let crc16_expected = reader.read_bits(16)? as u16;
    let bytes_consumed = reader.byte_position();
    let crc16_computed = Crc16::compute(&data[..crc16_frame_len]);
    if crc16_computed != crc16_expected {
        return Err(FlacError::FrameCrcMismatch { expected: crc16_expected, computed: crc16_computed });
    }

    Ok((Frame { header, subframes }, bytes_consumed))
}

/// Decode a FLAC frame, writing samples directly into pre-allocated channel buffers.
///
/// Eliminates the intermediate `Frame` / `Subframe` structs and all per-subframe
/// `Vec<i32>` allocations. For independent channels, samples are decoded straight
/// into `out_channels[ch]`. For stereo decorrelation modes (left-side, right-side,
/// mid-side), two scratch buffers are used and the decorrelation is applied while
/// appending to `out_channels`.
///
/// Returns the number of bytes consumed from `data`.
pub(crate) fn decode_frame_into_channels(
    data: &[u8],
    streaminfo_sample_rate: u32,
    streaminfo_bits_per_sample: u8,
    _streaminfo_channels: u8,
    out_channels: &mut [Vec<i32>],
    scratch: &mut (Vec<i32>, Vec<i32>),
) -> Result<usize, FlacError> {
    if data.len() < 6 {
        return Err(FlacError::UnexpectedEof);
    }

    let mut reader = BitReader::new(data);

    let sync = reader.read_bits(14)? as u16;
    if sync != FRAME_SYNC_CODE {
        return Err(FlacError::invalid_frame_sync(sync));
    }
    let reserved = reader.read_bit()?;
    if reserved {
        return Err(FlacError::InvalidFrameSync { found: 0xFFFF });
    }
    let _variable_blocksize = reader.read_bit()?;

    let block_size_code = reader.read_bits(4)? as u8;
    let sample_rate_code = reader.read_bits(4)? as u8;
    let channel_code = reader.read_bits(4)? as u8;
    let sample_size_code = reader.read_bits(3)? as u8;
    let reserved2 = reader.read_bit()?;
    if reserved2 {
        return Err(FlacError::ReservedBitsPerSampleCode);
    }

    let channel_assignment = ChannelAssignment::from_code(channel_code)
        .ok_or(FlacError::InvalidChannelAssignment(channel_code))?;

    let bits_per_sample = if sample_size_code == 0 {
        streaminfo_bits_per_sample
    } else if sample_size_code == 0b011 {
        return Err(FlacError::ReservedBitsPerSampleCode);
    } else {
        BITS_PER_SAMPLE_TABLE[sample_size_code as usize]
    };

    let _frame_or_sample_number = reader.read_utf8_coded()?;

    let block_size = match block_size_code {
        0b0000 => return Err(FlacError::ReservedBlockSizeCode),
        0b0110 => reader.read_bits(8)? + 1,
        0b0111 => reader.read_bits(16)? + 1,
        _ => BLOCK_SIZE_TABLE[block_size_code as usize],
    } as usize;

    let _sample_rate = match sample_rate_code {
        0b0000 => streaminfo_sample_rate,
        0b1100 => reader.read_bits(8)? * 1000,
        0b1101 => reader.read_bits(16)?,
        0b1110 => reader.read_bits(16)? * 10,
        0b1111 => return Err(FlacError::ReservedSampleRateCode),
        _ => SAMPLE_RATE_TABLE[sample_rate_code as usize],
    };

    let crc8_header_len = reader.byte_position();
    let crc8_expected = reader.read_bits(8)? as u8;
    let crc8_computed = Crc8::compute(&data[..crc8_header_len]);
    if crc8_computed != crc8_expected {
        return Err(FlacError::FrameHeaderCrcMismatch {
            expected: crc8_expected,
            computed: crc8_computed,
        });
    }

    let num_channels = channel_assignment.channels() as usize;

    match channel_assignment {
        ChannelAssignment::Independent(_) => {
            // Decode directly into the output buffers — zero scratch needed.
            for ch in out_channels.iter_mut().take(num_channels) {
                decode_subframe_into(&mut reader, block_size, bits_per_sample, ch)?;
            }
        }
        ChannelAssignment::LeftSide => {
            // subframe 0 = left (normal bps), subframe 1 = side (bps+1)
            // right = left - side; transform side → right in-place, then bulk copy.
            scratch.0.clear();
            scratch.1.clear();
            decode_subframe_into(&mut reader, block_size, bits_per_sample,     &mut scratch.0)?;
            decode_subframe_into(&mut reader, block_size, bits_per_sample + 1, &mut scratch.1)?;
            for (s, &l) in scratch.1.iter_mut().zip(scratch.0.iter()) {
                *s = l - *s;
            }
            let (a, b) = out_channels.split_at_mut(1);
            a[0].extend_from_slice(&scratch.0);
            b[0].extend_from_slice(&scratch.1);
        }
        ChannelAssignment::RightSide => {
            // subframe 0 = side (bps+1), subframe 1 = right (normal bps)
            // left = side + right; transform side → left in-place, then bulk copy.
            scratch.0.clear();
            scratch.1.clear();
            decode_subframe_into(&mut reader, block_size, bits_per_sample + 1, &mut scratch.0)?;
            decode_subframe_into(&mut reader, block_size, bits_per_sample,     &mut scratch.1)?;
            for (s, &r) in scratch.0.iter_mut().zip(scratch.1.iter()) {
                *s += r;
            }
            let (a, b) = out_channels.split_at_mut(1);
            a[0].extend_from_slice(&scratch.0);
            b[0].extend_from_slice(&scratch.1);
        }
        ChannelAssignment::MidSide => {
            // subframe 0 = mid (normal bps), subframe 1 = side (bps+1)
            // Decorrelate in-place on scratch (L1-hot), then bulk-copy (SIMD memcpy).
            scratch.0.clear();
            scratch.1.clear();
            decode_subframe_into(&mut reader, block_size, bits_per_sample,     &mut scratch.0)?;
            decode_subframe_into(&mut reader, block_size, bits_per_sample + 1, &mut scratch.1)?;
            for (m, s) in scratch.0.iter_mut().zip(scratch.1.iter_mut()) {
                let mid = (*m << 1) | (*s & 1);
                let side = *s;
                *m = (mid + side) >> 1;
                *s = (mid - side) >> 1;
            }
            let (a, b) = out_channels.split_at_mut(1);
            a[0].extend_from_slice(&scratch.0);
            b[0].extend_from_slice(&scratch.1);
        }
    }

    reader.align_to_byte();
    let crc16_frame_len = reader.byte_position();
    let crc16_expected = reader.read_bits(16)? as u16;
    let bytes_consumed = reader.byte_position();
    let crc16_computed = Crc16::compute(&data[..crc16_frame_len]);
    if crc16_computed != crc16_expected {
        return Err(FlacError::FrameCrcMismatch {
            expected: crc16_expected,
            computed: crc16_computed,
        });
    }

    Ok(bytes_consumed)
}

/// Decode one FLAC frame into per-channel scratch buffers (overwrite mode).
///
/// Unlike `decode_frame_into_channels`, this function:
/// - Clears `scratch[ch]` before writing (no accumulation across frames)
/// - Uses `scratch[0]`/`scratch[1]` directly for stereo decorrelation
///
/// On return, `scratch[ch]` holds exactly `block_size` decoded i32 samples.
/// The caller converts scratch → typed output while it is L1/L2-hot.
pub(crate) fn decode_frame_into_scratch(
    data: &[u8],
    streaminfo_sample_rate: u32,
    streaminfo_bits_per_sample: u8,
    scratch: &mut [Vec<i32>],
) -> Result<usize, FlacError> {
    if data.len() < 6 {
        return Err(FlacError::UnexpectedEof);
    }

    let mut reader = BitReader::new(data);

    let sync = reader.read_bits(14)? as u16;
    if sync != FRAME_SYNC_CODE {
        return Err(FlacError::invalid_frame_sync(sync));
    }
    let reserved = reader.read_bit()?;
    if reserved {
        return Err(FlacError::InvalidFrameSync { found: 0xFFFF });
    }
    let _variable_blocksize = reader.read_bit()?;

    let block_size_code = reader.read_bits(4)? as u8;
    let sample_rate_code = reader.read_bits(4)? as u8;
    let channel_code = reader.read_bits(4)? as u8;
    let sample_size_code = reader.read_bits(3)? as u8;
    let reserved2 = reader.read_bit()?;
    if reserved2 {
        return Err(FlacError::ReservedBitsPerSampleCode);
    }

    let channel_assignment = ChannelAssignment::from_code(channel_code)
        .ok_or(FlacError::InvalidChannelAssignment(channel_code))?;

    let bits_per_sample = if sample_size_code == 0 {
        streaminfo_bits_per_sample
    } else if sample_size_code == 0b011 {
        return Err(FlacError::ReservedBitsPerSampleCode);
    } else {
        BITS_PER_SAMPLE_TABLE[sample_size_code as usize]
    };

    let _frame_or_sample_number = reader.read_utf8_coded()?;

    let block_size = match block_size_code {
        0b0000 => return Err(FlacError::ReservedBlockSizeCode),
        0b0110 => reader.read_bits(8)? + 1,
        0b0111 => reader.read_bits(16)? + 1,
        _ => BLOCK_SIZE_TABLE[block_size_code as usize],
    } as usize;

    let _sample_rate = match sample_rate_code {
        0b0000 => streaminfo_sample_rate,
        0b1100 => reader.read_bits(8)? * 1000,
        0b1101 => reader.read_bits(16)?,
        0b1110 => reader.read_bits(16)? * 10,
        0b1111 => return Err(FlacError::ReservedSampleRateCode),
        _ => SAMPLE_RATE_TABLE[sample_rate_code as usize],
    };

    let crc8_header_len = reader.byte_position();
    let crc8_expected = reader.read_bits(8)? as u8;
    let crc8_computed = Crc8::compute(&data[..crc8_header_len]);
    if crc8_computed != crc8_expected {
        return Err(FlacError::FrameHeaderCrcMismatch {
            expected: crc8_expected,
            computed: crc8_computed,
        });
    }

    let num_channels = channel_assignment.channels() as usize;

    // Clear scratch for this frame (overwrite mode).
    for ch in scratch[..num_channels].iter_mut() {
        ch.clear();
    }

    match channel_assignment {
        ChannelAssignment::Independent(_) => {
            for ch in scratch.iter_mut().take(num_channels) {
                decode_subframe_into(&mut reader, block_size, bits_per_sample, ch)?;
            }
        }
        ChannelAssignment::LeftSide => {
            // scratch[0] = left, scratch[1] = side → decorate in-place to right
            decode_subframe_into(&mut reader, block_size, bits_per_sample,     &mut scratch[0])?;
            decode_subframe_into(&mut reader, block_size, bits_per_sample + 1, &mut scratch[1])?;
            let (left, rest) = scratch.split_at_mut(1);
            for (s, &l) in rest[0].iter_mut().zip(left[0].iter()) {
                *s = l - *s;
            }
        }
        ChannelAssignment::RightSide => {
            // scratch[0] = side → left, scratch[1] = right
            decode_subframe_into(&mut reader, block_size, bits_per_sample + 1, &mut scratch[0])?;
            decode_subframe_into(&mut reader, block_size, bits_per_sample,     &mut scratch[1])?;
            let (side, rest) = scratch.split_at_mut(1);
            for (s, &r) in side[0].iter_mut().zip(rest[0].iter()) {
                *s += r;
            }
        }
        ChannelAssignment::MidSide => {
            decode_subframe_into(&mut reader, block_size, bits_per_sample,     &mut scratch[0])?;
            decode_subframe_into(&mut reader, block_size, bits_per_sample + 1, &mut scratch[1])?;
            let (mid_slice, side_slice) = scratch.split_at_mut(1);
            for (m, s) in mid_slice[0].iter_mut().zip(side_slice[0].iter_mut()) {
                let mid = (*m << 1) | (*s & 1);
                let side = *s;
                *m = (mid + side) >> 1;
                *s = (mid - side) >> 1;
            }
        }
    }

    reader.align_to_byte();
    let crc16_frame_len = reader.byte_position();
    let crc16_expected = reader.read_bits(16)? as u16;
    let bytes_consumed = reader.byte_position();
    let crc16_computed = Crc16::compute(&data[..crc16_frame_len]);
    if crc16_computed != crc16_expected {
        return Err(FlacError::FrameCrcMismatch {
            expected: crc16_expected,
            computed: crc16_computed,
        });
    }

    Ok(bytes_consumed)
}

/// Encode samples into a FLAC frame.
///
/// # Arguments
/// * `samples` - Interleaved samples
/// * `channels` - Number of channels
/// * `block_size` - Number of samples per channel
/// * `sample_rate` - Sample rate in Hz
/// * `bits_per_sample` - Bits per sample
/// * `frame_number` - Frame number (for fixed-blocksize)
/// * `max_lpc_order` - Maximum LPC order (0 = fixed only)
/// * `qlp_precision` - QLP coefficient precision
/// * `min_partition_order` - Minimum Rice partition order
/// * `max_partition_order` - Maximum Rice partition order
/// * `try_mid_side` - Whether to try mid-side stereo
/// * `exhaustive_rice` - Use exhaustive Rice parameter search
pub fn encode_frame(
    samples: &[i32],
    channels: u8,
    block_size: u32,
    sample_rate: u32,
    bits_per_sample: u8,
    frame_number: u64,
    max_lpc_order: usize,
    qlp_precision: u8,
    min_partition_order: u8,
    max_partition_order: u8,
    try_mid_side: bool,
    exhaustive_rice: bool,
) -> Result<Vec<u8>, FlacError> {
    let num_channels = channels as usize;
    let samples_per_channel = block_size as usize;

    // Deinterleave samples
    let mut ch_vecs: Vec<Vec<i32>> = vec![Vec::with_capacity(samples_per_channel); num_channels];
    for (i, &sample) in samples.iter().enumerate() {
        ch_vecs[i % num_channels].push(sample);
    }

    let ch_slices: Vec<&[i32]> = ch_vecs.iter().map(|v| v.as_slice()).collect();
    encode_frame_from_channels(
        &ch_slices,
        bits_per_sample,
        sample_rate,
        frame_number,
        max_lpc_order,
        qlp_precision,
        min_partition_order,
        max_partition_order,
        try_mid_side,
        exhaustive_rice,
    )
}

/// Encode frame header.
fn encode_frame_header(
    writer: &mut BitWriter,
    block_size: u32,
    sample_rate: u32,
    channel_assignment: ChannelAssignment,
    bits_per_sample: u8,
    frame_number: u64,
) -> Result<(), FlacError> {
    // Sync code (14 bits)
    writer.write_bits(FRAME_SYNC_CODE as u32, 14);

    // Reserved (1 bit) = 0
    writer.write_bit(false);

    // Blocking strategy (1 bit) - using fixed for now
    writer.write_bit(false);

    // Block size code
    let (bs_code, bs_extra) = block_size_to_code(block_size);
    writer.write_bits(bs_code as u32, 4);

    // Sample rate code
    let (sr_code, sr_extra) = sample_rate_to_code(sample_rate);
    writer.write_bits(sr_code as u32, 4);

    // Channel assignment
    writer.write_bits(channel_assignment.code() as u32, 4);

    // Sample size
    let bps_code = bits_per_sample_to_code(bits_per_sample).unwrap_or(0);
    writer.write_bits(bps_code as u32, 3);

    // Reserved (1 bit) = 0
    writer.write_bit(false);

    // Frame number (UTF-8 coded)
    writer.write_utf8_coded(frame_number);

    // Extra block size bytes
    match bs_extra {
        1 => writer.write_bits((block_size - 1) as u32, 8),
        2 => writer.write_bits((block_size - 1) as u32, 16),
        _ => {}
    }

    // Extra sample rate bytes
    match sr_extra {
        1 => writer.write_bits(sample_rate / 1000, 8),
        2 => writer.write_bits(sample_rate, 16),
        3 => writer.write_bits(sample_rate / 10, 16),
        _ => {}
    }

    Ok(())
}

/// Find the best stereo encoding mode using residual energy estimation.
///
/// Rather than fully encoding all four candidate modes (8 subframe encodes),
/// this uses L1 residual energy as a proxy for encoding cost. The mode with
/// the lowest estimated energy is selected and only that mode's channel data
/// is returned. The side-channel modes incur a +1 bit-per-sample overhead
/// which is approximated as a penalty of `block_size` to the energy score.
fn find_best_stereo_mode(
    left: &[i32],
    right: &[i32],
) -> (ChannelAssignment, Vec<Vec<i32>>) {
    let n = left.len() as u64;

    let side: Vec<i32> = left.iter().zip(right.iter()).map(|(&l, &r)| l - r).collect();
    let mid: Vec<i32> = left.iter().zip(right.iter()).map(|(&l, &r)| (l + r) >> 1).collect();

    let e_left: u64 = left.iter().map(|&s| (s as i64).unsigned_abs()).sum();
    let e_right: u64 = right.iter().map(|&s| (s as i64).unsigned_abs()).sum();
    let e_side: u64 = side.iter().map(|&s| (s as i64).unsigned_abs()).sum();
    let e_mid: u64 = mid.iter().map(|&s| (s as i64).unsigned_abs()).sum();

    // Side-channel modes require 1 extra bit/sample — model as +n penalty
    let score_ind = e_left + e_right;
    let score_ls  = e_left + e_side + n;
    let score_rs  = e_side + e_right + n;
    let score_ms  = e_mid  + e_side + n;

    if score_ind <= score_ls && score_ind <= score_rs && score_ind <= score_ms {
        (ChannelAssignment::Independent(2), vec![left.to_vec(), right.to_vec()])
    } else if score_ls <= score_rs && score_ls <= score_ms {
        (ChannelAssignment::LeftSide, vec![left.to_vec(), side])
    } else if score_rs <= score_ms {
        (ChannelAssignment::RightSide, vec![side, right.to_vec()])
    } else {
        (ChannelAssignment::MidSide, vec![mid, side])
    }
}

/// Encode samples into a FLAC frame from already-deinterleaved channel slices.
///
/// This avoids the interleave→deinterleave round-trip of `encode_frame`.
/// For independent channels the input slices are encoded directly without
/// allocating intermediate owned Vecs.
pub fn encode_frame_from_channels(
    channel_samples: &[&[i32]],
    bits_per_sample: u8,
    sample_rate: u32,
    frame_number: u64,
    max_lpc_order: usize,
    qlp_precision: u8,
    min_partition_order: u8,
    max_partition_order: u8,
    try_mid_side: bool,
    exhaustive_rice: bool,
) -> Result<Vec<u8>, FlacError> {
    let num_channels = channel_samples.len() as u8;
    if channel_samples.is_empty() {
        return Err(FlacError::InvalidChannelCount { channels: 0 });
    }
    let block_size = channel_samples[0].len() as u32;

    let mut writer =
        BitWriter::with_capacity(channel_samples[0].len() * num_channels as usize * 4);

    if num_channels == 2 && try_mid_side {
        let (channel_assignment, owned) =
            find_best_stereo_mode(channel_samples[0], channel_samples[1]);

        // Write header before subframe encoding so the sequential path can write
        // subframes directly into the frame writer without an intermediate buffer.
        encode_frame_header(&mut writer, block_size, sample_rate, channel_assignment,
            bits_per_sample, frame_number)?;
        writer.align_to_byte();
        let crc8 = Crc8::compute(writer.data());
        writer.write_bits(crc8 as u32, 8);

        #[cfg(feature = "flac-parallel")]
        {
            use rayon::prelude::*;
            let subframe_data: Vec<EncodedSubframe> = owned
                .par_iter()
                .enumerate()
                .map(|(ch, ch_data)| {
                    let subframe_bps = match (channel_assignment, ch) {
                        (ChannelAssignment::LeftSide, 1)
                        | (ChannelAssignment::RightSide, 0)
                        | (ChannelAssignment::MidSide, 1) => bits_per_sample + 1,
                        _ => bits_per_sample,
                    };
                    encode_subframe(ch_data, subframe_bps, max_lpc_order, qlp_precision,
                        min_partition_order, max_partition_order, exhaustive_rice)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for encoded in &subframe_data {
                writer.write_packed_bits(&encoded.data, encoded.bits);
            }
        }
        #[cfg(not(feature = "flac-parallel"))]
        for (ch, ch_data) in owned.iter().enumerate() {
            let subframe_bps = match (channel_assignment, ch) {
                (ChannelAssignment::LeftSide, 1)
                | (ChannelAssignment::RightSide, 0)
                | (ChannelAssignment::MidSide, 1) => bits_per_sample + 1,
                _ => bits_per_sample,
            };
            encode_subframe_into(&mut writer, ch_data, subframe_bps, max_lpc_order,
                qlp_precision, min_partition_order, max_partition_order, exhaustive_rice)?;
        }
    } else {
        let channel_assignment = ChannelAssignment::Independent(num_channels);

        encode_frame_header(&mut writer, block_size, sample_rate, channel_assignment,
            bits_per_sample, frame_number)?;
        writer.align_to_byte();
        let crc8 = Crc8::compute(writer.data());
        writer.write_bits(crc8 as u32, 8);

        #[cfg(feature = "flac-parallel")]
        {
            use rayon::prelude::*;
            let subframe_data: Vec<EncodedSubframe> = channel_samples
                .par_iter()
                .map(|&ch_data| {
                    encode_subframe(ch_data, bits_per_sample, max_lpc_order, qlp_precision,
                        min_partition_order, max_partition_order, exhaustive_rice)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for encoded in &subframe_data {
                writer.write_packed_bits(&encoded.data, encoded.bits);
            }
        }
        #[cfg(not(feature = "flac-parallel"))]
        for &ch_data in channel_samples {
            encode_subframe_into(&mut writer, ch_data, bits_per_sample, max_lpc_order,
                qlp_precision, min_partition_order, max_partition_order, exhaustive_rice)?;
        }
    }

    // Single frame-level padding to byte boundary before CRC-16
    writer.align_to_byte();

    let frame_data = writer.data();
    let crc16 = Crc16::compute(frame_data);
    writer.write_bits(crc16 as u32, 16);

    Ok(writer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_assignment_code_roundtrip() {
        let assignments = [
            ChannelAssignment::Independent(1),
            ChannelAssignment::Independent(2),
            ChannelAssignment::Independent(8),
            ChannelAssignment::LeftSide,
            ChannelAssignment::RightSide,
            ChannelAssignment::MidSide,
        ];

        for assignment in assignments {
            let code = assignment.code();
            let parsed = ChannelAssignment::from_code(code);
            assert!(
                parsed.is_some(),
                "Failed to parse code for {:?}",
                assignment
            );
            assert_eq!(parsed.unwrap().channels(), assignment.channels());
        }
    }

    #[test]
    fn test_stereo_decorrelation_roundtrip() {
        // Create test stereo data
        let left: Vec<i32> = (0..100).map(|i| (i * 100) as i32).collect();
        let right: Vec<i32> = (0..100).map(|i| (i * 50 + 25) as i32).collect();

        // Interleave
        let mut interleaved = Vec::with_capacity(200);
        for i in 0..100 {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }

        // Create a frame with independent channels
        let frame = Frame {
            header: FrameHeader {
                variable_blocksize: false,
                block_size: 100,
                sample_rate: 44100,
                channel_assignment: ChannelAssignment::Independent(2),
                bits_per_sample: 16,
                frame_or_sample_number: 0,
            },
            subframes: vec![
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: left.clone(),
                },
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: right.clone(),
                },
            ],
        };

        let decoded = frame.into_channel_samples();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], left);
        assert_eq!(decoded[1], right);
    }

    #[test]
    fn test_mid_side_decorrelation() {
        let left = vec![100i32, 200, 300];
        let right = vec![50i32, 150, 250];

        // Compute mid and side
        let mid: Vec<i32> = left
            .iter()
            .zip(&right)
            .map(|(&l, &r)| (l + r) >> 1)
            .collect();
        let side: Vec<i32> = left.iter().zip(&right).map(|(&l, &r)| l - r).collect();

        let frame = Frame {
            header: FrameHeader {
                variable_blocksize: false,
                block_size: 3,
                sample_rate: 44100,
                channel_assignment: ChannelAssignment::MidSide,
                bits_per_sample: 16,
                frame_or_sample_number: 0,
            },
            subframes: vec![
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: mid,
                },
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: side,
                },
            ],
        };

        let decoded = frame.into_channel_samples();

        // Should recover original left/right channels
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], left);
        assert_eq!(decoded[1], right);
    }

    // =========================================================================
    // encode_frame / decode_frame roundtrip tests
    // =========================================================================

    /// Encode a mono frame then decode it; verify sample count and block_size.
    #[test]
    fn test_encode_decode_frame_mono_constant() {
        // Use silence (all zeros) — will encode as CONSTANT subframe, which is lossless.
        let block_size = 512u32;
        let samples: Vec<i32> = vec![0i32; block_size as usize];

        let frame_bytes = encode_frame(
            &samples,
            1,         // mono
            block_size,
            44100,
            16,
            0,         // frame number
            0,         // max_lpc_order = 0 → fixed only
            12,        // qlp_precision
            0, 3,      // rice partition order range
            false,     // no mid-side
            false,     // no exhaustive rice
        )
        .expect("encode_frame should succeed");

        assert!(!frame_bytes.is_empty(), "encoded frame must not be empty");

        let (frame, bytes_consumed) =
            decode_frame(&frame_bytes, 44100, 16, 1).expect("decode_frame should succeed");

        assert_eq!(bytes_consumed, frame_bytes.len(), "all bytes consumed");
        assert_eq!(frame.block_size(), block_size as usize, "block size");
        assert_eq!(frame.num_channels(), 1, "channels");

        let channels = frame.into_channel_samples();
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].len(), block_size as usize);
        assert!(channels[0].iter().all(|&s| s == 0), "silence should decode to zeros");
    }

    /// Encode constant non-zero audio mono.
    #[test]
    fn test_encode_decode_frame_mono_dc() {
        let block_size = 1024u32;
        let dc_value = 1000i32;
        let samples: Vec<i32> = vec![dc_value; block_size as usize];

        let frame_bytes = encode_frame(
            &samples, 1, block_size, 44100, 16, 0, 0, 12, 0, 3, false, false,
        )
        .expect("encode");

        let (frame, _) = decode_frame(&frame_bytes, 44100, 16, 1).expect("decode");
        let channels = frame.into_channel_samples();
        assert_eq!(channels[0].len(), block_size as usize);
        // DC audio should be lossless via CONSTANT subframe
        assert!(
            channels[0].iter().all(|&s| s == dc_value),
            "DC signal should decode exactly"
        );
    }

    /// Encode a mono frame of block_size 4096.
    #[test]
    fn test_encode_decode_frame_large_block() {
        let block_size = 4096u32;
        let samples: Vec<i32> = vec![0i32; block_size as usize];

        let frame_bytes = encode_frame(
            &samples, 1, block_size, 48000, 16, 5, 0, 12, 0, 4, false, false,
        )
        .expect("encode large block");

        let (frame, consumed) = decode_frame(&frame_bytes, 48000, 16, 1).expect("decode large block");
        assert_eq!(consumed, frame_bytes.len());
        assert_eq!(frame.block_size(), block_size as usize);
    }

    /// Frame number is encoded correctly (UTF-8 coded in header).
    #[test]
    fn test_encode_decode_frame_number() {
        let block_size = 1024u32;
        let samples: Vec<i32> = vec![0i32; block_size as usize];

        for frame_number in [0u64, 1, 127, 128, 0x3FFF, 0x1FFFFF] {
            let frame_bytes = encode_frame(
                &samples, 1, block_size, 44100, 16, frame_number, 0, 12, 0, 3, false, false,
            )
            .unwrap_or_else(|e| panic!("encode frame {frame_number}: {e}"));

            let (frame, _) = decode_frame(&frame_bytes, 44100, 16, 1)
                .unwrap_or_else(|e| panic!("decode frame {frame_number}: {e}"));
            assert_eq!(
                frame.header.frame_or_sample_number, frame_number,
                "frame number should match"
            );
        }
    }

    /// Different sample rates survive the encode→decode header.
    #[test]
    fn test_encode_decode_various_sample_rates() {
        let block_size = 512u32;
        let samples: Vec<i32> = vec![0i32; block_size as usize];

        for sr in [44100u32, 48000, 96000] {
            let frame_bytes = encode_frame(
                &samples, 1, block_size, sr, 16, 0, 0, 12, 0, 3, false, false,
            )
            .unwrap_or_else(|e| panic!("encode sr={sr}: {e}"));

            let (frame, _) = decode_frame(&frame_bytes, sr, 16, 1)
                .unwrap_or_else(|e| panic!("decode sr={sr}: {e}"));

            // Sample rate is encoded in the frame header; verify it survives
            assert_eq!(frame.header.sample_rate, sr, "sample rate should be preserved");
        }
    }

    /// Left-side stereo decorrelation: encode stereo with left-side mode,
    /// decode and verify the two channels come back as independent.
    #[test]
    fn test_left_side_decorrelation_roundtrip() {
        // Simple ascending samples for predictable encoding
        let left: Vec<i32> = (0..100).map(|i| i as i32 * 10).collect();
        let right: Vec<i32> = (0..100).map(|i| i as i32 * 10 + 5).collect();

        // Manually compute left-side encoded form
        let side: Vec<i32> = left.iter().zip(&right).map(|(&l, &r)| l - r).collect();

        let frame = Frame {
            header: FrameHeader {
                variable_blocksize: false,
                block_size: 100,
                sample_rate: 44100,
                channel_assignment: ChannelAssignment::LeftSide,
                bits_per_sample: 16,
                frame_or_sample_number: 0,
            },
            subframes: vec![
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: left.clone(),
                },
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: side,
                },
            ],
        };

        let decoded = frame.into_channel_samples();
        assert_eq!(decoded[0], left, "left channel should match");
        assert_eq!(decoded[1], right, "right channel should be restored");
    }

    /// Right-side stereo decorrelation roundtrip.
    #[test]
    fn test_right_side_decorrelation_roundtrip() {
        let left: Vec<i32> = (0..50).map(|i| i as i32 * 20).collect();
        let right: Vec<i32> = (0..50).map(|i| i as i32 * 20 + 10).collect();
        let side: Vec<i32> = left.iter().zip(&right).map(|(&l, &r)| l - r).collect();

        let frame = Frame {
            header: FrameHeader {
                variable_blocksize: false,
                block_size: 50,
                sample_rate: 44100,
                channel_assignment: ChannelAssignment::RightSide,
                bits_per_sample: 16,
                frame_or_sample_number: 0,
            },
            subframes: vec![
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: side,
                },
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Verbatim,
                    wasted_bits: 0,
                    samples: right.clone(),
                },
            ],
        };

        let decoded = frame.into_channel_samples();
        assert_eq!(decoded[0], left, "left channel");
        assert_eq!(decoded[1], right, "right channel");
    }

    /// Frame block_size() and num_channels() accessors.
    #[test]
    fn test_frame_accessors() {
        let frame = Frame {
            header: FrameHeader {
                variable_blocksize: false,
                block_size: 256,
                sample_rate: 44100,
                channel_assignment: ChannelAssignment::Independent(3),
                bits_per_sample: 16,
                frame_or_sample_number: 42,
            },
            subframes: vec![
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Constant,
                    wasted_bits: 0,
                    samples: vec![0i32; 256],
                },
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Constant,
                    wasted_bits: 0,
                    samples: vec![0i32; 256],
                },
                Subframe {
                    subframe_type: crate::flac::subframe::SubframeType::Constant,
                    wasted_bits: 0,
                    samples: vec![0i32; 256],
                },
            ],
        };
        assert_eq!(frame.block_size(), 256);
        assert_eq!(frame.num_channels(), 3);
    }
}
