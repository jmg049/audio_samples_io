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
use crate::flac::subframe::{EncodedSubframe, Subframe, decode_subframe, encode_subframe};

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
    pub fn num_channels(&self) -> usize {
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
    streaminfo_channels: u8,
) -> Result<(Frame, usize), FlacError> {
    if data.len() < 6 {
        return Err(FlacError::UnexpectedEof);
    }

    // Track CRC-8 for header
    let mut crc8 = Crc8::new();

    let mut reader = BitReader::new(data);

    // Sync code (14 bits) + reserved (1 bit) + blocking strategy (1 bit)
    let sync = reader.read_bits(14)? as u16;
    if sync != FRAME_SYNC_CODE {
        return Err(FlacError::invalid_frame_sync(sync));
    }
    crc8.update(&[(sync >> 6) as u8]);

    let reserved = reader.read_bit()?;
    if reserved {
        return Err(FlacError::InvalidFrameSync { found: 0xFFFF });
    }

    let variable_blocksize = reader.read_bit()?;
    let first_byte = ((sync & 0x3F) << 2) as u8 | (variable_blocksize as u8);
    crc8.update(&[first_byte]);

    // Block size code (4 bits) + sample rate code (4 bits)
    let block_size_code = reader.read_bits(4)? as u8;
    let sample_rate_code = reader.read_bits(4)? as u8;
    crc8.update(&[(block_size_code << 4) | sample_rate_code]);

    // Channel assignment (4 bits) + sample size code (3 bits) + reserved (1 bit)
    let channel_code = reader.read_bits(4)? as u8;
    let sample_size_code = reader.read_bits(3)? as u8;
    let reserved2 = reader.read_bit()?;
    if reserved2 {
        return Err(FlacError::ReservedBitsPerSampleCode);
    }
    crc8.update(&[(channel_code << 4) | (sample_size_code << 1)]);

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
        0b0110 => (reader.read_bits(8)? + 1),
        0b0111 => (reader.read_bits(16)? + 1),
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

    // Read and verify CRC-8
    // Note: We need to compute CRC over all header bytes
    // For now, skip verification in this simplified implementation
    let _crc8_read = reader.read_bits(8)? as u8;
    // TODO: Proper CRC-8 verification

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
            (ChannelAssignment::LeftSide, 1) => bits_per_sample + 1, // side channel
            (ChannelAssignment::RightSide, 0) => bits_per_sample + 1, // side channel
            (ChannelAssignment::MidSide, 1) => bits_per_sample + 1,  // side channel
            _ => bits_per_sample,
        };

        let subframe = decode_subframe(&mut reader, block_size as usize, subframe_bps)?;
        subframes.push(subframe);
    }

    // Align to byte boundary
    reader.align_to_byte();

    // Read CRC-16 footer
    let _crc16_read = reader.read_bits(16)? as u16;
    // TODO: Proper CRC-16 verification

    let bytes_consumed = reader.byte_position();

    Ok((Frame { header, subframes }, bytes_consumed))
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
    let mut channel_samples: Vec<Vec<i32>> =
        vec![Vec::with_capacity(samples_per_channel); num_channels];
    for (i, &sample) in samples.iter().enumerate() {
        channel_samples[i % num_channels].push(sample);
    }

    // Determine best channel assignment for stereo
    let (channel_assignment, encoded_channels) = if num_channels == 2 && try_mid_side {
        find_best_stereo_mode(
            &channel_samples[0],
            &channel_samples[1],
            bits_per_sample,
            max_lpc_order,
            qlp_precision,
            min_partition_order,
            max_partition_order,
            exhaustive_rice,
        )?
    } else {
        (ChannelAssignment::Independent(channels), channel_samples)
    };

    // Encode each channel
    let mut subframe_data: Vec<EncodedSubframe> = Vec::with_capacity(num_channels);
    for (ch, ch_samples) in encoded_channels.iter().enumerate() {
        // Adjust bits per sample for side channel
        let subframe_bps = match (channel_assignment, ch) {
            (ChannelAssignment::LeftSide, 1)
            | (ChannelAssignment::RightSide, 0)
            | (ChannelAssignment::MidSide, 1) => bits_per_sample + 1,
            _ => bits_per_sample,
        };

        let encoded = encode_subframe(
            ch_samples,
            subframe_bps,
            max_lpc_order,
            qlp_precision,
            min_partition_order,
            max_partition_order,
            exhaustive_rice,
        )?;
        subframe_data.push(encoded);
    }

    // Build frame
    let mut writer = BitWriter::with_capacity(samples.len() * 4);

    // Encode header
    encode_frame_header(
        &mut writer,
        block_size,
        sample_rate,
        channel_assignment,
        bits_per_sample,
        frame_number,
    )?;

    // Get header bytes for CRC-8
    let header_data = writer.data().to_vec();
    let crc8 = Crc8::compute(&header_data);
    writer.write_bits(crc8 as u32, 8);

    // Write subframes
    for encoded in &subframe_data {
        // Write raw subframe bits
        for &byte in &encoded.data {
            writer.write_bits(byte as u32, 8);
        }
    }

    // Align to byte boundary
    writer.align_to_byte();

    // Compute CRC-16 over entire frame (excluding footer)
    let frame_data = writer.data();
    let crc16 = Crc16::compute(frame_data);
    writer.write_bits(crc16 as u32, 16);

    Ok(writer.finish())
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

/// Find the best stereo encoding mode.
fn find_best_stereo_mode(
    left: &[i32],
    right: &[i32],
    bits_per_sample: u8,
    max_lpc_order: usize,
    qlp_precision: u8,
    min_partition_order: u8,
    max_partition_order: u8,
    exhaustive_rice: bool,
) -> Result<(ChannelAssignment, Vec<Vec<i32>>), FlacError> {
    let block_size = left.len();

    // Compute side channel
    let side: Vec<i32> = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| l - r)
        .collect();

    // Compute mid channel: (left + right) >> 1
    let mid: Vec<i32> = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| (l + r) >> 1)
        .collect();

    // Estimate bits for each mode
    let modes = [
        (
            ChannelAssignment::Independent(2),
            vec![left.to_vec(), right.to_vec()],
            bits_per_sample,
            bits_per_sample,
        ),
        (
            ChannelAssignment::LeftSide,
            vec![left.to_vec(), side.clone()],
            bits_per_sample,
            bits_per_sample + 1,
        ),
        (
            ChannelAssignment::RightSide,
            vec![side.clone(), right.to_vec()],
            bits_per_sample + 1,
            bits_per_sample,
        ),
        (
            ChannelAssignment::MidSide,
            vec![mid, side],
            bits_per_sample,
            bits_per_sample + 1,
        ),
    ];

    let mut best_assignment = ChannelAssignment::Independent(2);
    let mut best_channels = vec![left.to_vec(), right.to_vec()];
    let mut best_bits = usize::MAX;

    for (assignment, channels, bps0, bps1) in modes {
        let encoded0 = encode_subframe(
            &channels[0],
            bps0,
            max_lpc_order,
            qlp_precision,
            min_partition_order,
            max_partition_order,
            exhaustive_rice,
        )?;
        let encoded1 = encode_subframe(
            &channels[1],
            bps1,
            max_lpc_order,
            qlp_precision,
            min_partition_order,
            max_partition_order,
            exhaustive_rice,
        )?;

        let total_bits = encoded0.bits + encoded1.bits;
        if total_bits < best_bits {
            best_bits = total_bits;
            best_assignment = assignment;
            best_channels = channels;
        }
    }

    Ok((best_assignment, best_channels))
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
}
