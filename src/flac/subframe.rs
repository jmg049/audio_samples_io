//! FLAC subframe encoding and decoding.
//!
//! Each FLAC frame contains one subframe per channel. Subframe types:
//! - CONSTANT: All samples have the same value
//! - VERBATIM: Uncompressed samples
//! - FIXED: Fixed linear predictor (orders 0-4)
//! - LPC: Linear predictive coding (orders 1-32)

use crate::flac::bitstream::{BitReader, BitWriter};
use crate::flac::error::FlacError;
use crate::flac::lpc::{
    fixed_predictor_residual, fixed_predictor_restore, lpc_predictor_residual,
    lpc_predictor_restore,
};
use crate::flac::rice::{RiceMethod, decode_residual, encode_residual};

/// Subframe type codes (from 6-bit header)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubframeType {
    /// All samples are the same value
    Constant,
    /// Uncompressed samples
    Verbatim,
    /// Fixed predictor with given order (0-4)
    Fixed(u8),
    /// LPC predictor with given order (1-32)
    Lpc(u8),
}

impl SubframeType {
    /// Parse subframe type from 6-bit header value.
    pub fn from_header(value: u8) -> Result<Self, FlacError> {
        match value {
            0b000000 => Ok(SubframeType::Constant),
            0b000001 => Ok(SubframeType::Verbatim),
            0b001000..=0b001100 => Ok(SubframeType::Fixed((value & 0x07) as u8)),
            0b100000..=0b111111 => Ok(SubframeType::Lpc(((value & 0x1F) + 1) as u8)),
            _ => Err(FlacError::InvalidSubframeType(value)),
        }
    }

    /// Convert to 6-bit header value.
    pub fn to_header(self) -> u8 {
        match self {
            SubframeType::Constant => 0b000000,
            SubframeType::Verbatim => 0b000001,
            SubframeType::Fixed(order) => 0b001000 | (order & 0x07),
            SubframeType::Lpc(order) => 0b100000 | ((order - 1) & 0x1F),
        }
    }

    /// Get the predictor order for this subframe type.
    pub const fn order(self) -> usize {
        match self {
            SubframeType::Constant | SubframeType::Verbatim => 0,
            SubframeType::Fixed(o) | SubframeType::Lpc(o) => o as usize,
        }
    }
}

/// Decoded subframe data.
#[derive(Debug, Clone)]
pub struct Subframe {
    /// Subframe type
    pub subframe_type: SubframeType,
    /// Number of wasted bits per sample (shifted out)
    pub wasted_bits: u8,
    /// Decoded samples
    pub samples: Vec<i32>,
}

/// Subframe header information.
#[derive(Debug, Clone, Copy)]
pub struct SubframeHeader {
    /// Subframe type
    pub subframe_type: SubframeType,
    /// Number of wasted bits per sample
    pub wasted_bits: u8,
}

impl SubframeHeader {
    /// Parse subframe header from bitstream.
    pub fn decode(reader: &mut BitReader) -> Result<Self, FlacError> {
        // Zero padding bit (must be 0)
        let zero = reader.read_bit()?;
        if zero {
            return Err(FlacError::InvalidSubframeType(0xFF));
        }

        // 6-bit subframe type
        let type_code = reader.read_bits(6)? as u8;
        let subframe_type = SubframeType::from_header(type_code)?;

        // Wasted bits flag
        let has_wasted = reader.read_bit()?;
        let wasted_bits = if has_wasted {
            // Read unary-coded wasted bits count (k+1 where k is the count of 0s)
            let k = reader.read_unary()? as u8;
            k + 1
        } else {
            0
        };

        Ok(SubframeHeader {
            subframe_type,
            wasted_bits,
        })
    }

    /// Encode subframe header to bitstream.
    pub fn encode(&self, writer: &mut BitWriter) {
        // Zero padding bit
        writer.write_bit(false);

        // 6-bit subframe type
        writer.write_bits(self.subframe_type.to_header() as u32, 6);

        // Wasted bits flag
        if self.wasted_bits > 0 {
            writer.write_bit(true);
            writer.write_unary((self.wasted_bits - 1) as u32);
        } else {
            writer.write_bit(false);
        }
    }
}

/// Decode a complete subframe.
///
/// # Arguments
/// * `reader` - Bitstream reader
/// * `block_size` - Number of samples in the block
/// * `bits_per_sample` - Bits per sample (may be adjusted for channel assignment)
pub fn decode_subframe(
    reader: &mut BitReader,
    block_size: usize,
    bits_per_sample: u8,
) -> Result<Subframe, FlacError> {
    let header = SubframeHeader::decode(reader)?;

    // Adjust bits per sample for wasted bits
    let effective_bits = bits_per_sample - header.wasted_bits;

    let samples = match header.subframe_type {
        SubframeType::Constant => decode_constant(reader, block_size, effective_bits)?,
        SubframeType::Verbatim => decode_verbatim(reader, block_size, effective_bits)?,
        SubframeType::Fixed(order) => {
            decode_fixed(reader, block_size, effective_bits, order as usize)?
        }
        SubframeType::Lpc(order) => decode_lpc(reader, block_size, effective_bits, order as usize)?,
    };

    // Restore wasted bits
    let samples = if header.wasted_bits > 0 {
        samples
            .into_iter()
            .map(|s| s << header.wasted_bits)
            .collect()
    } else {
        samples
    };

    Ok(Subframe {
        subframe_type: header.subframe_type,
        wasted_bits: header.wasted_bits,
        samples,
    })
}

/// Decode a CONSTANT subframe.
fn decode_constant(
    reader: &mut BitReader,
    block_size: usize,
    bits_per_sample: u8,
) -> Result<Vec<i32>, FlacError> {
    let value = reader.read_bits_signed(bits_per_sample)?;
    Ok(vec![value; block_size])
}

/// Decode a VERBATIM subframe.
fn decode_verbatim(
    reader: &mut BitReader,
    block_size: usize,
    bits_per_sample: u8,
) -> Result<Vec<i32>, FlacError> {
    let mut samples = Vec::with_capacity(block_size);
    for _ in 0..block_size {
        samples.push(reader.read_bits_signed(bits_per_sample)?);
    }
    Ok(samples)
}

/// Decode a FIXED subframe.
fn decode_fixed(
    reader: &mut BitReader,
    block_size: usize,
    bits_per_sample: u8,
    order: usize,
) -> Result<Vec<i32>, FlacError> {
    // Read warm-up samples
    let mut warmup = Vec::with_capacity(order);
    for _ in 0..order {
        warmup.push(reader.read_bits_signed(bits_per_sample)?);
    }

    // Read residual coding method
    let method_code = reader.read_bits(2)? as u8;
    let method = match method_code {
        0b00 => RiceMethod::Rice,
        0b01 => RiceMethod::Rice2,
        _ => return Err(FlacError::InvalidSubframeType(method_code)),
    };

    // Read partition order
    let partition_order = reader.read_bits(4)? as u8;

    // Decode residuals
    let residuals = decode_residual(reader, method, partition_order, block_size, order)?;

    // Restore samples
    fixed_predictor_restore(&warmup, &residuals, order)
}

/// Decode an LPC subframe.
fn decode_lpc(
    reader: &mut BitReader,
    block_size: usize,
    bits_per_sample: u8,
    order: usize,
) -> Result<Vec<i32>, FlacError> {
    // Read warm-up samples
    let mut warmup = Vec::with_capacity(order);
    for _ in 0..order {
        warmup.push(reader.read_bits_signed(bits_per_sample)?);
    }

    // Read QLP coefficient precision (4 bits, 0 = invalid, otherwise precision-1)
    let qlp_precision_code = reader.read_bits(4)? as u8;
    if qlp_precision_code == 0b1111 {
        return Err(FlacError::InvalidQlpPrecision {
            precision: qlp_precision_code,
        });
    }
    let qlp_precision = qlp_precision_code + 1;

    // Read QLP shift (5-bit signed)
    let qlp_shift = reader.read_bits_signed(5)? as i8;
    if qlp_shift < 0 {
        return Err(FlacError::InvalidLpcShift { shift: qlp_shift });
    }

    // Read quantized LPC coefficients
    let mut qlp_coeffs = Vec::with_capacity(order);
    for _ in 0..order {
        qlp_coeffs.push(reader.read_bits_signed(qlp_precision)?);
    }

    // Read residual coding method
    let method_code = reader.read_bits(2)? as u8;
    let method = match method_code {
        0b00 => RiceMethod::Rice,
        0b01 => RiceMethod::Rice2,
        _ => return Err(FlacError::InvalidSubframeType(method_code)),
    };

    // Read partition order
    let partition_order = reader.read_bits(4)? as u8;

    // Decode residuals
    let residuals = decode_residual(reader, method, partition_order, block_size, order)?;

    // Restore samples
    lpc_predictor_restore(&warmup, &residuals, &qlp_coeffs, qlp_shift)
}

/// Encoded subframe result.
#[derive(Debug)]
pub struct EncodedSubframe {
    /// Subframe type that was used
    pub subframe_type: SubframeType,
    /// Number of bits used
    pub bits: usize,
    /// Encoded data
    pub data: Vec<u8>,
}

/// Encode a subframe with automatic type selection.
///
/// # Arguments
/// * `samples` - Samples to encode
/// * `bits_per_sample` - Original bits per sample
/// * `block_size` - Number of samples
/// * `max_lpc_order` - Maximum LPC order to try (0 = fixed only)
/// * `qlp_precision` - Quantization precision for LPC coefficients
/// * `min_partition_order` - Minimum Rice partition order
/// * `max_partition_order` - Maximum Rice partition order
/// * `exhaustive_rice` - Use exhaustive Rice parameter search
pub fn encode_subframe(
    samples: &[i32],
    bits_per_sample: u8,
    max_lpc_order: usize,
    qlp_precision: u8,
    min_partition_order: u8,
    max_partition_order: u8,
    exhaustive_rice: bool,
) -> Result<EncodedSubframe, FlacError> {
    let block_size = samples.len();

    // Check for wasted bits (common low bits that are zero)
    let wasted_bits = compute_wasted_bits(samples);
    let effective_samples: Vec<i32> = if wasted_bits > 0 {
        samples.iter().map(|&s| s >> wasted_bits).collect()
    } else {
        samples.to_vec()
    };
    let effective_bits = bits_per_sample - wasted_bits;

    // Try CONSTANT
    if let Some(encoded) = try_encode_constant(&effective_samples, effective_bits, wasted_bits) {
        return Ok(encoded);
    }

    // Estimate VERBATIM size
    let verbatim_bits = 8 + block_size * effective_bits as usize;

    let mut best = encode_verbatim(&effective_samples, effective_bits, wasted_bits);

    // Try FIXED predictors
    for order in 0..=4.min(block_size - 1) {
        if let Ok(encoded) = try_encode_fixed(
            &effective_samples,
            effective_bits,
            wasted_bits,
            order,
            min_partition_order,
            max_partition_order,
            exhaustive_rice,
        ) {
            if encoded.bits < best.bits {
                best = encoded;
            }
        }
    }

    // Try LPC if enabled and beneficial
    if max_lpc_order > 0 && block_size > max_lpc_order {
        if let Ok(encoded) = try_encode_lpc(
            &effective_samples,
            effective_bits,
            wasted_bits,
            max_lpc_order,
            qlp_precision,
            min_partition_order,
            max_partition_order,
            exhaustive_rice,
        ) {
            if encoded.bits < best.bits {
                best = encoded;
            }
        }
    }

    // Fall back to verbatim if nothing is better
    if best.bits > verbatim_bits {
        best = encode_verbatim(&effective_samples, effective_bits, wasted_bits);
    }

    Ok(best)
}

/// Compute number of wasted bits (common trailing zeros).
fn compute_wasted_bits(samples: &[i32]) -> u8 {
    if samples.is_empty() {
        return 0;
    }

    // OR all samples together
    let combined = samples.iter().fold(0i32, |acc, &s| acc | s);

    if combined == 0 {
        return 0; // All zeros, no wasted bits concept applies
    }

    combined.trailing_zeros() as u8
}

/// Try to encode as CONSTANT subframe.
fn try_encode_constant(
    samples: &[i32],
    bits_per_sample: u8,
    wasted_bits: u8,
) -> Option<EncodedSubframe> {
    if samples.is_empty() {
        return None;
    }

    let first = samples[0];
    if samples.iter().all(|&s| s == first) {
        let mut writer = BitWriter::new();

        // Header
        let header = SubframeHeader {
            subframe_type: SubframeType::Constant,
            wasted_bits,
        };
        header.encode(&mut writer);

        // Single sample value
        writer.write_bits_signed(first, bits_per_sample);

        let data = writer.finish();
        let bits = data.len() * 8;

        Some(EncodedSubframe {
            subframe_type: SubframeType::Constant,
            bits,
            data,
        })
    } else {
        None
    }
}

/// Encode as VERBATIM subframe.
fn encode_verbatim(samples: &[i32], bits_per_sample: u8, wasted_bits: u8) -> EncodedSubframe {
    let mut writer = BitWriter::new();

    // Header
    let header = SubframeHeader {
        subframe_type: SubframeType::Verbatim,
        wasted_bits,
    };
    header.encode(&mut writer);

    // All samples
    for &sample in samples {
        writer.write_bits_signed(sample, bits_per_sample);
    }

    let data = writer.finish();
    let bits = data.len() * 8;

    EncodedSubframe {
        subframe_type: SubframeType::Verbatim,
        bits,
        data,
    }
}

/// Try to encode as FIXED subframe.
fn try_encode_fixed(
    samples: &[i32],
    bits_per_sample: u8,
    wasted_bits: u8,
    order: usize,
    min_partition_order: u8,
    max_partition_order: u8,
    exhaustive_rice: bool,
) -> Result<EncodedSubframe, FlacError> {
    let block_size = samples.len();

    // Compute residuals
    let residuals = fixed_predictor_residual(samples, order)?;

    // Find optimal partition order
    let partition_order = crate::flac::rice::find_optimal_partition_order(
        &residuals,
        block_size,
        order,
        RiceMethod::Rice,
        min_partition_order,
        max_partition_order,
        exhaustive_rice,
    );

    let mut writer = BitWriter::new();

    // Header
    let header = SubframeHeader {
        subframe_type: SubframeType::Fixed(order as u8),
        wasted_bits,
    };
    header.encode(&mut writer);

    // Warm-up samples
    for &sample in &samples[..order] {
        writer.write_bits_signed(sample, bits_per_sample);
    }

    // Residual coding method (RICE = 0b00)
    writer.write_bits(0b00, 2);

    // Partition order
    writer.write_bits(partition_order as u32, 4);

    // Encode residuals
    encode_residual(
        &mut writer,
        &residuals,
        RiceMethod::Rice,
        partition_order,
        block_size,
        order,
    )?;

    let data = writer.finish();
    let bits = data.len() * 8;

    Ok(EncodedSubframe {
        subframe_type: SubframeType::Fixed(order as u8),
        bits,
        data,
    })
}

/// Try to encode as LPC subframe.
fn try_encode_lpc(
    samples: &[i32],
    bits_per_sample: u8,
    wasted_bits: u8,
    max_order: usize,
    qlp_precision: u8,
    min_partition_order: u8,
    max_partition_order: u8,
    exhaustive_rice: bool,
) -> Result<EncodedSubframe, FlacError> {
    let block_size = samples.len();

    // Find best LPC order and coefficients
    let (order, qlp_coeffs, qlp_shift) =
        crate::flac::lpc::find_best_lpc_order(samples, max_order, qlp_precision, false)?;

    if order == 0 {
        return Err(FlacError::InvalidLpcOrder { order: 0 });
    }

    // Compute residuals
    let residuals = lpc_predictor_residual(samples, &qlp_coeffs, qlp_shift)?;

    // Find optimal partition order
    let partition_order = crate::flac::rice::find_optimal_partition_order(
        &residuals,
        block_size,
        order,
        RiceMethod::Rice,
        min_partition_order,
        max_partition_order,
        exhaustive_rice,
    );

    let mut writer = BitWriter::new();

    // Header
    let header = SubframeHeader {
        subframe_type: SubframeType::Lpc(order as u8),
        wasted_bits,
    };
    header.encode(&mut writer);

    // Warm-up samples
    for &sample in &samples[..order] {
        writer.write_bits_signed(sample, bits_per_sample);
    }

    // QLP precision (stored as precision - 1)
    writer.write_bits((qlp_precision - 1) as u32, 4);

    // QLP shift
    writer.write_bits_signed(qlp_shift as i32, 5);

    // QLP coefficients
    for &coeff in &qlp_coeffs {
        writer.write_bits_signed(coeff, qlp_precision);
    }

    // Residual coding method (RICE = 0b00)
    writer.write_bits(0b00, 2);

    // Partition order
    writer.write_bits(partition_order as u32, 4);

    // Encode residuals
    encode_residual(
        &mut writer,
        &residuals,
        RiceMethod::Rice,
        partition_order,
        block_size,
        order,
    )?;

    let data = writer.finish();
    let bits = data.len() * 8;

    Ok(EncodedSubframe {
        subframe_type: SubframeType::Lpc(order as u8),
        bits,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subframe_type_roundtrip() {
        let types = [
            SubframeType::Constant,
            SubframeType::Verbatim,
            SubframeType::Fixed(0),
            SubframeType::Fixed(4),
            SubframeType::Lpc(1),
            SubframeType::Lpc(32),
        ];

        for stype in types {
            let header = stype.to_header();
            let parsed = SubframeType::from_header(header).unwrap();
            assert_eq!(parsed, stype, "Roundtrip failed for {:?}", stype);
        }
    }

    #[test]
    fn test_constant_subframe() {
        let samples = vec![42i32; 100];
        let encoded = try_encode_constant(&samples, 16, 0);
        assert!(encoded.is_some(), "Should encode as constant");

        let encoded = encoded.unwrap();
        assert_eq!(encoded.subframe_type, SubframeType::Constant);
    }

    #[test]
    fn test_verbatim_subframe() {
        let samples: Vec<i32> = (0..100).collect();
        let encoded = encode_verbatim(&samples, 16, 0);
        assert_eq!(encoded.subframe_type, SubframeType::Verbatim);
    }

    #[test]
    fn test_fixed_subframe_roundtrip() {
        let samples: Vec<i32> = (0..256).map(|i| i * 10).collect();
        let bits_per_sample = 16;

        let encoded =
            try_encode_fixed(&samples, bits_per_sample, 0, 2, 0, 4, false).expect("Encode failed");

        assert_eq!(encoded.subframe_type, SubframeType::Fixed(2));

        // Decode
        let mut reader = BitReader::new(&encoded.data);
        let decoded =
            decode_subframe(&mut reader, samples.len(), bits_per_sample).expect("Decode failed");

        assert_eq!(decoded.samples, samples);
    }

    #[test]
    fn test_wasted_bits() {
        // All samples have 2 trailing zeros
        let samples: Vec<i32> = vec![4, 8, 12, 16, 20];
        let wasted = compute_wasted_bits(&samples);
        assert_eq!(wasted, 2);

        // No wasted bits
        let samples: Vec<i32> = vec![1, 3, 5, 7];
        let wasted = compute_wasted_bits(&samples);
        assert_eq!(wasted, 0);
    }

    #[test]
    fn test_encode_subframe_auto() {
        // Linear ramp should prefer FIXED
        let samples: Vec<i32> = (0..256).map(|i| i * 100).collect();
        let encoded = encode_subframe(&samples, 24, 0, 12, 0, 4, false).expect("Encode failed");

        // Should not choose verbatim for compressible data
        assert_ne!(encoded.subframe_type, SubframeType::Verbatim);
    }
}
