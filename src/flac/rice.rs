//! Rice entropy coding for FLAC residuals.
//!
//! FLAC uses Rice coding (a form of Golomb coding) to compress prediction residuals.
//! Two variants are supported:
//! - RICE (4-bit parameters, max 14)
//! - RICE2 (5-bit parameters, max 30)
//!
//! The residuals are partitioned, with each partition having its own Rice parameter.

use crate::flac::bitstream::{BitReader, BitWriter};
use crate::flac::error::FlacError;

/// Maximum Rice parameter for RICE partition type
pub const RICE_PARAM_MAX: u8 = 14;

/// Maximum Rice parameter for RICE2 partition type
pub const RICE2_PARAM_MAX: u8 = 30;

/// Escape code for RICE (parameter = 15)
pub const RICE_ESCAPE: u8 = 15;

/// Escape code for RICE2 (parameter = 31)
pub const RICE2_ESCAPE: u8 = 31;

/// Rice coding method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiceMethod {
    /// 4-bit Rice parameters (max 14)
    Rice,
    /// 5-bit Rice parameters (max 30)
    Rice2,
}

impl RiceMethod {
    /// Get the parameter bit width
    pub const fn param_bits(self) -> u8 {
        match self {
            RiceMethod::Rice => 4,
            RiceMethod::Rice2 => 5,
        }
    }

    /// Get the maximum parameter value (before escape)
    pub const fn param_max(self) -> u8 {
        match self {
            RiceMethod::Rice => RICE_PARAM_MAX,
            RiceMethod::Rice2 => RICE2_PARAM_MAX,
        }
    }

    /// Get the escape code value
    pub const fn escape_code(self) -> u8 {
        match self {
            RiceMethod::Rice => RICE_ESCAPE,
            RiceMethod::Rice2 => RICE2_ESCAPE,
        }
    }

    /// Subframe residual coding method code (2 bits)
    pub const fn method_code(self) -> u8 {
        match self {
            RiceMethod::Rice => 0b00,
            RiceMethod::Rice2 => 0b01,
        }
    }
}

/// Encode a signed value to unsigned using FLAC's zigzag encoding.
///
/// Positive values map to even numbers, negative to odd:
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
#[inline]
pub const fn signed_to_unsigned(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        ((-value as u32) << 1) - 1
    }
}

/// Decode an unsigned value to signed using FLAC's zigzag encoding.
#[inline]
pub const fn unsigned_to_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value >> 1) as i32
    } else {
        -(((value + 1) >> 1) as i32)
    }
}

/// Decode Rice-coded residuals from a bitstream.
///
/// # Arguments
/// * `reader` - Bitstream reader positioned at residual data
/// * `method` - Rice or Rice2 method
/// * `partition_order` - Number of partitions = 2^partition_order
/// * `block_size` - Total samples in the block
/// * `predictor_order` - Order of the predictor (affects first partition size)
///
/// # Returns
/// Vector of decoded residual values
pub fn decode_residual(
    reader: &mut BitReader,
    method: RiceMethod,
    partition_order: u8,
    block_size: usize,
    predictor_order: usize,
) -> Result<Vec<i32>, FlacError> {
    let num_partitions = 1usize << partition_order;
    let param_bits = method.param_bits();
    let escape_code = method.escape_code();

    // Calculate samples per partition
    // First partition has (block_size / num_partitions) - predictor_order samples
    // All other partitions have (block_size / num_partitions) samples
    let partition_samples = block_size / num_partitions;

    if partition_samples < predictor_order {
        return Err(FlacError::RicePartitionOverflow);
    }

    let total_residuals = block_size - predictor_order;
    let mut residuals = Vec::with_capacity(total_residuals);

    for partition in 0..num_partitions {
        // Read Rice parameter for this partition
        let param = reader.read_bits(param_bits)? as u8;

        // Number of samples in this partition
        let samples = if partition == 0 {
            partition_samples - predictor_order
        } else {
            partition_samples
        };

        if param == escape_code {
            // Escape: raw bits per sample follows
            let bits_per_sample = reader.read_bits(5)? as u8;

            if bits_per_sample > 32 {
                return Err(FlacError::RiceEscapeBitsTooLarge {
                    bits: bits_per_sample,
                });
            }

            for _ in 0..samples {
                let value = if bits_per_sample == 0 {
                    0
                } else {
                    reader.read_bits_signed(bits_per_sample)?
                };
                residuals.push(value);
            }
        } else {
            // Standard Rice decoding
            for _ in 0..samples {
                // Read unary-coded quotient (number of 0s before 1)
                let quotient = reader.read_unary()?;

                // Read binary-coded remainder
                let remainder = if param > 0 {
                    reader.read_bits(param)?
                } else {
                    0
                };

                // Reconstruct unsigned value
                let unsigned = (quotient << param) | remainder;

                // Convert to signed
                let signed = unsigned_to_signed(unsigned);
                residuals.push(signed);
            }
        }
    }

    Ok(residuals)
}

/// Encode residuals using Rice coding.
///
/// # Arguments
/// * `writer` - Bitstream writer
/// * `residuals` - Residual values to encode
/// * `method` - Rice or Rice2 method
/// * `partition_order` - Number of partitions = 2^partition_order
/// * `block_size` - Total samples in the block
/// * `predictor_order` - Order of the predictor
///
/// # Returns
/// Number of bits written
pub fn encode_residual(
    writer: &mut BitWriter,
    residuals: &[i32],
    method: RiceMethod,
    partition_order: u8,
    block_size: usize,
    predictor_order: usize,
) -> Result<usize, FlacError> {
    let start_bits = writer.bits_written();
    let num_partitions = 1usize << partition_order;
    let param_bits = method.param_bits();
    let partition_samples = block_size / num_partitions;

    let mut residual_idx = 0;

    for partition in 0..num_partitions {
        let samples = if partition == 0 {
            partition_samples - predictor_order
        } else {
            partition_samples
        };

        let partition_residuals = &residuals[residual_idx..residual_idx + samples];
        residual_idx += samples;

        // Find optimal Rice parameter for this partition
        let param = find_optimal_rice_param(partition_residuals, method);

        // Check if escape coding would be more efficient
        let rice_bits = estimate_rice_bits(partition_residuals, param);
        let max_abs = partition_residuals
            .iter()
            .map(|&r| r.unsigned_abs())
            .max()
            .unwrap_or(0);
        let escape_bits_per_sample = if max_abs == 0 {
            0
        } else {
            32 - max_abs.leading_zeros()
        } as usize;
        let escape_bits = param_bits as usize + 5 + samples * escape_bits_per_sample;

        if escape_bits < rice_bits {
            // Use escape coding
            writer.write_bits(method.escape_code() as u32, param_bits);
            writer.write_bits(escape_bits_per_sample as u32, 5);

            for &residual in partition_residuals {
                if escape_bits_per_sample > 0 {
                    writer.write_bits_signed(residual, escape_bits_per_sample as u8);
                }
            }
        } else {
            // Use Rice coding
            writer.write_bits(param as u32, param_bits);

            for &residual in partition_residuals {
                let unsigned = signed_to_unsigned(residual);
                let quotient = unsigned >> param;
                let remainder = unsigned & ((1 << param) - 1);

                writer.write_unary(quotient);
                if param > 0 {
                    writer.write_bits(remainder, param);
                }
            }
        }
    }

    Ok(writer.bits_written() - start_bits)
}

/// Find the optimal Rice parameter for a set of residuals.
///
/// Uses a simple heuristic based on the mean absolute value.
pub fn find_optimal_rice_param(residuals: &[i32], method: RiceMethod) -> u8 {
    if residuals.is_empty() {
        return 0;
    }

    // Calculate mean of unsigned (zigzag) values
    let sum: u64 = residuals
        .iter()
        .map(|&r| signed_to_unsigned(r) as u64)
        .sum();
    let mean = sum / residuals.len() as u64;

    // Optimal parameter is approximately log2(mean) when mean > 0
    // k = floor(log2(mean)) for mean >= 1
    let param = if mean == 0 {
        0
    } else {
        (64 - mean.leading_zeros() - 1) as u8
    };

    param.min(method.param_max())
}

/// Find optimal Rice parameter with exhaustive search.
///
/// Tests all valid parameter values and returns the one that minimizes bits.
pub fn find_optimal_rice_param_exhaustive(residuals: &[i32], method: RiceMethod) -> u8 {
    if residuals.is_empty() {
        return 0;
    }

    let mut best_param = 0u8;
    let mut best_bits = usize::MAX;

    for param in 0..=method.param_max() {
        let bits = estimate_rice_bits(residuals, param);
        if bits < best_bits {
            best_bits = bits;
            best_param = param;
        }
    }

    best_param
}

/// Estimate the number of bits needed to Rice-encode residuals with a given parameter.
fn estimate_rice_bits(residuals: &[i32], param: u8) -> usize {
    let mut bits = 0usize;

    for &residual in residuals {
        let unsigned = signed_to_unsigned(residual);
        let quotient = unsigned >> param;

        // Unary: quotient + 1 bits
        bits += quotient as usize + 1;
        // Binary: param bits
        bits += param as usize;
    }

    bits
}

/// Find the optimal partition order for encoding.
///
/// Tests partition orders from min to max and returns the one with minimum bits.
pub fn find_optimal_partition_order(
    residuals: &[i32],
    block_size: usize,
    predictor_order: usize,
    method: RiceMethod,
    min_order: u8,
    max_order: u8,
    exhaustive: bool,
) -> u8 {
    let mut best_order = min_order;
    let mut best_bits = usize::MAX;

    for order in min_order..=max_order {
        let num_partitions = 1usize << order;
        let partition_samples = block_size / num_partitions;

        // Partition must have at least predictor_order samples for first partition
        if partition_samples < predictor_order {
            break;
        }

        // Block size must be divisible by num_partitions
        if block_size % num_partitions != 0 {
            continue;
        }

        let bits = estimate_partition_bits(
            residuals,
            block_size,
            predictor_order,
            order,
            method,
            exhaustive,
        );

        if bits < best_bits {
            best_bits = bits;
            best_order = order;
        }
    }

    best_order
}

/// Estimate bits for a given partition order.
fn estimate_partition_bits(
    residuals: &[i32],
    block_size: usize,
    predictor_order: usize,
    partition_order: u8,
    method: RiceMethod,
    exhaustive: bool,
) -> usize {
    let num_partitions = 1usize << partition_order;
    let partition_samples = block_size / num_partitions;
    let param_bits = method.param_bits() as usize;

    let mut total_bits = 0;
    let mut residual_idx = 0;

    for partition in 0..num_partitions {
        let samples = if partition == 0 {
            partition_samples - predictor_order
        } else {
            partition_samples
        };

        let partition_residuals = &residuals[residual_idx..residual_idx + samples];
        residual_idx += samples;

        let param = if exhaustive {
            find_optimal_rice_param_exhaustive(partition_residuals, method)
        } else {
            find_optimal_rice_param(partition_residuals, method)
        };

        // Parameter bits + encoded residual bits
        total_bits += param_bits + estimate_rice_bits(partition_residuals, param);
    }

    total_bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_unsigned_roundtrip() {
        for value in [-100, -1, 0, 1, 100, i32::MIN + 1, i32::MAX] {
            let unsigned = signed_to_unsigned(value);
            let signed = unsigned_to_signed(unsigned);
            assert_eq!(signed, value, "Roundtrip failed for {}", value);
        }
    }

    #[test]
    fn test_zigzag_encoding() {
        assert_eq!(signed_to_unsigned(0), 0);
        assert_eq!(signed_to_unsigned(-1), 1);
        assert_eq!(signed_to_unsigned(1), 2);
        assert_eq!(signed_to_unsigned(-2), 3);
        assert_eq!(signed_to_unsigned(2), 4);
    }

    #[test]
    fn test_zigzag_decoding() {
        assert_eq!(unsigned_to_signed(0), 0);
        assert_eq!(unsigned_to_signed(1), -1);
        assert_eq!(unsigned_to_signed(2), 1);
        assert_eq!(unsigned_to_signed(3), -2);
        assert_eq!(unsigned_to_signed(4), 2);
    }

    #[test]
    fn test_rice_encode_decode_roundtrip() {
        let residuals: Vec<i32> = vec![0, 1, -1, 2, -2, 5, -10, 100, -100];
        let block_size = 16;
        let predictor_order = 0;
        let partition_order = 0;

        // Pad to block size
        let mut padded = residuals.clone();
        padded.resize(block_size, 0);

        let mut writer = BitWriter::new();
        encode_residual(
            &mut writer,
            &padded,
            RiceMethod::Rice,
            partition_order,
            block_size,
            predictor_order,
        )
        .expect("Encode failed");

        let data = writer.finish();
        let mut reader = BitReader::new(&data);

        let decoded = decode_residual(
            &mut reader,
            RiceMethod::Rice,
            partition_order,
            block_size,
            predictor_order,
        )
        .expect("Decode failed");

        assert_eq!(decoded, padded);
    }

    #[test]
    fn test_optimal_rice_param() {
        // Small residuals should have small parameter
        let small: Vec<i32> = vec![0, 1, -1, 0, 1, 0, -1, 1];
        let param_small = find_optimal_rice_param(&small, RiceMethod::Rice);
        assert!(param_small <= 2);

        // Larger residuals should have larger parameter
        let large: Vec<i32> = vec![100, -150, 200, -250, 300, -350, 400, -450];
        let param_large = find_optimal_rice_param(&large, RiceMethod::Rice);
        assert!(param_large > param_small);
    }

    #[test]
    fn test_rice2_larger_params() {
        // RICE2 should be able to handle larger residuals with larger params
        let residuals: Vec<i32> = vec![10000, -20000, 30000, -40000];
        let param_rice = find_optimal_rice_param_exhaustive(&residuals, RiceMethod::Rice);
        let param_rice2 = find_optimal_rice_param_exhaustive(&residuals, RiceMethod::Rice2);

        // RICE2 should be able to use a larger parameter
        assert!(param_rice <= RICE_PARAM_MAX);
        assert!(param_rice2 <= RICE2_PARAM_MAX);
    }

    #[test]
    fn test_partition_order() {
        let residuals: Vec<i32> = (0..4096).map(|i| ((i % 10) as i32) - 5).collect();
        let block_size = 4096;
        let predictor_order = 4;

        let order = find_optimal_partition_order(
            &residuals[predictor_order..],
            block_size,
            predictor_order,
            RiceMethod::Rice,
            0,
            6,
            false,
        );

        // Should find a reasonable partition order
        assert!(order <= 6);
    }
}
