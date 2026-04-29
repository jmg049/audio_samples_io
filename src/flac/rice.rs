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
///
/// Branchless: u32 right-shift first (logical), then XOR with sign mask.
/// `-(value & 1)` is 0x00000000 (even) or 0xFFFFFFFF (odd).
#[inline]
pub const fn unsigned_to_signed(value: u32) -> i32 {
    ((value >> 1) as i32) ^ -((value & 1) as i32)
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
    let mut write_idx = 0usize;

    for partition in 0..num_partitions {
        let param = reader.read_bits(param_bits)? as u8;

        let samples = if partition == 0 {
            partition_samples - predictor_order
        } else {
            partition_samples
        };

        if param == escape_code {
            let bits_per_sample = reader.read_bits(5)? as u8;

            if bits_per_sample > 32 {
                return Err(FlacError::RiceEscapeBitsTooLarge {
                    bits: bits_per_sample,
                });
            }

            // Safety: total_residuals is pre-allocated capacity; write_idx advances
            // monotonically and never exceeds total_residuals.
            let ptr: *mut i32 = residuals.as_mut_ptr();
            for _ in 0..samples {
                let value = if bits_per_sample == 0 {
                    0
                } else {
                    reader.read_bits_signed(bits_per_sample)?
                };
                unsafe { ptr.add(write_idx).write(value); }
                write_idx += 1;
            }
        } else {
            let ptr = residuals.as_mut_ptr();
            if param == 0 {
                // No remainder bits — only the unary quotient maps to the value.
                for _ in 0..samples {
                    let quotient = reader.read_unary()?;
                    let signed = unsigned_to_signed(quotient);
                    unsafe { ptr.add(write_idx).write(signed); }
                    write_idx += 1;
                }
            } else {
                for _ in 0..samples {
                    let quotient = reader.read_unary()?;
                    let remainder = reader.read_bits(param)?;
                    let unsigned = (quotient << param) | remainder;
                    let signed = unsigned_to_signed(unsigned);
                    unsafe { ptr.add(write_idx).write(signed); }
                    write_idx += 1;
                }
            }
        }
    }

    // Safety: write_idx == total_residuals after the loop (all partitions decoded).
    unsafe { residuals.set_len(write_idx); }
    Ok(residuals)
}

/// Decode Rice-coded residuals into an existing buffer (zero-allocation path).
///
/// Clears `out`, reserves capacity, then fills it in-place using the same
/// unsafe pointer-write logic as `decode_residual`. Eliminates one heap
/// allocation per subframe on the read path.
pub fn decode_residual_into(
    reader: &mut BitReader,
    method: RiceMethod,
    partition_order: u8,
    block_size: usize,
    predictor_order: usize,
    out: &mut Vec<i32>,
) -> Result<(), FlacError> {
    let num_partitions = 1usize << partition_order;
    let param_bits = method.param_bits();
    let escape_code = method.escape_code();

    let partition_samples = block_size / num_partitions;

    if partition_samples < predictor_order {
        return Err(FlacError::RicePartitionOverflow);
    }

    let total_residuals = block_size - predictor_order;
    out.clear();
    out.reserve(total_residuals);
    let mut write_idx = 0usize;

    for partition in 0..num_partitions {
        let param = reader.read_bits(param_bits)? as u8;

        let samples = if partition == 0 {
            partition_samples - predictor_order
        } else {
            partition_samples
        };

        if param == escape_code {
            let bits_per_sample = reader.read_bits(5)? as u8;

            if bits_per_sample > 32 {
                return Err(FlacError::RiceEscapeBitsTooLarge {
                    bits: bits_per_sample,
                });
            }

            let ptr: *mut i32 = out.as_mut_ptr();
            for _ in 0..samples {
                let value = if bits_per_sample == 0 {
                    0
                } else {
                    reader.read_bits_signed(bits_per_sample)?
                };
                unsafe { ptr.add(write_idx).write(value); }
                write_idx += 1;
            }
        } else {
            let ptr = out.as_mut_ptr();
            if param == 0 {
                for _ in 0..samples {
                    let quotient = reader.read_unary()?;
                    let signed = unsigned_to_signed(quotient);
                    unsafe { ptr.add(write_idx).write(signed); }
                    write_idx += 1;
                }
            } else {
                for _ in 0..samples {
                    let quotient = reader.read_unary()?;
                    let remainder = reader.read_bits(param)?;
                    let unsigned = (quotient << param) | remainder;
                    let signed = unsigned_to_signed(unsigned);
                    unsafe { ptr.add(write_idx).write(signed); }
                    write_idx += 1;
                }
            }
        }
    }

    unsafe { out.set_len(write_idx); }
    Ok(())
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
        let rice_bits = estimate_rice_bits_exact(partition_residuals, param);
        let max_abs = partition_residuals
            .iter()
            .map(|&r| r.unsigned_abs())
            .max()
            .unwrap_or(0);
        // +1 for the sign bit: signed N-bit needs 2^(N-1) > max_abs.
        // Cap at 31 to fit in the 5-bit escape field.
        let escape_bits_per_sample = if max_abs == 0 {
            0u32
        } else {
            (32 - max_abs.leading_zeros() + 1).min(31)
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
            // Use Rice coding — fused unary+binary write (see encode_residual_planned)
            writer.write_bits(param as u32, param_bits);
            for &residual in partition_residuals {
                let unsigned = signed_to_unsigned(residual);
                let quotient = unsigned >> param;
                let remainder = unsigned & ((1u32 << param) - 1);
                let total_bits = quotient + 1 + param as u32;
                if total_bits <= 32 {
                    writer.write_bits((1u32 << param) | remainder, total_bits as u8);
                } else {
                    let mut q = quotient;
                    while q >= 32 { writer.write_bits(0, 32); q -= 32; }
                    writer.write_bits((1u32 << param) | remainder, (q + 1 + param as u32) as u8);
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
    rice_param_from_sum(
        residuals.iter().map(|&r| signed_to_unsigned(r) as u64).sum(),
        residuals.len(),
        method,
    )
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
        let bits = estimate_rice_bits_exact(residuals, param);
        if bits < best_bits {
            best_bits = bits;
            best_param = param;
        }
    }

    best_param
}

/// Derive the optimal Rice parameter from a precomputed zigzag sum and count.
#[inline]
fn rice_param_from_sum(sum: u64, count: usize, method: RiceMethod) -> u8 {
    if count == 0 || sum == 0 {
        return 0;
    }
    let mean = sum / count as u64;
    let param = if mean == 0 { 0u8 } else { (63 - mean.leading_zeros()) as u8 };
    param.min(method.param_max())
}

/// Estimate Rice bits using the fast sum-based approximation.
///
/// Approximation: bits ≈ count*(k+1) + sum_zigzag>>k.
/// Error bounded by `count` (one bit per sample rounding), suitable for
/// partition-order comparison and escape-coding decisions.
#[inline]
const fn estimate_rice_bits_from_sum(sum: u64, count: usize, param: u8) -> usize {
    count * (param as usize + 1) + (sum >> param) as usize
}

/// Exact Rice bits (used only for exhaustive search).
fn estimate_rice_bits_exact(residuals: &[i32], param: u8) -> usize {
    residuals.iter().map(|&r| {
        let u = signed_to_unsigned(r);
        (u >> param) as usize + 1 + param as usize
    }).sum()
}

/// Maximum partition order we support on the stack (covers all compression levels 0-8).
/// Partition order is 4 bits in the bitstream, but our levels cap at 6; 8 gives headroom.
const MAX_RICE_PARTITION_ORDER: u8 = 8;
const MAX_RICE_PARTITIONS: usize = 1 << MAX_RICE_PARTITION_ORDER; // 256

/// Per-partition data precomputed in a single O(N) pass.
#[derive(Copy, Clone)]
struct PartitionData {
    sum_zigzag: u64,
    max_abs:    u32,
    count:      usize,
}

/// Stack-allocated result of `plan_residual_coding`.
///
/// Replaces the `(u8, Vec<u8>)` return type, eliminating one heap allocation
/// per call. Access the valid slice with `params_slice()`.
/// `est_bits` is the estimated residual section bit count (includes per-partition
/// param storage but not the method/partition-order fields).
pub struct RicePlan {
    pub order: u8,
    pub est_bits: usize,
    params: [u8; MAX_RICE_PARTITIONS],
}

impl RicePlan {
    #[inline]
    pub fn params_slice(&self) -> &[u8] {
        &self.params[..1usize << self.order]
    }
}

/// Compute per-partition data at the finest granularity (max_order) in one pass.
/// Writes into `buf[..num_leaf]` and returns `num_leaf = 2^max_order`.
fn compute_leaf_data(
    residuals: &[i32],
    block_size: usize,
    predictor_order: usize,
    max_order: u8,
    buf: &mut [PartitionData; MAX_RICE_PARTITIONS],
) -> usize {
    let num_leaf = 1usize << max_order;
    let leaf_size = block_size / num_leaf;

    for (p, leaf) in buf[..num_leaf].iter_mut().enumerate() {
        leaf.sum_zigzag = 0;
        leaf.max_abs = 0;
        leaf.count = if p == 0 { leaf_size.saturating_sub(predictor_order) } else { leaf_size };
    }

    let mut pos = 0usize;
    for (leaf_idx, leaf) in buf[..num_leaf].iter_mut().enumerate() {
        let start = if leaf_idx == 0 { 0 } else { pos };
        let end = start + leaf.count;
        for &r in &residuals[start..end.min(residuals.len())] {
            let u = signed_to_unsigned(r);
            leaf.sum_zigzag += u as u64;
            if u > leaf.max_abs { leaf.max_abs = u; }
        }
        pos = end;
    }

    num_leaf
}

/// Plan residual coding: find the best partition order and Rice params in O(N).
///
/// Returns a `RicePlan` (stack-allocated) — no heap allocation.
/// Pass `plan.order` and `plan.params_slice()` to `encode_residual_planned`.
pub fn plan_residual_coding(
    residuals: &[i32],
    block_size: usize,
    predictor_order: usize,
    method: RiceMethod,
    min_order: u8,
    max_order: u8,
    exhaustive: bool,
) -> RicePlan {
    // Clamp max_order so leaf partitions are at least 1 sample
    let max_order = {
        let mut m = max_order.min(MAX_RICE_PARTITION_ORDER);
        while m > min_order {
            let leaf_size = block_size >> m;
            if leaf_size >= predictor_order.max(1) && block_size.is_multiple_of(1 << m) { break; }
            m -= 1;
        }
        m
    };

    // One pass: precompute per-leaf (finest granularity) zigzag sums and max_abs.
    // Stack-allocated — eliminates a heap allocation per call.
    let mut leaf_buf = [PartitionData { sum_zigzag: 0, max_abs: 0, count: 0 }; MAX_RICE_PARTITIONS];
    let num_leaf = compute_leaf_data(residuals, block_size, predictor_order, max_order, &mut leaf_buf);
    let leaves = &leaf_buf[..num_leaf];

    let param_bits = method.param_bits() as usize;
    let escape_param_overhead = param_bits + 5;

    let mut plan = RicePlan { order: min_order, est_bits: 0, params: [0u8; MAX_RICE_PARTITIONS] };
    let mut best_bits = usize::MAX;
    // Stack buffer for per-iteration params — only 256 bytes, cheap to init.
    let mut cur_params = [0u8; MAX_RICE_PARTITIONS];

    for order in min_order..=max_order {
        let num_partitions = 1usize << order;
        if !block_size.is_multiple_of(num_partitions) { continue; }
        let partition_size = block_size / num_partitions;
        if partition_size < predictor_order { break; }

        let leaves_per = num_leaf / num_partitions;
        let mut total_bits = 0usize;

        for (p, item) in cur_params.iter_mut().enumerate().take(num_partitions) {
            let ls = p * leaves_per;
            let le = ls + leaves_per;
            let mut sum = 0u64;
            let mut max_abs = 0u32;
            let mut count = 0usize;
            for leaf in &leaves[ls..le] {
                sum += leaf.sum_zigzag;
                if leaf.max_abs > max_abs { max_abs = leaf.max_abs; }
                count += leaf.count;
            }

            let param = if exhaustive {
                let start: usize = leaves[..ls].iter().map(|l| l.count).sum();
                let slice = &residuals[start..start + count];
                find_optimal_rice_param_exhaustive(slice, method)
            } else {
                rice_param_from_sum(sum, count, method)
            };

            let rice_bits = param_bits + estimate_rice_bits_from_sum(sum, count, param);
            let escape_bps = if max_abs == 0 { 0u32 }
                             else { (32 - max_abs.leading_zeros() + 1).min(31) };
            let escape_bits = escape_param_overhead + count * escape_bps as usize;

            total_bits += rice_bits.min(escape_bits);
            *item = if escape_bits < rice_bits { method.escape_code() } else { param };
        }

        if total_bits < best_bits {
            best_bits = total_bits;
            plan.order = order;
            plan.est_bits = total_bits;
            plan.params[..num_partitions].copy_from_slice(&cur_params[..num_partitions]);
        }
    }

    plan
}

/// Encode residuals using pre-planned Rice params (no re-scan for param selection).
pub fn encode_residual_planned(
    writer: &mut BitWriter,
    residuals: &[i32],
    method: RiceMethod,
    partition_order: u8,
    params: &[u8],
    block_size: usize,
    predictor_order: usize,
) -> Result<usize, FlacError> {
    let start_bits = writer.bits_written();
    let num_partitions = 1usize << partition_order;
    let param_bits = method.param_bits();
    let partition_samples = block_size / num_partitions;
    let escape_code = method.escape_code();

    let mut residual_idx = 0;

    for (partition, &param) in params.iter().enumerate().take(num_partitions) {
        let samples = if partition == 0 {
            partition_samples - predictor_order
        } else {
            partition_samples
        };

        let partition_residuals = &residuals[residual_idx..residual_idx + samples];
        residual_idx += samples;

        if param == escape_code {
            // Escape coding: raw signed values
            let max_abs = partition_residuals.iter().map(|&r| r.unsigned_abs()).max().unwrap_or(0);
            let escape_bps = if max_abs == 0 { 0u32 }
                             else { (32 - max_abs.leading_zeros() + 1).min(31) };
            writer.write_bits(escape_code as u32, param_bits);
            writer.write_bits(escape_bps, 5);
            for &residual in partition_residuals {
                if escape_bps > 0 {
                    writer.write_bits_signed(residual, escape_bps as u8);
                }
            }
        } else {
            writer.write_bits(param as u32, param_bits);
            for &residual in partition_residuals {
                let unsigned = signed_to_unsigned(residual);
                let quotient = unsigned >> param;
                let remainder = unsigned & ((1u32 << param) - 1);
                // Fuse the unary prefix (q zeros + terminating 1) with the k-bit
                // remainder into a single write_bits call.
                //
                // The Rice code for `unsigned` with parameter k is:
                //   q zero bits | 1 bit | k bits of remainder
                //
                // As an integer with (q+1+k) bits: (1<<k | rem), which has exactly
                // k+1 significant bits.  The leading q zeros are implicit when we
                // specify total_bits = q+1+k.
                let total_bits = quotient + 1 + param as u32;
                if total_bits <= 32 {
                    writer.write_bits((1u32 << param) | remainder, total_bits as u8);
                } else {
                    // Very large quotient (near-escape residual) — write in chunks.
                    let mut q = quotient;
                    while q >= 32 {
                        writer.write_bits(0, 32);
                        q -= 32;
                    }
                    writer.write_bits((1u32 << param) | remainder, (q + 1 + param as u32) as u8);
                }
            }
        }
    }

    Ok(writer.bits_written() - start_bits)
}

/// Find the optimal partition order for encoding (kept for backward compat).
pub fn find_optimal_partition_order(
    residuals: &[i32],
    block_size: usize,
    predictor_order: usize,
    method: RiceMethod,
    min_order: u8,
    max_order: u8,
    exhaustive: bool,
) -> u8 {
    plan_residual_coding(residuals, block_size, predictor_order, method, min_order, max_order, exhaustive).order
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

    #[test]
    fn test_rice_roundtrip_17bps_residuals() {
        use crate::flac::bitstream::{BitReader, BitWriter};
        // First few residuals from the failing 17-bps Fixed(3) test
        let residuals: Vec<i32> = vec![2, -3, 4, -5, 3, -1, 0, 0, 1, -2];
        let block_size = 10 + 3; // 3 warmup + 10 residuals
        let predictor_order = 3;
        let partition_order = 0;

        let mut writer = BitWriter::new();
        encode_residual(&mut writer, &residuals, RiceMethod::Rice, partition_order, block_size, predictor_order)
            .expect("encode failed");
        let data = writer.finish();

        eprintln!("Encoded bytes: {:?}", &data[..data.len().min(6)]);

        let mut reader = BitReader::new(&data);
        let decoded = decode_residual(&mut reader, RiceMethod::Rice, partition_order, block_size, predictor_order)
            .expect("decode failed");

        eprintln!("Encoded residuals: {:?}", residuals);
        eprintln!("Decoded residuals: {:?}", decoded);
        assert_eq!(decoded, residuals, "Rice roundtrip failed");
    }
