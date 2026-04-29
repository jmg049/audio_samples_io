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
    fixed_predictor_residual, fixed_predictor_restore, fixed_predictor_restore_into,
    lpc_predictor_residual, lpc_predictor_restore, lpc_predictor_restore_into,
};
use crate::flac::rice::{RiceMethod, RicePlan, decode_residual, encode_residual_planned, plan_residual_coding};

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
    pub const fn from_header(value: u8) -> Result<Self, FlacError> {
        match value {
            0b000000 => Ok(SubframeType::Constant),
            0b000001 => Ok(SubframeType::Verbatim),
            0b001000..=0b001100 => Ok(SubframeType::Fixed((value & 0x07) as u8)),
            0b100000..=0b111111 => Ok(SubframeType::Lpc(((value & 0x1F) + 1) as u8)),
            _ => Err(FlacError::InvalidSubframeType(value)),
        }
    }

    /// Convert to 6-bit header value.
    pub const fn to_header(self) -> u8 {
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

    let mut samples = match header.subframe_type {
        SubframeType::Constant => decode_constant(reader, block_size, effective_bits)?,
        SubframeType::Verbatim => decode_verbatim(reader, block_size, effective_bits)?,
        SubframeType::Fixed(order) => {
            decode_fixed(reader, block_size, effective_bits, order as usize)?
        }
        SubframeType::Lpc(order) => decode_lpc(reader, block_size, effective_bits, order as usize)?,
    };

    // Restore wasted bits (shift in-place to avoid a second allocation)
    if header.wasted_bits > 0 {
        for s in samples.iter_mut() {
            *s <<= header.wasted_bits;
        }
    }

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

/// Decode a FLAC subframe directly into a caller-provided buffer (zero-allocation path).
///
/// On return `out` contains exactly `block_size` samples. Wasted bits have
/// already been shifted back in. The buffer is extended (not replaced) so
/// callers can pre-allocate it once and reuse it across frames.
///
/// This eliminates the intermediate `Subframe { samples: Vec<i32> }` allocation,
/// the separate residual Vec, and the restore-output Vec — all replaced by writes
/// into the single pre-allocated `out` buffer.
pub(crate) fn decode_subframe_into(
    reader: &mut BitReader,
    block_size: usize,
    bits_per_sample: u8,
    out: &mut Vec<i32>,
) -> Result<(), FlacError> {
    let header = SubframeHeader::decode(reader)?;
    let effective_bits = bits_per_sample - header.wasted_bits;
    let start = out.len(); // samples written by this call start here

    match header.subframe_type {
        SubframeType::Constant => {
            let value = reader.read_bits_signed(effective_bits)?;
            out.resize(start + block_size, value);
        }
        SubframeType::Verbatim => {
            out.reserve(block_size);
            for _ in 0..block_size {
                out.push(reader.read_bits_signed(effective_bits)?);
            }
        }
        SubframeType::Fixed(order) => {
            let order = order as usize;
            // Warm-up samples
            out.reserve(block_size);
            for _ in 0..order {
                out.push(reader.read_bits_signed(effective_bits)?);
            }
            // Residuals appended directly after warmup
            let method_code = reader.read_bits(2)? as u8;
            let method = match method_code {
                0b00 => RiceMethod::Rice,
                0b01 => RiceMethod::Rice2,
                _ => return Err(FlacError::InvalidSubframeType(method_code)),
            };
            let partition_order = reader.read_bits(4)? as u8;
            // Decode residuals into the tail of `out` (after the warmup we just pushed)
            // by temporarily taking a sub-view.  We do this by decoding into a side
            // buffer and then extending, OR by treating the out tail as the residual buf.
            // The cleanest zero-alloc path: use out directly with careful unsafe writes.
            let res_count = block_size - order;
            let res_start = out.len();
            out.reserve(res_count);
            // Temporarily treat out[res_start..] as uninitialized storage for residuals
            // then do the in-place restore. We extend out's len as we decode.
            let num_partitions = 1usize << partition_order;
            let param_bits = method.param_bits();
            let escape_code = method.escape_code();
            let partition_samples = block_size / num_partitions;
            if partition_samples < order {
                return Err(crate::flac::error::FlacError::RicePartitionOverflow);
            }
            let ptr = out.as_mut_ptr();
            let mut write_idx = res_start;
            for partition in 0..num_partitions {
                let param = reader.read_bits(param_bits)? as u8;
                let samples = if partition == 0 { partition_samples - order } else { partition_samples };
                if param == escape_code {
                    let bps = reader.read_bits(5)? as u8;
                    if bps > 32 { return Err(crate::flac::error::FlacError::RiceEscapeBitsTooLarge { bits: bps }); }
                    for _ in 0..samples {
                        let v = if bps == 0 { 0 } else { reader.read_bits_signed(bps)? };
                        unsafe { ptr.add(write_idx).write(v); }
                        write_idx += 1;
                    }
                } else if param == 0 {
                    for _ in 0..samples {
                        let v = crate::flac::rice::unsigned_to_signed(reader.read_unary()?);
                        unsafe { ptr.add(write_idx).write(v); }
                        write_idx += 1;
                    }
                } else {
                    for _ in 0..samples {
                        let v = crate::flac::rice::unsigned_to_signed(reader.read_rice_unsigned(param)?);
                        unsafe { ptr.add(write_idx).write(v); }
                        write_idx += 1;
                    }
                }
            }
            unsafe { out.set_len(write_idx); }
            // In-place restore: out[start..start+order]=warmup, out[start+order..]=residuals
            fixed_predictor_restore_into(out, start, order, block_size)?;
        }
        SubframeType::Lpc(order) => {
            let order = order as usize;
            out.reserve(block_size);
            // Warm-up samples
            for _ in 0..order {
                out.push(reader.read_bits_signed(effective_bits)?);
            }
            // QLP coefficients — stack-allocated to avoid a heap alloc per subframe
            let qlp_precision_code = reader.read_bits(4)? as u8;
            if qlp_precision_code == 0b1111 {
                return Err(FlacError::InvalidQlpPrecision { precision: qlp_precision_code });
            }
            let qlp_precision = qlp_precision_code + 1;
            let qlp_shift = reader.read_bits_signed(5)? as i8;
            if qlp_shift < 0 {
                return Err(FlacError::InvalidLpcShift { shift: qlp_shift });
            }
            let mut qlp_buf = [0i32; 32];

            for qlp_item in qlp_buf.iter_mut().take(order) {
                *qlp_item = reader.read_bits_signed(qlp_precision)?;
            }
            
            let qlp_coeffs = &qlp_buf[..order];
            // Decode residuals inline (same pattern as FIXED above)
            let method_code = reader.read_bits(2)? as u8;
            let method = match method_code {
                0b00 => RiceMethod::Rice,
                0b01 => RiceMethod::Rice2,
                _ => return Err(FlacError::InvalidSubframeType(method_code)),
            };
            let partition_order = reader.read_bits(4)? as u8;
            let res_count = block_size - order;
            let res_start = out.len();
            out.reserve(res_count);
            let num_partitions = 1usize << partition_order;
            let param_bits = method.param_bits();
            let escape_code = method.escape_code();
            let partition_samples = block_size / num_partitions;
            if partition_samples < order {
                return Err(crate::flac::error::FlacError::RicePartitionOverflow);
            }
            let ptr = out.as_mut_ptr();
            let mut write_idx = res_start;
            for partition in 0..num_partitions {
                let param = reader.read_bits(param_bits)? as u8;
                let samples = if partition == 0 { partition_samples - order } else { partition_samples };
                if param == escape_code {
                    let bps = reader.read_bits(5)? as u8;
                    if bps > 32 { return Err(crate::flac::error::FlacError::RiceEscapeBitsTooLarge { bits: bps }); }
                    for _ in 0..samples {
                        let v = if bps == 0 { 0 } else { reader.read_bits_signed(bps)? };
                        unsafe { ptr.add(write_idx).write(v); }
                        write_idx += 1;
                    }
                } else if param == 0 {
                    for _ in 0..samples {
                        let v = crate::flac::rice::unsigned_to_signed(reader.read_unary()?);
                        unsafe { ptr.add(write_idx).write(v); }
                        write_idx += 1;
                    }
                } else {
                    for _ in 0..samples {
                        let v = crate::flac::rice::unsigned_to_signed(reader.read_rice_unsigned(param)?);
                        unsafe { ptr.add(write_idx).write(v); }
                        write_idx += 1;
                    }
                }
            }
            unsafe { out.set_len(write_idx); }
            lpc_predictor_restore_into(out, start, order, qlp_coeffs, qlp_shift)?;
        }
    }

    // Restore wasted bits in-place
    if header.wasted_bits > 0 {
        for s in out[start..].iter_mut() {
            *s <<= header.wasted_bits;
        }
    }

    Ok(())
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

/// Write a FIXED subframe (warmup + residuals) directly to `writer`.
fn write_fixed_into(
    writer: &mut BitWriter,
    samples: &[i32],
    bits_per_sample: u8,
    wasted_bits: u8,
    order: usize,
    residuals: &[i32],
    plan: &RicePlan,
    block_size: usize,
) -> Result<(), FlacError> {
    SubframeHeader { subframe_type: SubframeType::Fixed(order as u8), wasted_bits }
        .encode(writer);
    for &sample in &samples[..order] {
        writer.write_bits_signed(sample, bits_per_sample);
    }
    writer.write_bits(0b00, 2);
    writer.write_bits(plan.order as u32, 4);
    encode_residual_planned(writer, residuals, RiceMethod::Rice,
        plan.order, plan.params_slice(), block_size, order)?;
    Ok(())
}

/// Write an LPC subframe (warmup + coefficients + residuals) directly to `writer`.
#[allow(clippy::too_many_arguments)]
fn write_lpc_into(
    writer: &mut BitWriter,
    samples: &[i32],
    bits_per_sample: u8,
    wasted_bits: u8,
    order: usize,
    qlp_coeffs: &[i32],
    qlp_shift: i8,
    qlp_precision: u8,
    residuals: &[i32],
    plan: &RicePlan,
    block_size: usize,
) -> Result<(), FlacError> {
    SubframeHeader { subframe_type: SubframeType::Lpc(order as u8), wasted_bits }
        .encode(writer);
    for &sample in &samples[..order] {
        writer.write_bits_signed(sample, bits_per_sample);
    }
    writer.write_bits((qlp_precision - 1) as u32, 4);
    writer.write_bits_signed(qlp_shift as i32, 5);
    for &coeff in qlp_coeffs {
        writer.write_bits_signed(coeff, qlp_precision);
    }
    writer.write_bits(0b00, 2);
    writer.write_bits(plan.order as u32, 4);
    encode_residual_planned(writer, residuals, RiceMethod::Rice,
        plan.order, plan.params_slice(), block_size, order)?;
    Ok(())
}

/// Core subframe encoder: selects the best type and writes it directly to `writer`.
///
/// Used by the sequential frame encoder to avoid per-subframe `BitWriter` allocations.
/// Returns the `SubframeType` selected.
#[allow(clippy::type_complexity)]
pub(crate) fn encode_subframe_into(
    writer: &mut BitWriter,
    samples: &[i32],
    bits_per_sample: u8,
    max_lpc_order: usize,
    qlp_precision: u8,
    min_partition_order: u8,
    max_partition_order: u8,
    exhaustive_rice: bool,
) -> Result<SubframeType, FlacError> {
    let block_size = samples.len();

    let wasted_bits = compute_wasted_bits(samples);
    let shifted_buf: Vec<i32>;
    let effective_samples: &[i32] = if wasted_bits > 0 {
        shifted_buf = samples.iter().map(|&s| s >> wasted_bits).collect();
        &shifted_buf
    } else {
        samples
    };
    let effective_bits = bits_per_sample - wasted_bits;

    // Constant check
    if !effective_samples.is_empty() {
        let first = effective_samples[0];
        if effective_samples.iter().all(|&s| s == first) {
            SubframeHeader { subframe_type: SubframeType::Constant, wasted_bits }.encode(writer);
            writer.write_bits_signed(first, effective_bits);
            return Ok(SubframeType::Constant);
        }
    }

    let verbatim_bits = 8 + block_size * effective_bits as usize;
    let header_overhead = 8 + wasted_bits as usize;

    // --- Phase 1: Prepare residuals + rice plan for each candidate, no writing ---

    // Fixed candidate
    let max_fixed_order = 4.min(block_size.saturating_sub(1));
    let fixed_state: Option<(usize, Vec<i32>, RicePlan, usize)> = {
        let order = if max_fixed_order > 0 {
            Some(crate::flac::lpc::find_best_fixed_order(effective_samples).min(max_fixed_order))
        } else if block_size > 1 {
            Some(0)
        } else {
            None
        };
        order.and_then(|o| {
            fixed_predictor_residual(effective_samples, o).ok().map(|res| {
                let plan = plan_residual_coding(&res, block_size, o, RiceMethod::Rice,
                    min_partition_order, max_partition_order, exhaustive_rice);
                let est = header_overhead + o * effective_bits as usize + 6 + plan.est_bits;
                (o, res, plan, est)
            })
        })
    };

    // LPC candidate (only if enabled and block is large enough)
    let lpc_state: Option<(usize, Vec<i32>, i8, Vec<i32>, RicePlan, usize)> =
        if max_lpc_order > 0 && block_size > max_lpc_order {
            crate::flac::lpc::find_best_lpc_order(
                effective_samples, max_lpc_order, qlp_precision, false,
            ).ok().and_then(|(order, qlp_coeffs, qlp_shift)| {
                if order == 0 { return None; }
                lpc_predictor_residual(effective_samples, &qlp_coeffs, qlp_shift).ok()
                    .map(|res| {
                        let plan = plan_residual_coding(&res, block_size, order, RiceMethod::Rice,
                            min_partition_order, max_partition_order, exhaustive_rice);
                        let est = header_overhead
                            + order * effective_bits as usize
                            + 4 + 5 + order * qlp_precision as usize
                            + 6 + plan.est_bits;
                        (order, qlp_coeffs, qlp_shift, res, plan, est)
                    })
            })
        } else {
            None
        };

    // --- Phase 2: Pick winner by estimated bits, write only the winner ---

    let fixed_est = fixed_state.as_ref().map_or(usize::MAX, |s| s.3);
    let lpc_est   = lpc_state.as_ref().map_or(usize::MAX, |s| s.5);
    let best_est  = fixed_est.min(lpc_est);

    if best_est >= verbatim_bits {
        SubframeHeader { subframe_type: SubframeType::Verbatim, wasted_bits }.encode(writer);
        for &sample in effective_samples {
            writer.write_bits_signed(sample, effective_bits);
        }
        return Ok(SubframeType::Verbatim);
    }

    if lpc_est <= fixed_est {
        if let Some((order, qlp_coeffs, qlp_shift, residuals, plan, _)) = lpc_state {
            write_lpc_into(writer, effective_samples, effective_bits, wasted_bits,
                order, &qlp_coeffs, qlp_shift, qlp_precision, &residuals, &plan, block_size)?;
            return Ok(SubframeType::Lpc(order as u8));
        }
    }

    if let Some((order, residuals, plan, _)) = fixed_state {
        write_fixed_into(writer, effective_samples, effective_bits, wasted_bits,
            order, &residuals, &plan, block_size)?;
        return Ok(SubframeType::Fixed(order as u8));
    }

    // Fallback: verbatim
    SubframeHeader { subframe_type: SubframeType::Verbatim, wasted_bits }.encode(writer);
    for &sample in effective_samples {
        writer.write_bits_signed(sample, effective_bits);
    }
    Ok(SubframeType::Verbatim)
}

/// Encode a subframe with automatic type selection.
///
/// Thin wrapper around [`encode_subframe_into`] for the parallel encoder path and
/// external callers. For the sequential path, use `encode_subframe_into` directly.
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
    let capacity = (block_size * bits_per_sample as usize).div_ceil(8) + 16;
    let mut writer = BitWriter::with_capacity(capacity);
    let subframe_type = encode_subframe_into(
        &mut writer, samples, bits_per_sample, max_lpc_order, qlp_precision,
        min_partition_order, max_partition_order, exhaustive_rice,
    )?;
    let bits = writer.bits_written();
    let data = writer.finish();
    Ok(EncodedSubframe { subframe_type, bits, data })
}

/// Compute number of wasted bits (common trailing zeros).
fn compute_wasted_bits(samples: &[i32]) -> u8 {
    if samples.is_empty() {
        return 0;
    }

    let combined = samples.iter().fold(0i32, |acc, &s| acc | s);

    if combined == 0 {
        return 0;
    }

    combined.trailing_zeros() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_encode_constant(samples: &[i32], bits_per_sample: u8, wasted_bits: u8) -> Option<EncodedSubframe> {
        if samples.is_empty() { return None; }
        let first = samples[0];
        if !samples.iter().all(|&s| s == first) { return None; }
        let mut writer = BitWriter::new();
        SubframeHeader { subframe_type: SubframeType::Constant, wasted_bits }.encode(&mut writer);
        writer.write_bits_signed(first, bits_per_sample);
        let bits = writer.bits_written();
        let data = writer.finish();
        Some(EncodedSubframe { subframe_type: SubframeType::Constant, bits, data })
    }

    fn encode_verbatim(samples: &[i32], bits_per_sample: u8, wasted_bits: u8) -> EncodedSubframe {
        let capacity = (samples.len() * bits_per_sample as usize + 7) / 8 + 8;
        let mut writer = BitWriter::with_capacity(capacity);
        SubframeHeader { subframe_type: SubframeType::Verbatim, wasted_bits }.encode(&mut writer);
        for &sample in samples { writer.write_bits_signed(sample, bits_per_sample); }
        let bits = writer.bits_written();
        let data = writer.finish();
        EncodedSubframe { subframe_type: SubframeType::Verbatim, bits, data }
    }

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
            encode_subframe(&samples, bits_per_sample, 0, 0, 0, 4, false).expect("Encode failed");

        assert!(matches!(encoded.subframe_type, SubframeType::Fixed(_)));

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

    #[test]
    fn test_fixed_subframe_17bps_roundtrip() {
        use std::f64::consts::PI;
        // Simulate side channel (17 bps): sine - cosine at 110Hz/165Hz
        let n = 4096;
        let sr = 44100.0f64;
        let amp = 32767.0;
        let samples: Vec<i32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                let l = (amp * (2.0 * PI * 110.0 * t).sin()) as i32;
                let r = (amp * (2.0 * PI * 165.0 * t).cos()) as i32;
                l - r // side channel
            })
            .collect();

        let bits_per_sample = 17u8;
        let encoded = encode_subframe(&samples, bits_per_sample, 0, 12, 0, 6, false)
            .expect("encode failed");

        let mut reader = BitReader::new(&encoded.data);
        let decoded = decode_subframe(&mut reader, n, bits_per_sample)
            .expect("decode failed");

        assert_eq!(decoded.samples.len(), n);
        for (i, (&orig, &dec)) in samples.iter().zip(decoded.samples.iter()).enumerate() {
            assert_eq!(orig, dec, "mismatch at sample {i}: orig={orig} dec={dec}");
        }
    }
