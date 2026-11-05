//! FLAC frame encoder.
//!
//! A [`FrameEncoder`] serves a whole stream and keeps its buffers between frames, so encoding
//! allocates once however long the stream is. Each frame goes through three steps:
//!
//! 1. For two-channel audio, choose how the channels are coded (independent, left-side,
//!    right-side or mid-side).
//! 2. For each coded signal, choose a model: constant, verbatim, fixed predictor or LPC, each
//!    costed with its best Rice partitioning. LPC coefficients come from windowed
//!    autocorrelation; higher levels add partial windows, which suit signals that change within
//!    a block.
//! 3. Write the chosen models.
//!
//! The whole frame is compiled twice, for the baseline target and for AVX2 with FMA, and the
//! choice is made once when the encoder is built.

use crate::flac::ChannelAssignment;
use crate::flac::CompressionLevel;
use crate::flac::bitstream::BitWriter;
use crate::flac::crc::{Crc8, Crc16};
use crate::flac::error::FlacError;
use crate::flac::frame::encode_frame_header;
use crate::flac::lpc;
use crate::flac::subframe::{SubframeHeader, SubframeType};

/// Largest supported partition order. Partition order 8 means 256 partitions.
const MAX_PARTITION_ORDER: u8 = 8;
const MAX_PARTITIONS: usize = 1 << MAX_PARTITION_ORDER;
const MAX_LPC_ORDER: usize = crate::flac::constants::MAX_LPC_ORDER;

/// Marks a partition of zeros in [`RiceChoice::params`]: escape code with zero-width samples.
const ZERO_PARTITION: u8 = 0xFF;

/// How to choose the coding of two channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StereoSearch {
    /// Always code the channels independently.
    Independent,
    /// Estimate the cost of each coding from fixed-predictor error sums and encode the cheapest.
    Estimate,
    /// Plan all four signals and keep the cheapest pair.
    Full,
}

/// Parameters that stay the same for every frame of a stream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EncoderConfig {
    pub max_lpc_order: usize,
    /// Quantised LPC coefficient precision in bits.
    pub qlp_precision: u8,
    pub min_partition_order: u8,
    pub max_partition_order: u8,
    pub stereo: StereoSearch,
    /// Evaluate neighbouring Rice parameters as well as the one implied by the partition's mean.
    pub wide_rice: bool,
    /// Largest number of parts a block's analysis window is divided into. 1 is the whole block.
    pub window_parts: u8,
    /// Candidates from the partial windows that get a full evaluation.
    pub window_trials: u8,
}

impl EncoderConfig {
    /// Parameters for `level` at the given sample width and block size.
    pub(crate) const fn for_level(level: CompressionLevel, bits_per_sample: u8, block_size: u32) -> Self {
        let (min_partition_order, max_partition_order) = level.rice_partition_order_range();
        Self {
            max_lpc_order: level.max_lpc_order() as usize,
            qlp_precision: auto_qlp_precision(bits_per_sample, block_size),
            min_partition_order,
            max_partition_order,
            stereo: match (level.try_mid_side(), level.level()) {
                (false, _) => StereoSearch::Independent,
                (true, 8) => StereoSearch::Full,
                (true, _) => StereoSearch::Estimate,
            },
            wide_rice: level.exhaustive_rice_search(),
            window_parts: level.window_parts(),
            window_trials: match level.level() {
                0..=5 => 0,
                6 | 7 => 2,
                _ => 4,
            },
        }
    }
}

/// Coefficient precision for a sample width and block size. A larger block spends the bits of
/// its coefficients over more samples, and a wider sample leaves more signal to predict.
pub(crate) const fn auto_qlp_precision(bits_per_sample: u8, block_size: u32) -> u8 {
    if bits_per_sample <= 16 {
        match block_size {
            0..=192 => 7,
            193..=384 => 8,
            385..=576 => 9,
            577..=1152 => 10,
            1153..=2304 => 11,
            2305..=4608 => 12,
            _ => 13,
        }
    } else {
        match block_size {
            0..=384 => 13,
            385..=1152 => 14,
            _ => 15,
        }
    }
}

/// The partition order and Rice parameter of each partition for one signal.
#[derive(Clone, Copy)]
struct RiceChoice {
    order: u8,
    /// Uses 5-bit parameters.
    extended: bool,
    params: [u8; MAX_PARTITIONS],
}

impl RiceChoice {
    const fn empty() -> Self {
        Self {
            order: 0,
            extended: false,
            params: [0; MAX_PARTITIONS],
        }
    }
}

#[derive(Clone, Copy)]
enum Model {
    Constant(i32),
    Verbatim,
    Fixed(u8),
    Lpc {
        order: u8,
        shift: i8,
        precision: u8,
        coeffs: [i32; MAX_LPC_ORDER],
    },
}

/// Buffers and outcome of planning one signal.
struct Slot {
    /// The signal with its wasted bits removed, when it has any.
    shifted: Vec<i32>,
    /// Residuals of the best model so far.
    residual: Vec<i32>,
    /// Residuals of the model being evaluated.
    trial: Vec<i32>,
    windowed: Vec<f64>,
    /// Sums of zigzag residuals per partition, finest order first.
    sums: Vec<u64>,
    model: Model,
    wasted: u8,
    /// Sample width after removing wasted bits.
    width: u8,
    rice: RiceChoice,
    /// Estimated size of the subframe.
    bits: usize,
}

impl Slot {
    fn new(block_size: usize) -> Self {
        Self {
            shifted: Vec::with_capacity(block_size),
            residual: Vec::with_capacity(block_size),
            trial: Vec::with_capacity(block_size),
            windowed: Vec::with_capacity(block_size),
            sums: vec![0; 2 * MAX_PARTITIONS],
            model: Model::Verbatim,
            wasted: 0,
            width: 0,
            rice: RiceChoice::empty(),
            bits: 0,
        }
    }
}

/// The analysis window for one block length.
struct Windows {
    block_size: usize,
    root: Vec<f64>,
}

impl Windows {
    const fn empty() -> Self {
        Self {
            block_size: 0,
            root: Vec::new(),
        }
    }

    fn prepare(&mut self, block_size: usize) {
        if self.block_size != block_size {
            self.block_size = block_size;
            tukey(&mut self.root, block_size, 0.5);
        }
    }
}

/// Fill `window` with a Tukey window of `len` points whose tapers span `p` of its length.
fn tukey(window: &mut Vec<f64>, len: usize, p: f64) {
    window.clear();
    window.resize(len, 1.0);
    let np = (p / 2.0 * len as f64) as usize;
    if np > 1 {
        let np1 = np - 1;
        for n in 0..=np1 {
            let w = 0.5 - 0.5 * (std::f64::consts::PI * n as f64 / np1 as f64).cos();
            window[n] = w;
            window[len - 1 - n] = w;
        }
    }
}

/// Encodes frames of one stream, reusing its buffers between them.
pub(crate) struct FrameEncoder {
    config: EncoderConfig,
    bits: BitWriter,
    mid: Vec<i32>,
    side: Vec<i32>,
    /// One slot per channel; two more when all four stereo signals are planned.
    slots: Vec<Slot>,
    windows: Windows,
    #[cfg(target_arch = "x86_64")]
    use_avx2: bool,
}

impl FrameEncoder {
    pub(crate) fn new(config: EncoderConfig, channels: usize, block_size: usize) -> Self {
        let stereo = channels == 2 && config.stereo != StereoSearch::Independent;
        let slot_count = if stereo { 4 } else { channels };
        Self {
            config,
            bits: BitWriter::with_capacity(block_size * channels * 2),
            mid: Vec::with_capacity(if stereo { block_size } else { 0 }),
            side: Vec::with_capacity(if stereo { block_size } else { 0 }),
            slots: (0..slot_count).map(|_| Slot::new(block_size)).collect(),
            windows: Windows::empty(),
            #[cfg(target_arch = "x86_64")]
            use_avx2: is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("bmi1")
                && is_x86_feature_detected!("bmi2")
                && is_x86_feature_detected!("lzcnt"),
        }
    }

    /// Encode one frame and return its bytes, valid until the next call.
    pub(crate) fn encode<S: AsRef<[i32]>>(
        &mut self,
        channels: &[S],
        bits_per_sample: u8,
        sample_rate: u32,
        frame_number: u64,
    ) -> Result<&[u8], FlacError> {
        #[cfg(target_arch = "x86_64")]
        if self.use_avx2 {
            // SAFETY: `new` found the features this function enables.
            unsafe { self.encode_avx2(channels, bits_per_sample, sample_rate, frame_number)? };
            return Ok(self.bits.data());
        }
        self.encode_generic::<S, false>(channels, bits_per_sample, sample_rate, frame_number)?;
        Ok(self.bits.data())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,bmi1,bmi2,lzcnt")]
    unsafe fn encode_avx2<S: AsRef<[i32]>>(
        &mut self,
        channels: &[S],
        bits_per_sample: u8,
        sample_rate: u32,
        frame_number: u64,
    ) -> Result<(), FlacError> {
        self.encode_generic::<S, true>(channels, bits_per_sample, sample_rate, frame_number)
    }

    #[inline(always)]
    fn encode_generic<S: AsRef<[i32]>, const AVX2: bool>(
        &mut self,
        channels: &[S],
        bits_per_sample: u8,
        sample_rate: u32,
        frame_number: u64,
    ) -> Result<(), FlacError> {
        let Some(first) = channels.first() else {
            return Err(FlacError::InvalidChannelCount { channels: 0 });
        };
        let block_size = first.as_ref().len();
        self.windows.prepare(block_size);
        let config = self.config;
        let windows = &self.windows;

        let assignment = if channels.len() == 2 && config.stereo != StereoSearch::Independent {
            let left = channels[0].as_ref();
            let right = channels[1].as_ref();
            decorrelate(left, right, &mut self.mid, &mut self.side);
            let (mid, side) = (self.mid.as_slice(), self.side.as_slice());
            let width = bits_per_sample;
            let [slot_left, slot_right, slot_mid, slot_side] = self.slots.as_mut_slice() else {
                unreachable!("stereo encoders hold four slots");
            };
            match config.stereo {
                StereoSearch::Full => {
                    plan_signal::<AVX2>(slot_left, left, width, &config, windows);
                    plan_signal::<AVX2>(slot_right, right, width, &config, windows);
                    plan_signal::<AVX2>(slot_mid, mid, width, &config, windows);
                    plan_signal::<AVX2>(slot_side, side, width + 1, &config, windows);
                    let (l, r, m, s) = (slot_left.bits, slot_right.bits, slot_mid.bits, slot_side.bits);
                    let best = (l + r).min(l + s).min(s + r).min(m + s);
                    if best == l + r {
                        ChannelAssignment::Independent(2)
                    } else if best == l + s {
                        ChannelAssignment::LeftSide
                    } else if best == s + r {
                        ChannelAssignment::RightSide
                    } else {
                        ChannelAssignment::MidSide
                    }
                },
                _ => {
                    let l = estimate_bits::<AVX2>(left, width);
                    let r = estimate_bits::<AVX2>(right, width);
                    let m = estimate_bits::<AVX2>(mid, width);
                    let s = estimate_bits::<AVX2>(side, width + 1);
                    let best = (l + r).min(l + s).min(s + r).min(m + s);
                    let assignment = if best == l + r {
                        ChannelAssignment::Independent(2)
                    } else if best == l + s {
                        ChannelAssignment::LeftSide
                    } else if best == s + r {
                        ChannelAssignment::RightSide
                    } else {
                        ChannelAssignment::MidSide
                    };
                    let (first_signal, first_width): (&[i32], u8) = match assignment {
                        ChannelAssignment::RightSide => (side, width + 1),
                        ChannelAssignment::MidSide => (mid, width),
                        _ => (left, width),
                    };
                    let (second_signal, second_width): (&[i32], u8) = match assignment {
                        ChannelAssignment::Independent(_) | ChannelAssignment::RightSide => (right, width),
                        _ => (side, width + 1),
                    };
                    // Keep each signal in its own slot, as `Full` does.
                    let (first_slot, second_slot) = match assignment {
                        ChannelAssignment::Independent(_) => (slot_left, slot_right),
                        ChannelAssignment::LeftSide => (slot_left, slot_side),
                        ChannelAssignment::RightSide => (slot_side, slot_right),
                        ChannelAssignment::MidSide => (slot_mid, slot_side),
                    };
                    plan_signal::<AVX2>(first_slot, first_signal, first_width, &config, windows);
                    plan_signal::<AVX2>(second_slot, second_signal, second_width, &config, windows);
                    assignment
                },
            }
        } else {
            for (slot, channel) in self.slots.iter_mut().zip(channels) {
                plan_signal::<AVX2>(slot, channel.as_ref(), bits_per_sample, &config, windows);
            }
            ChannelAssignment::Independent(channels.len() as u8)
        };

        self.bits.clear();
        encode_frame_header(
            &mut self.bits,
            block_size as u32,
            sample_rate,
            assignment,
            bits_per_sample,
            frame_number,
        )?;
        self.bits.align_to_byte();
        let crc8 = Crc8::compute(self.bits.data());
        self.bits.write_bits(u32::from(crc8), 8);

        if channels.len() == 2 && config.stereo != StereoSearch::Independent {
            let left = channels[0].as_ref();
            let right = channels[1].as_ref();
            let (mid, side) = (self.mid.as_slice(), self.side.as_slice());
            let (slot_a, signal_a, slot_b, signal_b): (usize, &[i32], usize, &[i32]) = match assignment {
                ChannelAssignment::Independent(_) => (0, left, 1, right),
                ChannelAssignment::LeftSide => (0, left, 3, side),
                ChannelAssignment::RightSide => (3, side, 1, right),
                ChannelAssignment::MidSide => (2, mid, 3, side),
            };
            write_subframe(&mut self.bits, &self.slots[slot_a], signal_a);
            write_subframe(&mut self.bits, &self.slots[slot_b], signal_b);
        } else {
            for (slot, channel) in self.slots.iter().zip(channels) {
                write_subframe(&mut self.bits, slot, channel.as_ref());
            }
        }

        self.bits.align_to_byte();
        let crc16 = Crc16::compute(self.bits.data());
        self.bits.write_bits(u32::from(crc16), 16);
        self.bits.align_to_byte();
        Ok(())
    }
}

#[inline(always)]
fn decorrelate(left: &[i32], right: &[i32], mid: &mut Vec<i32>, side: &mut Vec<i32>) {
    mid.clear();
    side.clear();
    mid.extend(left.iter().zip(right).map(|(&l, &r)| (l + r) >> 1));
    side.extend(left.iter().zip(right).map(|(&l, &r)| l - r));
}

/// Cost in bits of coding `samples` with its best fixed predictor, from the error sums alone.
#[inline(always)]
fn estimate_bits<const AVX2: bool>(samples: &[i32], width: u8) -> u64 {
    let sums = fixed_error_sums(samples, width);
    let best = sums.iter().copied().min().unwrap_or(0);
    let n = samples.len().saturating_sub(4).max(1) as f64;
    if best == 0 {
        return 0;
    }
    // Bits per sample of a Laplacian source with this mean magnitude.
    let per_sample = (std::f64::consts::LN_2 * best as f64 / n).log2().max(0.0) + 1.0;
    (per_sample * n) as u64
}

/// Sums of absolute residuals of the fixed predictors of orders 0 to 4 over samples 4 onwards.
#[inline(always)]
fn fixed_error_sums(s: &[i32], width: u8) -> [u64; 5] {
    let mut totals = [0u64; 5];
    if s.len() <= 4 {
        // Too short to compare; order 0 wins.
        totals[1..].fill(u64::MAX);
        return totals;
    }
    let n = s.len() - 4;
    let (s0, s1, s2, s3, s4) = (&s[..n], &s[1..n + 1], &s[2..n + 2], &s[3..n + 3], &s[4..n + 4]);
    if width <= 17 {
        // Order 4 of a 17-bit signal is at most 21 bits, so 1024 of them fit in 32 bits.
        let mut start = 0;
        while start < n {
            let end = (start + 1024).min(n);
            let (mut e0, mut e1, mut e2, mut e3, mut e4) = (0u32, 0u32, 0u32, 0u32, 0u32);
            for i in start..end {
                let (a, b, c, d, e) = (s4[i], s3[i], s2[i], s1[i], s0[i]);
                let r1 = a.wrapping_sub(b);
                let r2 = r1.wrapping_sub(b.wrapping_sub(c));
                let r3 = a
                    .wrapping_sub(b.wrapping_mul(3))
                    .wrapping_add(c.wrapping_mul(3))
                    .wrapping_sub(d);
                let r4 = a
                    .wrapping_sub(b.wrapping_shl(2))
                    .wrapping_add(c.wrapping_mul(6))
                    .wrapping_sub(d.wrapping_shl(2))
                    .wrapping_add(e);
                e0 = e0.wrapping_add(a.unsigned_abs());
                e1 = e1.wrapping_add(r1.unsigned_abs());
                e2 = e2.wrapping_add(r2.unsigned_abs());
                e3 = e3.wrapping_add(r3.unsigned_abs());
                e4 = e4.wrapping_add(r4.unsigned_abs());
            }
            totals[0] += u64::from(e0);
            totals[1] += u64::from(e1);
            totals[2] += u64::from(e2);
            totals[3] += u64::from(e3);
            totals[4] += u64::from(e4);
            start = end;
        }
    } else {
        // Wider residuals overflow 32 bits sooner, so their low and high halves are summed
        // separately; 65536 values of 16 bits fit in 32.
        let mut start = 0;
        while start < n {
            let end = (start + 65536).min(n);
            let mut lo = [0u32; 5];
            let mut hi = [0u32; 5];
            for i in start..end {
                let (a, b, c, d, e) = (s4[i], s3[i], s2[i], s1[i], s0[i]);
                let r1 = a.wrapping_sub(b);
                let r2 = r1.wrapping_sub(b.wrapping_sub(c));
                let r3 = a
                    .wrapping_sub(b.wrapping_mul(3))
                    .wrapping_add(c.wrapping_mul(3))
                    .wrapping_sub(d);
                let r4 = a
                    .wrapping_sub(b.wrapping_shl(2))
                    .wrapping_add(c.wrapping_mul(6))
                    .wrapping_sub(d.wrapping_shl(2))
                    .wrapping_add(e);
                let v = [
                    a.unsigned_abs(),
                    r1.unsigned_abs(),
                    r2.unsigned_abs(),
                    r3.unsigned_abs(),
                    r4.unsigned_abs(),
                ];
                for o in 0..5 {
                    lo[o] = lo[o].wrapping_add(v[o] & 0xFFFF);
                    hi[o] = hi[o].wrapping_add(v[o] >> 16);
                }
            }
            for o in 0..5 {
                totals[o] += u64::from(lo[o]) + (u64::from(hi[o]) << 16);
            }
            start = end;
        }
    }
    totals
}

/// Residuals of the fixed predictor of `order`. `out` ends up with `s.len() - order` values.
#[inline(always)]
fn fixed_residual(s: &[i32], order: usize, out: &mut Vec<i32>) {
    out.clear();
    if s.len() <= order {
        return;
    }
    let n = s.len() - order;
    out.reserve(n);
    let dst = &mut out.spare_capacity_mut()[..n];
    match order {
        0 => {
            for (d, &a) in dst.iter_mut().zip(s) {
                d.write(a);
            }
        },
        1 => {
            for (d, (&a, &b)) in dst.iter_mut().zip(s[1..].iter().zip(s)) {
                d.write(a.wrapping_sub(b));
            }
        },
        2 => {
            for (i, d) in dst.iter_mut().enumerate() {
                let (a, b, c) = (s[i + 2], s[i + 1], s[i]);
                d.write(a.wrapping_sub(b.wrapping_shl(1)).wrapping_add(c));
            }
        },
        3 => {
            for (i, d) in dst.iter_mut().enumerate() {
                let (a, b, c, e) = (s[i + 3], s[i + 2], s[i + 1], s[i]);
                d.write(
                    a.wrapping_sub(b.wrapping_mul(3))
                        .wrapping_add(c.wrapping_mul(3))
                        .wrapping_sub(e),
                );
            }
        },
        _ => {
            for (i, d) in dst.iter_mut().enumerate() {
                let (a, b, c, e, f) = (s[i + 4], s[i + 3], s[i + 2], s[i + 1], s[i]);
                d.write(
                    a.wrapping_sub(b.wrapping_shl(2))
                        .wrapping_add(c.wrapping_mul(6))
                        .wrapping_sub(e.wrapping_shl(2))
                        .wrapping_add(f),
                );
            }
        },
    }
    // SAFETY: the loop above initialised the first `n` elements of the spare capacity.
    unsafe { out.set_len(n) };
}

/// Sum the zigzag values of a run of residuals.
#[inline(always)]
fn sum_zigzag(values: &[i32], narrow: bool) -> u64 {
    if narrow {
        // A 21-bit residual zigzags to 22 bits, so 1024 of them fit in 32 bits.
        let mut total = 0u64;
        for chunk in values.chunks(1024) {
            let mut s = 0u32;
            for &r in chunk {
                s = s.wrapping_add(((r << 1) ^ (r >> 31)) as u32);
            }
            total += u64::from(s);
        }
        total
    } else {
        let mut total = 0u64;
        for chunk in values.chunks(65536) {
            let (mut lo, mut hi) = (0u32, 0u32);
            for &r in chunk {
                let u = ((r << 1) ^ (r >> 31)) as u32;
                lo = lo.wrapping_add(u & 0xFFFF);
                hi = hi.wrapping_add(u >> 16);
            }
            total += u64::from(lo) + (u64::from(hi) << 16);
        }
        total
    }
}

/// Exact bits of the Rice codes of `values` at parameter `k`.
#[inline(always)]
fn exact_rice_bits(values: &[i32], k: u32) -> usize {
    let mut quotients = 0u64;
    for chunk in values.chunks(65536) {
        let (mut lo, mut hi) = (0u32, 0u32);
        for &r in chunk {
            let q = (((r << 1) ^ (r >> 31)) as u32) >> k;
            lo = lo.wrapping_add(q & 0xFFFF);
            hi = hi.wrapping_add(q >> 16);
        }
        quotients += u64::from(lo) + (u64::from(hi) << 16);
    }
    quotients as usize + values.len() * (1 + k as usize)
}

/// Best Rice parameter and its cost in bits for a partition, parameter field excluded.
#[inline(always)]
fn partition_cost(sum: u64, count: usize, limit: u8, wide: bool) -> (u8, usize) {
    #[inline(always)]
    fn cost(sum: u64, count: usize, k: u32) -> usize {
        // Every sample has a stop bit and k remainder bits. Its quotient is on average half a
        // unit below sum >> k, once k is above zero.
        let quotients = (sum >> k) as usize;
        let rounding = if k == 0 { 0 } else { count / 2 };
        count * (1 + k as usize) + quotients.saturating_sub(rounding)
    }
    if count == 0 {
        return (0, 0);
    }
    let mean = sum / count as u64;
    let k = if mean == 0 { 0 } else { 63 - mean.leading_zeros() };
    let k = k.min(u32::from(limit));
    let mut best = (k as u8, cost(sum, count, k));
    if k < u32::from(limit) {
        let c = cost(sum, count, k + 1);
        if c < best.1 {
            best = ((k + 1) as u8, c);
        }
    }
    if wide && k > 0 {
        let c = cost(sum, count, k - 1);
        if c < best.1 {
            best = ((k - 1) as u8, c);
        }
    }
    best
}

/// Choose the partition order and parameters for `residual` and return the total bit cost,
/// including the 6 bits of the method and partition-order fields.
#[inline(always)]
fn plan_rice(
    residual: &[i32],
    block_size: usize,
    predictor_order: usize,
    width: u8,
    config: &EncoderConfig,
    sums: &mut [u64],
    out: &mut RiceChoice,
) -> usize {
    let limit: u8 = if width > 17 { 30 } else { 14 };
    let min_order = config.min_partition_order.min(MAX_PARTITION_ORDER);
    let mut max_order = config.max_partition_order.min(MAX_PARTITION_ORDER);
    while max_order > 0 && ((block_size >> max_order) <= predictor_order || block_size & ((1 << max_order) - 1) != 0) {
        max_order -= 1;
    }
    let min_order = min_order.min(max_order);
    let narrow = width <= 17;

    // Finest partitions from the residuals; each coarser order from the one below it. `sums`
    // holds order max at [0..2^max], order max-1 at [2^max..2^max + 2^(max-1)], and so on.
    let leaves = 1usize << max_order;
    let leaf_size = block_size >> max_order;
    let mut pos = 0usize;
    for (p, sum) in sums[..leaves].iter_mut().enumerate() {
        let count = if p == 0 { leaf_size - predictor_order } else { leaf_size };
        *sum = sum_zigzag(&residual[pos..pos + count], narrow);
        pos += count;
    }
    let mut base = 0usize;
    let mut size = leaves;
    for _ in min_order..max_order {
        let (done, rest) = sums[base..].split_at_mut(size);
        for (merged, pair) in rest[..size / 2].iter_mut().zip(done.chunks_exact(2)) {
            *merged = pair[0] + pair[1];
        }
        base += size;
        size /= 2;
    }

    let mut best_bits = usize::MAX;
    let mut base = 0usize;
    let mut order = max_order;
    loop {
        let parts = 1usize << order;
        let part_size = block_size >> order;
        let mut bits = 0usize;
        let mut extended = false;
        let mut params = [0u8; MAX_PARTITIONS];
        for (p, (&sum, param)) in sums[base..base + parts].iter().zip(&mut params).enumerate() {
            let count = if p == 0 { part_size - predictor_order } else { part_size };
            let (k, cost) = partition_cost(sum, count, limit, config.wide_rice);
            if sum == 0 && count > 9 {
                // All zeros: the escape code with zero-width samples takes only the 5-bit width.
                *param = ZERO_PARTITION;
                bits += 5;
            } else {
                *param = k;
                extended |= k > 14;
                bits += cost;
            }
        }
        bits += parts * if extended { 5 } else { 4 };
        if bits < best_bits {
            best_bits = bits;
            out.order = order;
            out.extended = extended;
            out.params[..parts].copy_from_slice(&params[..parts]);
        }
        if order == min_order {
            break;
        }
        base += parts;
        order -= 1;
    }
    best_bits + 6
}

/// Choose the best model for `signal` and leave it, with its residuals, in `slot`.
#[inline(always)]
fn plan_signal<const AVX2: bool>(
    slot: &mut Slot,
    signal: &[i32],
    width: u8,
    config: &EncoderConfig,
    windows: &Windows,
) {
    let combined = signal.iter().fold(0i32, |acc, &s| acc | s);
    let wasted = if combined == 0 {
        0
    } else {
        combined.trailing_zeros() as u8
    };
    slot.wasted = wasted;
    slot.width = width - wasted;
    if wasted == 0 {
        plan_samples::<AVX2>(slot, signal, config, windows);
    } else {
        let mut shifted = std::mem::take(&mut slot.shifted);
        shifted.clear();
        shifted.extend(signal.iter().map(|&s| s >> wasted));
        plan_samples::<AVX2>(slot, &shifted, config, windows);
        slot.shifted = shifted;
    }
}

#[inline(always)]
fn plan_samples<const AVX2: bool>(slot: &mut Slot, samples: &[i32], config: &EncoderConfig, windows: &Windows) {
    let n = samples.len();
    let width = slot.width;
    let header_bits = 8 + slot.wasted as usize;

    let first = samples.first().copied().unwrap_or(0);
    if samples.iter().fold(0i32, |acc, &s| acc | (s ^ first)) == 0 {
        slot.model = Model::Constant(first);
        slot.bits = header_bits + width as usize;
        return;
    }

    slot.model = Model::Verbatim;
    slot.bits = header_bits + n * width as usize;

    // Fixed predictor: the order with the smallest error sum.
    let sums = fixed_error_sums(samples, width);
    let mut order = 0usize;
    for (candidate, &sum) in sums.iter().enumerate() {
        if sum < sums[order] {
            order = candidate;
        }
    }
    if n > order {
        fixed_residual(samples, order, &mut slot.trial);
        let mut rice = RiceChoice::empty();
        let rice_bits = plan_rice(&slot.trial, n, order, width, config, &mut slot.sums, &mut rice);
        let bits = header_bits + order * width as usize + rice_bits;
        if bits < slot.bits {
            slot.bits = bits;
            slot.model = Model::Fixed(order as u8);
            slot.rice = rice;
            std::mem::swap(&mut slot.residual, &mut slot.trial);
        }
    }

    let max_order = config.max_lpc_order.min(MAX_LPC_ORDER).min(n.saturating_sub(1));
    if max_order > 0 {
        plan_lpc::<AVX2>(slot, samples, width, header_bits, max_order, config, windows);
    }
}

/// One set of LPC coefficients awaiting evaluation.
#[derive(Clone, Copy)]
struct Candidate {
    order: usize,
    coeffs: [f64; MAX_LPC_ORDER],
    /// Estimated cost in bits, from the whole-block autocorrelation.
    score: f64,
}

/// Evaluate LPC models for a signal and keep the best in `slot`, if better than what it holds.
#[inline(always)]
fn plan_lpc<const AVX2: bool>(
    slot: &mut Slot,
    samples: &[i32],
    width: u8,
    header_bits: usize,
    max_order: usize,
    config: &EncoderConfig,
    windows: &Windows,
) {
    let n = samples.len();
    let lags = max_order + 1;

    apply_window(samples, &windows.root, &mut slot.windowed);
    let mut root = [0.0f64; MAX_LPC_ORDER + 1];
    autocorrelate::<AVX2>(&slot.windowed, &mut root[..lags]);
    if root[0] <= 0.0 {
        return;
    }

    let overhead = f64::from(width) + f64::from(config.qlp_precision);
    let mut best_root = Candidate {
        order: 0,
        coeffs: [0.0; MAX_LPC_ORDER],
        score: f64::INFINITY,
    };
    if !solve(&root[..lags], max_order, n, overhead, &mut best_root) {
        return;
    }
    evaluate_lpc::<AVX2>(slot, samples, width, header_bits, &best_root, config);

    let parts_max = config.window_parts as usize;
    if parts_max < 2 || config.window_trials == 0 {
        return;
    }

    // Partial windows and their complements. Each is scored on a stretch of the block, which
    // takes a fraction of a full evaluation and sees the signal's real noise.
    let mut trials = [Candidate {
        order: 0,
        coeffs: [0.0; MAX_LPC_ORDER],
        score: f64::INFINITY,
    }; 12];
    let keep = (config.window_trials as usize).min(trials.len());
    let mut stretch = std::mem::take(&mut slot.trial);
    let mut consider = |candidate: &Candidate| {
        let mut scored = *candidate;
        scored.score = score_on_stretch::<AVX2>(samples, candidate, width, overhead, config, &mut stretch);
        let mut at = keep;
        while at > 0 && scored.score < trials[at - 1].score {
            if at < keep {
                trials[at] = trials[at - 1];
            }
            at -= 1;
        }
        if at < keep {
            trials[at] = scored;
        }
    };
    for parts in 2..=parts_max {
        let part_len = n / parts;
        if part_len <= max_order + 1 {
            break;
        }
        for part in 0..parts {
            let start = part * part_len;
            apply_partial_window(&samples[start..start + part_len], &windows.root, &mut slot.windowed);
            let mut partial = [0.0f64; MAX_LPC_ORDER + 1];
            autocorrelate::<AVX2>(&slot.windowed, &mut partial[..lags]);
            let mut candidate = Candidate {
                order: 0,
                coeffs: [0.0; MAX_LPC_ORDER],
                score: f64::INFINITY,
            };
            if partial[0] > 0.0 && solve(&partial[..lags], max_order, n, overhead, &mut candidate) {
                consider(&candidate);
            }
            // The rest of the block, from the whole-block autocorrelation less this part's.
            let mut rest = [0.0f64; MAX_LPC_ORDER + 1];
            for ((r, &a), &b) in rest[..lags].iter_mut().zip(&root[..lags]).zip(&partial[..lags]) {
                *r = a - b;
            }
            if rest[0] > 0.0 && solve(&rest[..lags], max_order, n, overhead, &mut candidate) {
                consider(&candidate);
            }
        }
    }
    slot.trial = stretch;
    for candidate in &trials[..keep] {
        if candidate.score.is_finite() {
            evaluate_lpc::<AVX2>(slot, samples, width, header_bits, candidate, config);
        }
    }
}

#[inline(always)]
fn apply_window(samples: &[i32], window: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.extend(samples.iter().zip(window).map(|(&s, &w)| f64::from(s) * w));
}

/// Window one part of a block with the taper of `window`: the first half of the part uses the
/// window's start and the second half its end.
#[inline(always)]
fn apply_partial_window(samples: &[i32], window: &[f64], out: &mut Vec<f64>) {
    let len = samples.len();
    let half = len / 2;
    out.clear();
    out.extend(samples[..half].iter().zip(window).map(|(&s, &w)| f64::from(s) * w));
    let tail = &window[window.len() - (len - half)..];
    out.extend(samples[half..].iter().zip(tail).map(|(&s, &w)| f64::from(s) * w));
}

#[inline(always)]
fn autocorrelate<const AVX2: bool>(x: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if AVX2 {
        // SAFETY: the AVX2 path is only taken on a machine with these features.
        unsafe { lpc::autocorrelate_fma(x, out) };
        return;
    }
    lpc::autocorrelate_scalar(x, out);
}

/// Levinson-Durbin recursion. Chooses the order with the smallest estimated cost and leaves its
/// coefficients in `out`. Returns false when no order is usable.
#[inline(always)]
fn solve(autoc: &[f64], max_order: usize, n: usize, overhead: f64, out: &mut Candidate) -> bool {
    let mut a = [0.0f64; MAX_LPC_ORDER];
    let mut reflection = [0.0f64; MAX_LPC_ORDER];
    let mut error = autoc[0];
    let scale = 0.5 / n as f64;
    let mut best_order = 0usize;
    let mut best_bits = f64::INFINITY;
    let max_order = max_order.min(autoc.len() - 1);
    for i in 0..max_order {
        let mut sum = autoc[i + 1];
        for j in 0..i {
            sum -= a[j] * autoc[i - j];
        }
        if error <= 0.0 {
            break;
        }
        let k = sum / error;
        reflection[i] = k;
        a[i] = k;
        for j in 0..i / 2 {
            let (x, y) = (a[j], a[i - 1 - j]);
            a[j] = x - k * y;
            a[i - 1 - j] = y - k * x;
        }
        if i % 2 == 1 {
            a[i / 2] -= k * a[i / 2];
        }
        error *= 1.0 - k * k;
        let order = i + 1;
        let bits_per_sample = if error > 0.0 {
            (0.5 * (scale * error).log2()).max(0.0)
        } else {
            0.0
        };
        let bits = bits_per_sample * (n - order) as f64 + order as f64 * overhead;
        if bits < best_bits {
            best_bits = bits;
            best_order = order;
        }
    }
    if best_order == 0 {
        return false;
    }
    // Rebuild the coefficients of the chosen order from its reflection coefficients.
    let mut c = [0.0f64; MAX_LPC_ORDER];
    for i in 0..best_order {
        let k = reflection[i];
        c[i] = k;
        for j in 0..i / 2 {
            let (x, y) = (c[j], c[i - 1 - j]);
            c[j] = x - k * y;
            c[i - 1 - j] = y - k * x;
        }
        if i % 2 == 1 {
            c[i / 2] -= k * c[i / 2];
        }
    }
    out.order = best_order;
    out.coeffs = c;
    out.score = best_bits;
    true
}

/// Estimated cost of `candidate` from its real residuals on the middle of the block.
#[inline(always)]
fn score_on_stretch<const AVX2: bool>(
    samples: &[i32],
    candidate: &Candidate,
    width: u8,
    overhead: f64,
    config: &EncoderConfig,
    scratch: &mut Vec<i32>,
) -> f64 {
    const STRETCH: usize = 512;
    let order = candidate.order;
    let n = samples.len();
    let mut coeffs = [0i32; MAX_LPC_ORDER];
    let precision = stretch_precision(config.qlp_precision, width, order);
    let Some(shift) = lpc::quantize_with_feedback(&candidate.coeffs[..order], precision, &mut coeffs[..order]) else {
        return f64::INFINITY;
    };
    let len = (STRETCH + order).min(n);
    let start = (n - len) / 2;
    lpc::lpc_residual_into::<AVX2>(&samples[start..start + len], &coeffs[..order], shift, width, scratch);
    let sum: u64 = scratch.iter().map(|r| u64::from(r.unsigned_abs())).sum();
    let mean = (sum as f64 / scratch.len().max(1) as f64).max(0.25);
    (mean.log2() + 1.5) * n as f64 + order as f64 * overhead
}

/// Keep the prediction within 32 bits for narrow signals, where that gives the faster path.
#[inline(always)]
const fn stretch_precision(precision: u8, width: u8, order: usize) -> u8 {
    if width <= 17 {
        let order_log = usize::BITS - 1 - order.leading_zeros();
        let most = 32 - width as u32 - order_log;
        let most = if most < 5 { 5 } else { most };
        if precision as u32 > most { most as u8 } else { precision }
    } else {
        precision
    }
}

/// Quantise, compute residuals and plan Rice coding for a candidate; keep it if it wins.
#[inline(always)]
fn evaluate_lpc<const AVX2: bool>(
    slot: &mut Slot,
    samples: &[i32],
    width: u8,
    header_bits: usize,
    candidate: &Candidate,
    config: &EncoderConfig,
) {
    let order = candidate.order;
    let n = samples.len();
    let precision = stretch_precision(config.qlp_precision, width, order);
    let mut coeffs = [0i32; MAX_LPC_ORDER];
    let Some(shift) = lpc::quantize_with_feedback(&candidate.coeffs[..order], precision, &mut coeffs[..order]) else {
        return;
    };
    lpc::lpc_residual_into::<AVX2>(samples, &coeffs[..order], shift, width, &mut slot.trial);
    let mut rice = RiceChoice::empty();
    let rice_bits = plan_rice(&slot.trial, n, order, width, config, &mut slot.sums, &mut rice);
    let bits = header_bits + order * width as usize + 4 + 5 + order * precision as usize + rice_bits;
    if bits < slot.bits {
        slot.bits = bits;
        slot.model = Model::Lpc {
            order: order as u8,
            shift,
            precision,
            coeffs,
        };
        slot.rice = rice;
        std::mem::swap(&mut slot.residual, &mut slot.trial);
    }
}

/// Write the model planned in `slot` for `signal`.
#[inline(always)]
fn write_subframe(writer: &mut BitWriter, slot: &Slot, signal: &[i32]) {
    let samples: &[i32] = if slot.wasted > 0 { &slot.shifted } else { signal };
    let width = slot.width;
    let wasted_bits = slot.wasted;
    match slot.model {
        Model::Constant(value) => {
            SubframeHeader {
                subframe_type: SubframeType::Constant,
                wasted_bits,
            }
            .encode(writer);
            writer.write_bits_signed(value, width);
        },
        Model::Verbatim => {
            SubframeHeader {
                subframe_type: SubframeType::Verbatim,
                wasted_bits,
            }
            .encode(writer);
            for &sample in samples {
                writer.write_bits_signed(sample, width);
            }
        },
        Model::Fixed(order) => {
            SubframeHeader {
                subframe_type: SubframeType::Fixed(order),
                wasted_bits,
            }
            .encode(writer);
            for &sample in &samples[..order as usize] {
                writer.write_bits_signed(sample, width);
            }
            write_residual(writer, slot, samples.len(), order as usize);
        },
        Model::Lpc {
            order,
            shift,
            precision,
            coeffs,
        } => {
            SubframeHeader {
                subframe_type: SubframeType::Lpc(order),
                wasted_bits,
            }
            .encode(writer);
            for &sample in &samples[..order as usize] {
                writer.write_bits_signed(sample, width);
            }
            writer.write_bits(u32::from(precision - 1), 4);
            writer.write_bits_signed(i32::from(shift), 5);
            for &coeff in &coeffs[..order as usize] {
                writer.write_bits_signed(coeff, precision);
            }
            write_residual(writer, slot, samples.len(), order as usize);
        },
    }
}

#[inline(always)]
fn write_residual(writer: &mut BitWriter, slot: &Slot, block_size: usize, predictor_order: usize) {
    let rice = &slot.rice;
    let (method, param_bits, escape) = if rice.extended { (1, 5, 31) } else { (0, 4, 15) };
    writer.write_bits(method, 2);
    writer.write_bits(u32::from(rice.order), 4);
    let limit: u32 = if rice.extended { 30 } else { 14 };
    let part_size = block_size >> rice.order;
    let mut pos = 0usize;
    for (p, &param) in rice.params[..1usize << rice.order].iter().enumerate() {
        let count = if p == 0 { part_size - predictor_order } else { part_size };
        if param == ZERO_PARTITION {
            writer.write_bits(escape, param_bits);
            writer.write_bits(0, 5);
        } else {
            // The plan's estimate is close: settle the parameter with exact counts around it.
            let run = &slot.residual[pos..pos + count];
            let mut k = u32::from(param);
            let mut bits = exact_rice_bits(run, k);
            if k < limit {
                let above = exact_rice_bits(run, k + 1);
                if above < bits {
                    (k, bits) = (k + 1, above);
                }
            }
            if k > 0 && k == u32::from(param) {
                let below = exact_rice_bits(run, k - 1);
                if below < bits {
                    (k, bits) = (k - 1, below);
                }
            }
            writer.write_bits(k, param_bits);
            writer.reserve_bits(bits + 64);
            // SAFETY: `bits` is the exact size of these codes and `reserve_bits` covers them.
            unsafe { writer.write_rice_run(run, k) };
        }
        pos += count;
    }
}
