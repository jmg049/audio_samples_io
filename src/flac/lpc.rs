//! Linear Predictive Coding (LPC) for FLAC.
//!
//! This module implements:
//! - Fixed predictors (orders 0-4) using predefined coefficients
//! - LPC analysis using the Levinson-Durbin algorithm
//! - Quantized LPC coefficient encoding/decoding
//! - Prediction and residual computation

use crate::flac::constants::{MAX_FIXED_ORDER, MAX_LPC_ORDER};
use crate::flac::error::FlacError;

// AVX2 SIMD helpers for LPC dot-product: VPMOVSXDQ (sign-extend i32→i64) + VPMULDQ (widening multiply).
// VPMULLQ (i64×i64) requires AVX-512DQ which is not widely available; VPMULDQ uses the lower 32 bits of
// each 64-bit lane, so sign-extending i32→i64 first gives the correct 32→64-bit widening product.
//
// Each function computes Σ c[j] * s[i-1-j] for j in 0..N without serial i64 multiply.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx2_lpc {
    use std::arch::x86_64::*;

    /// Horizontal sum of 4 i64 lanes in a 256-bit register.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn hsum4_i64(v: __m256i) -> i64 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let s = _mm_add_epi64(lo, hi);
        let hi64 = _mm_unpackhi_epi64(s, s);
        _mm_cvtsi128_si64(_mm_add_epi64(s, hi64))
    }

    /// Dot product of 4 coefficients with 4 contiguous reversed samples.
    /// `s_ptr` points to `samples[i-4]`; reversed → matches c[0..3].
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn dot4(c: *const i32, s_ptr: *const i32) -> i64 {
        let c_vec = _mm_loadu_si128(c as *const __m128i);
        let s_raw = _mm_loadu_si128(s_ptr as *const __m128i);
        let s_rev = _mm_shuffle_epi32(s_raw, 0x1B); // [3,2,1,0] reverse
        let prod = _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(c_vec),
            _mm256_cvtepi32_epi64(s_rev),
        );
        hsum4_i64(prod)
    }

    /// Dot product of 8 coefficients. `s_ptr` points to `samples[i-8]`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn dot8(c: *const i32, s_ptr: *const i32) -> i64 {
        let c_lo = _mm_loadu_si128(c as *const __m128i);
        let c_hi = _mm_loadu_si128(c.add(4) as *const __m128i);
        // lower 4: samples[i-4..i-1], reversed → c[0..3]
        let s_lo_rev = _mm_shuffle_epi32(_mm_loadu_si128(s_ptr.add(4) as *const __m128i), 0x1B);
        // upper 4: samples[i-8..i-5], reversed → c[4..7]
        let s_hi_rev = _mm_shuffle_epi32(_mm_loadu_si128(s_ptr as *const __m128i), 0x1B);
        let sum = _mm256_add_epi64(
            _mm256_mul_epi32(_mm256_cvtepi32_epi64(c_lo), _mm256_cvtepi32_epi64(s_lo_rev)),
            _mm256_mul_epi32(_mm256_cvtepi32_epi64(c_hi), _mm256_cvtepi32_epi64(s_hi_rev)),
        );
        hsum4_i64(sum)
    }

    /// Dot product of 12 coefficients. `s_ptr` points to `samples[i-12]`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn dot12(c: *const i32, s_ptr: *const i32) -> i64 {
        let c0 = _mm_loadu_si128(c as *const __m128i);
        let c1 = _mm_loadu_si128(c.add(4) as *const __m128i);
        let c2 = _mm_loadu_si128(c.add(8) as *const __m128i);
        let s0 = _mm_shuffle_epi32(_mm_loadu_si128(s_ptr.add(8) as *const __m128i), 0x1B);
        let s1 = _mm_shuffle_epi32(_mm_loadu_si128(s_ptr.add(4) as *const __m128i), 0x1B);
        let s2 = _mm_shuffle_epi32(_mm_loadu_si128(s_ptr as *const __m128i), 0x1B);
        let sum = _mm256_add_epi64(
            _mm256_add_epi64(
                _mm256_mul_epi32(_mm256_cvtepi32_epi64(c0), _mm256_cvtepi32_epi64(s0)),
                _mm256_mul_epi32(_mm256_cvtepi32_epi64(c1), _mm256_cvtepi32_epi64(s1)),
            ),
            _mm256_mul_epi32(_mm256_cvtepi32_epi64(c2), _mm256_cvtepi32_epi64(s2)),
        );
        hsum4_i64(sum)
    }

    /// Compute Σ c[j] * s[si-1-j] for j in 0..ORDER using SIMD, ORDER known at compile time.
    /// LLVM monomorphizes this — the inner match is compile-time resolved and dot4/dot8/dot12
    /// are inlined, so the whole loop body becomes a straight VPMULDQ sequence.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn dot_order<const ORDER: usize>(c: *const i32, s: *const i32, si: usize) -> i64 {
        match ORDER {
            1  => *c as i64 * *s.add(si-1) as i64,
            2  => *c as i64 * *s.add(si-1) as i64
                + *c.add(1) as i64 * *s.add(si-2) as i64,
            3  => *c as i64 * *s.add(si-1) as i64
                + *c.add(1) as i64 * *s.add(si-2) as i64
                + *c.add(2) as i64 * *s.add(si-3) as i64,
            4  => dot4(c, s.add(si-4)),
            5  => dot4(c, s.add(si-4))
                + *c.add(4) as i64 * *s.add(si-5) as i64,
            6  => dot4(c, s.add(si-4))
                + *c.add(4) as i64 * *s.add(si-5) as i64
                + *c.add(5) as i64 * *s.add(si-6) as i64,
            7  => dot4(c, s.add(si-4))
                + *c.add(4) as i64 * *s.add(si-5) as i64
                + *c.add(5) as i64 * *s.add(si-6) as i64
                + *c.add(6) as i64 * *s.add(si-7) as i64,
            8  => dot8(c, s.add(si-8)),
            9  => dot8(c, s.add(si-8))
                + *c.add(8) as i64 * *s.add(si-9) as i64,
            10 => dot8(c, s.add(si-8))
                + *c.add(8) as i64 * *s.add(si-9) as i64
                + *c.add(9) as i64 * *s.add(si-10) as i64,
            11 => dot8(c, s.add(si-8))
                + *c.add(8) as i64 * *s.add(si-9) as i64
                + *c.add(9) as i64 * *s.add(si-10) as i64
                + *c.add(10) as i64 * *s.add(si-11) as i64,
            12 => dot12(c, s.add(si-12)),
            _  => {
                let mut acc = 0i64;
                for j in 0..ORDER { acc += *c.add(j) as i64 * *s.add(si-1-j) as i64; }
                acc
            }
        }
    }

    /// Compute 4 residuals at a time using AVX2 — eliminates horizontal reduction.
    ///
    /// For each coefficient c[j], the 4 sample values needed by residuals k..k+3 are
    /// consecutive in memory: s[ORDER+k-1-j .. ORDER+k+2-j]. One 128-bit load covers
    /// all four, sign-extended to i64 for VPMULDQ. Because we accumulate 4 independent
    /// dot products in 4 SIMD lanes, there is no horizontal sum — the bottleneck that
    /// made the 1-wide approach show no measurable speedup.
    ///
    /// After ORDER coefficient iterations the 4 accumulated i64 predictions are
    /// arithmetic-right-shifted (AVX2 workaround: logical shift + sign-fill mask),
    /// subtracted from the corresponding actual samples, and narrowed back to i32 with
    /// `_mm256_permutevar8x32_epi32` selecting the low 32 bits of each i64 lane.
    ///
    /// Tail samples (n % 4 != 0) fall through to the scalar `dot_order` path.
    #[target_feature(enable = "avx2")]
    pub unsafe fn lpc_residuals4<const ORDER: usize>(
        s: *const i32, c: *const i32, shift: i32, out: *mut i32, n: usize,
    ) {
        let ushift = shift as u64;
        // Arithmetic right shift workaround: logical shift + OR sign-fill bits.
        // fill_bits = upper `shift` bits all 1; applied only to negative accumulators.
        let fill_bits: u64 = if ushift == 0 { 0 } else { !0u64 << (64 - ushift) };
        let shift_v  = _mm256_set1_epi64x(ushift as i64);
        let fill_v   = _mm256_set1_epi64x(fill_bits as i64);
        let zero     = _mm256_setzero_si256();
        // Permutation to select lower 32 bits of each i64 lane into lower 128 bits.
        let narrow_perm = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);

        // Pre-load all coefficients into a stack-local array before the hot outer loop.
        // LLVM cannot prove that writes to `out` (heap) don't alias `c` (also heap), so
        // without this it reloads all ORDER coefficients from `c` on every outer iteration.
        // Stack-local `c_i32` is provably unaliased → LLVM keeps them in registers.
        let mut c_i32 = [0i32; ORDER];
        for (j, item) in c_i32.iter_mut().enumerate().take(ORDER) {
            *item = *c.add(j);
        }

        let n4 = (n / 4) * 4;
        let mut k = 0usize;

        while k < n4 {
            let base = ORDER + k;
            let mut acc = _mm256_setzero_si256();

            // Unrolled by LLVM because ORDER is a const generic.
            // For coefficient j: multiply c[j] by the 4 consecutive samples
            //   s[base-1-j], s[base-j], s[base+1-j], s[base+2-j]
            // which are the samples used by residuals k, k+1, k+2, k+3 respectively.
            for (j, &cj) in c_i32.iter().enumerate().take(ORDER) {
                let cj   = _mm256_set1_epi64x(cj as i64);
                let s4   = _mm_loadu_si128(s.add(base - 1 - j) as *const __m128i);
                let s4e  = _mm256_cvtepi32_epi64(s4);
                acc = _mm256_add_epi64(acc, _mm256_mul_epi32(cj, s4e));
            }

            // Arithmetic right shift by `shift`.
            let lshifted  = _mm256_srlv_epi64(acc, shift_v);
            let sign_mask = _mm256_cmpgt_epi64(zero, acc); // -1 where acc < 0
            let pred      = _mm256_or_si256(lshifted, _mm256_and_si256(sign_mask, fill_v));

            // actual[k..k+4] as i64
            let actual4  = _mm_loadu_si128(s.add(base) as *const __m128i);
            let actual4e = _mm256_cvtepi32_epi64(actual4);

            // residual = actual - pred (i64), then narrow to i32
            let res64  = _mm256_sub_epi64(actual4e, pred);
            let narrow = _mm256_permutevar8x32_epi32(res64, narrow_perm);
            _mm_storeu_si128(out.add(k) as *mut __m128i, _mm256_castsi256_si128(narrow));

            k += 4;
        }

        // Scalar tail for remaining 0-3 samples.
        while k < n {
            let i    = ORDER + k;
            let pred = dot_order::<ORDER>(c, s, i) >> shift;
            *out.add(k) = (*s.add(i) as i64 - pred) as i32;
            k += 1;
        }
    }

}

/// Restore one sample: compute the LPC dot product and add residual.
///
/// All ORDER multiplications are written as independent expressions so LLVM
/// (and the OOO back-end) can issue them in parallel.  A flat sum of N
/// independent products has a tree-reduced latency of ~log2(N) ADD cycles
/// instead of the ~8-cycle horizontal reduction (hsum4_i64) required by the
/// AVX2 SIMD path.
///
/// Critical-path for ORDER=12:
///   store-forwarding s[i-1]: ~5 cycles
///   12 parallel IMUL:         3 cycles (all independent)
///   tree reduce (12 → 1):     4 cycles
///   shift + add + store:      3 cycles
///   Total: ~15 cycles  (vs ~23 for dot12+hsum4)
#[inline(always)]
unsafe fn restore_sample<const ORDER: usize>(
    buf: *const i32, cv: &[i32; ORDER], si: usize,
) -> i64 {
    // Enumerate all products explicitly.  Since every p_j is independent,
    // LLVM issues all ORDER IMUL instructions simultaneously.
    // The flat-sum expression lets LLVM pick the optimal tree reduction.
    let mut sum = 0i64;
    for (j, item) in cv.iter().enumerate().take(ORDER) {
        sum += *item as i64 * unsafe { *buf.add(si - 1 - j) } as i64;
    }
    sum
}

/// Restore loop using scalar multi-accumulator dot product (no SIMD hsum).
///
/// Replaces the AVX2 `lpc_restore` dispatch.  See `restore_sample` for the
/// critical-path analysis.
#[inline(always)]
unsafe fn lpc_restore_fast<const ORDER: usize>(
    buf: *mut i32, res: *const i32, n: usize, c: *const i32, shift: i32,
) {
    // Stack-local copy: prevents LLVM from treating c and buf as aliases,
    // which would force coefficient reloads on every outer iteration.
    let mut cv = [0i32; ORDER];
    for (j, item) in cv.iter_mut().enumerate().take(ORDER) {
        // safety: caller guarantees c points to ORDER i32 coefficients; j < ORDER bounds the access.
        unsafe { *item = *c.add(j); }
    }

    for k in 0..n {
        let si = ORDER + k;
        let pred = unsafe { restore_sample::<ORDER>(buf, &cv, si) } >> shift;
        let residual = unsafe { *res.add(k) } as i64;
        unsafe { *buf.add(si) = (pred + residual) as i32; }
    }
}

/// Apply a fixed predictor and compute residuals.
///
/// Each order is a finite difference, written as direct formulas so LLVM can
/// auto-vectorize the loop (no loop-carried dependency, no multiplications).
///
/// - Order 0: residual = sample
/// - Order 1: residual = s[i] - s[i-1]
/// - Order 2: residual = s[i] - 2*s[i-1] + s[i-2]   (wrapping i32)
/// - Order 3: residual = s[i] - 3*s[i-1] + 3*s[i-2] - s[i-3]
/// - Order 4: residual = s[i] - 4*s[i-1] + 6*s[i-2] - 4*s[i-3] + s[i-4]
pub fn fixed_predictor_residual(samples: &[i32], order: usize) -> Result<Vec<i32>, FlacError> {
    if order > MAX_FIXED_ORDER {
        return Err(FlacError::InvalidFixedOrder { order: order as u8 });
    }
    if samples.len() < order {
        return Err(FlacError::InvalidBlockSize { size: samples.len() as u32 });
    }

    let n = samples.len();
    let out_len = n - order;
    let mut residuals = Vec::with_capacity(out_len);
    let ptr: *mut i32 = residuals.as_mut_ptr();

    // Each match arm is a branchless, multiplication-free loop that LLVM vectorizes
    // using VPSUBD/VPADDD/VPSLLD (8 i32 per cycle on AVX2).
    match order {
        0 => {
            residuals.extend_from_slice(samples);
            return Ok(residuals);
        }
        1 => {
            for i in 0..out_len {
                unsafe { ptr.add(i).write(samples[i + 1].wrapping_sub(samples[i])); }
            }
        }
        2 => {
            for i in 0..out_len {
                // s[i+2] - 2*s[i+1] + s[i]  (coefficients 2 → left-shift, no IMUL)
                let r = samples[i + 2]
                    .wrapping_sub(samples[i + 1].wrapping_shl(1))
                    .wrapping_add(samples[i]);
                unsafe { ptr.add(i).write(r); }
            }
        }
        3 => {
            for i in 0..out_len {
                // s[i+3] - 3*s[i+2] + 3*s[i+1] - s[i]
                let r = samples[i + 3]
                    .wrapping_sub(samples[i + 2].wrapping_mul(3))
                    .wrapping_add(samples[i + 1].wrapping_mul(3))
                    .wrapping_sub(samples[i]);
                unsafe { ptr.add(i).write(r); }
            }
        }
        4 => {
            for i in 0..out_len {
                // s[i+4] - 4*s[i+3] + 6*s[i+2] - 4*s[i+1] + s[i]
                let r = samples[i + 4]
                    .wrapping_sub(samples[i + 3].wrapping_shl(2))
                    .wrapping_add(samples[i + 2].wrapping_mul(6))
                    .wrapping_sub(samples[i + 1].wrapping_shl(2))
                    .wrapping_add(samples[i]);
                unsafe { ptr.add(i).write(r); }
            }
        }
        _ => unreachable!(),
    }

    unsafe { residuals.set_len(out_len); }
    Ok(residuals)
}

/// Restore samples from residuals using a fixed predictor.
///
/// # Arguments
/// * `warmup` - Warm-up samples (first `order` samples)
/// * `residuals` - Residual values
/// * `order` - Predictor order (0-4)
///
/// # Returns
/// Vector of restored samples (including warm-up)
pub fn fixed_predictor_restore(
    warmup: &[i32],
    residuals: &[i32],
    order: usize,
) -> Result<Vec<i32>, FlacError> {
    if order > MAX_FIXED_ORDER {
        return Err(FlacError::InvalidFixedOrder { order: order as u8 });
    }

    if warmup.len() < order {
        return Err(FlacError::InvalidBlockSize {
            size: warmup.len() as u32,
        });
    }

    let mut samples = Vec::with_capacity(order + residuals.len());
    samples.extend_from_slice(&warmup[..order]);

    // Unroll the inner loop using the known fixed coefficients for each order,
    // avoiding the generic loop overhead and table lookups on every sample.
    match order {
        0 => {
            for &r in residuals {
                samples.push(r);
            }
        }
        1 => {
            for &r in residuals {
                let i = samples.len();
                let p = samples[i - 1] as i64;
                samples.push((p + r as i64) as i32);
            }
        }
        2 => {
            for &r in residuals {
                let i = samples.len();
                let p = 2 * samples[i - 1] as i64 - samples[i - 2] as i64;
                samples.push((p + r as i64) as i32);
            }
        }
        3 => {
            for &r in residuals {
                let i = samples.len();
                let p = 3 * samples[i - 1] as i64
                    - 3 * samples[i - 2] as i64
                    + samples[i - 3] as i64;
                samples.push((p + r as i64) as i32);
            }
        }
        4 => {
            for &r in residuals {
                let i = samples.len();
                let p = 4 * samples[i - 1] as i64
                    - 6 * samples[i - 2] as i64
                    + 4 * samples[i - 3] as i64
                    - samples[i - 4] as i64;
                samples.push((p + r as i64) as i32);
            }
        }
        _ => unreachable!("fixed predictor order > 4 should have been caught above"),
    }

    Ok(samples)
}

/// Restore samples from residuals using a fixed predictor, in-place.
///
/// `buf[offset..offset+order]` must contain warmup samples and
/// `buf[offset+order..offset+block_size]` the residuals. Converts in-place,
/// left-to-right. Eliminates the output-Vec allocation of `fixed_predictor_restore`.
pub(crate) fn fixed_predictor_restore_into(
    buf: &mut [i32],
    offset: usize,
    order: usize,
    block_size: usize,
) -> Result<(), FlacError> {
    if order > MAX_FIXED_ORDER {
        return Err(FlacError::InvalidFixedOrder { order: order as u8 });
    }
    let end = offset + block_size;
    // Each match arm keeps the sliding window in explicit local variables.
    // Reading buf[i-k] after writing buf[i-k+1] creates a store-forwarding
    // stall (~5 cycles).  Local vars s0/s1/... live in registers throughout
    // the loop, so the write to buf[i] is output-only and never re-read
    // through memory.
    match order {
        0 => {}
        1 => {
            let mut s = buf[offset];
            for r in buf.iter_mut().take(end).skip(offset + 1) {
                let ns = s.wrapping_add(*r);
                *r = ns;
                s = ns;
            }
        }
        2 => {
            let mut s0 = buf[offset];
            let mut s1 = buf[offset + 1];
            for r in buf.iter_mut().take(end).skip(offset + 2) {
                let s2 = s1.wrapping_shl(1).wrapping_sub(s0).wrapping_add(*r);
                *r = s2;
                s0 = s1;
                s1 = s2;
            }
        }
        3 => {
            let mut s0 = buf[offset];
            let mut s1 = buf[offset + 1];
            let mut s2 = buf[offset + 2];
            for r in buf.iter_mut().take(end).skip(offset + 3) {
                let s3 = s2.wrapping_mul(3)
                    .wrapping_sub(s1.wrapping_mul(3))
                    .wrapping_add(s0)
                    .wrapping_add(*r);
                *r = s3;
                s0 = s1;
                s1 = s2;
                s2 = s3;
            }
        }
        4 => {
            let mut s0 = buf[offset];
            let mut s1 = buf[offset + 1];
            let mut s2 = buf[offset + 2];
            let mut s3 = buf[offset + 3];
            for r in buf.iter_mut().take(end).skip(offset + 4)  {
                let s4 = s3.wrapping_shl(2)
                    .wrapping_sub(s2.wrapping_mul(6))
                    .wrapping_add(s1.wrapping_shl(2))
                    .wrapping_sub(s0)
                    .wrapping_add(*r);
                *r = s4;
                s0 = s1; s1 = s2; s2 = s3; s3 = s4;
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Restore samples from LPC residuals, in-place.
///
/// `buf[offset..offset+order]` = warmup; `buf[offset+order..offset+block_size]` = residuals.
/// The base pointer passed to `lpc_restore_fast_into` is `buf.as_mut_ptr().add(offset)`,
/// so all index arithmetic inside that function is relative to the frame start.
pub(crate) fn lpc_predictor_restore_into(
    buf: &mut Vec<i32>,
    offset: usize,
    order: usize,
    qlp_coeffs: &[i32],
    qlp_shift: i8,
) -> Result<(), FlacError> {
    if order == 0 || order > MAX_LPC_ORDER {
        return Err(FlacError::InvalidLpcOrder { order: order as u8 });
    }
    if qlp_shift < 0 {
        return Err(FlacError::InvalidLpcShift { shift: qlp_shift });
    }

    let n = buf.len() - offset - order;
    let shift = qlp_shift as i32;
    let c = qlp_coeffs.as_ptr();
    // Pass buf+offset so lpc_restore_fast_into's indexing is relative to the frame start.
    let p = unsafe { buf.as_mut_ptr().add(offset) };

    macro_rules! dispatch_restore_into {
        ($N:literal) => { unsafe {
            lpc_restore_fast_into::<$N>(p, n, c, shift);
        }};
    }

    match order {
        1  => dispatch_restore_into!(1),  2  => dispatch_restore_into!(2),
        3  => dispatch_restore_into!(3),  4  => dispatch_restore_into!(4),
        5  => dispatch_restore_into!(5),  6  => dispatch_restore_into!(6),
        7  => dispatch_restore_into!(7),  8  => dispatch_restore_into!(8),
        9  => dispatch_restore_into!(9),  10 => dispatch_restore_into!(10),
        11 => dispatch_restore_into!(11), 12 => dispatch_restore_into!(12),
        _ => {
            for k in 0..n {
                let si = offset + order + k;
                let residual = buf[si] as i64;
                let mut pred: i64 = 0;
                for (j, &coeff) in qlp_coeffs.iter().enumerate() {
                    pred += coeff as i64 * buf[si - 1 - j] as i64;
                }
                buf[si] = ((pred >> shift) + residual) as i32;
            }
        }
    }

    Ok(())
}

/// In-place LPC restore; `buf` is already offset to the frame start (buf[0..order]=warmup).
/// Reads residual at buf[ORDER+k] before computing the prediction, then overwrites.
unsafe fn lpc_restore_fast_into<const ORDER: usize>(
    buf: *mut i32, n: usize, c: *const i32, shift: i32,
) {
    let mut cv = [0i32; ORDER];
    for (j, item) in cv.iter_mut().enumerate().take(ORDER) {
        unsafe { *item = *c.add(j); }
    }
    for k in 0..n {
        let si = ORDER + k;
        let residual = unsafe { *buf.add(si) } as i64;
        let pred = unsafe { restore_sample::<ORDER>(buf, &cv, si) } >> shift;
        unsafe { *buf.add(si) = (pred + residual) as i32; }
    }
}

/// Find the best fixed predictor order for given samples.
///
/// Replaces 5 separate O(N) passes (each with integer multiplications) with a single
/// O(N) pass using running finite differences. Orders 1-4 correspond to successive
/// differences, so all 5 energies accumulate simultaneously with 4 running state
/// variables and only wrapping_sub + unsigned_abs — no multiplications.
///
/// The first 4 iterations are peeled to warm up the state without branch guards in
/// the hot inner loop.
pub fn find_best_fixed_order(samples: &[i32]) -> usize {
    let n = samples.len();
    if n <= 4 {
        return 0;
    }
    let max_order = MAX_FIXED_ORDER.min(n - 1);
    let mut e = [0u64; 5];

    e[0] = samples[0].unsigned_abs() as u64;

    let r1_i1 = samples[1].wrapping_sub(samples[0]);
    e[0] += samples[1].unsigned_abs() as u64;
    e[1] += r1_i1.unsigned_abs() as u64;

    let r1_i2 = samples[2].wrapping_sub(samples[1]);
    let r2_i2 = r1_i2.wrapping_sub(r1_i1);
    e[0] += samples[2].unsigned_abs() as u64;
    e[1] += r1_i2.unsigned_abs() as u64;
    e[2] += r2_i2.unsigned_abs() as u64;

    let r1_i3 = samples[3].wrapping_sub(samples[2]);
    let r2_i3 = r1_i3.wrapping_sub(r1_i2);
    let r3_i3 = r2_i3.wrapping_sub(r2_i2);
    e[0] += samples[3].unsigned_abs() as u64;
    e[1] += r1_i3.unsigned_abs() as u64;
    e[2] += r2_i3.unsigned_abs() as u64;
    e[3] += r3_i3.unsigned_abs() as u64;

    let mut prev = samples[3];
    let mut d1 = r1_i3;
    let mut d2 = r2_i3;
    let mut d3 = r3_i3;

    for cur in samples.iter().take(n).skip(4)  {
        let r1 = cur.wrapping_sub(prev);
        let r2 = r1.wrapping_sub(d1);
        let r3 = r2.wrapping_sub(d2);
        let r4 = r3.wrapping_sub(d3);

        e[0] += cur.unsigned_abs() as u64;
        e[1] += r1.unsigned_abs() as u64;
        e[2] += r2.unsigned_abs() as u64;
        e[3] += r3.unsigned_abs() as u64;
        e[4] += r4.unsigned_abs() as u64;

        prev = *cur;
        d1 = r1;
        d2 = r2;
        d3 = r3;
    }

    e[..=max_order]
        .iter()
        .enumerate()
        .min_by_key(|&(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute autocorrelation coefficients.
///
/// With the `simd` feature: uses a transposed loop with `wide::f64x4` — a single
/// pass over the samples computing 4 lags at a time. This reduces memory bandwidth
/// from (max_order+1) passes × N samples down to one pass, and processes 4 lags
/// per SIMD lane instead of one.
///
/// Without `simd`: scalar f64 — LLVM can still auto-vectorize the inner loop
/// with AVX2 VMULPD (unlike i64, which requires AVX-512DQ for vectorization).
pub fn autocorrelation(samples: &[i32], max_order: usize) -> Vec<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        autocorrelation_simd(samples, max_order)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        autocorrelation_scalar(samples, max_order)
    }
}

/// AVX2 + FMA autocorrelation with 4-way unrolled inner loop.
///
/// The serial FMA chain `acc = fmadd(s[i], s4, acc)` is latency-bound at 5 cycles/iter
/// on processors with 5-cycle VFMADD latency. Unrolling by 4 with independent accumulators
/// allows the CPU to schedule 4 independent FMA chains across its 2 FMA ports, breaking
/// the serial dependency and increasing throughput ~3-4×.
///
/// Memory access pattern: for each group of 4 lags, the 4-way body reads
/// s[i-l3..i-l3+7] per outer iteration — 7 consecutive elements fit in one cache line.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
unsafe fn autocorrelation_avx2_fma(samples: &[i32], max_order: usize) -> Vec<f64> {
    use std::arch::x86_64::*;
    let n = samples.len();
    let n_lags = max_order + 1;
    let mut r = vec![0.0f64; n_lags];
    if n == 0 { return r; }
    let s = samples.as_ptr();
    let full_groups = n_lags / 4;

    for g in 0..full_groups {
        let l0 = g * 4;
        let l3 = l0 + 3;

        // Scalar warm-up: samples i in l0..l3 where lag l3 is not yet valid
        for i in l0..l3.min(n) {
            let si = *s.add(i) as f64;
            r[l0]                          += si * *s.add(i - l0) as f64;
            if i > l0 { r[l0 + 1] += si * *s.add(i - l0 - 1) as f64; }
            if i >= l0 + 2 { r[l0 + 2] += si * *s.add(i - l0 - 2) as f64; }
        }

        if n <= l3 { continue; }

        // 4-way unrolled SIMD loop: 4 independent accumulators break the serial FMA
        // dependency chain (5-cycle latency), filling both FMA ports for ~3× throughput.
        // Each body iteration processes i, i+1, i+2, i+3 simultaneously.
        // Access pattern per iteration: s[i-l3..i-l3+7] — 7 i32s, ≤ 2 cache lines.
        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd();
        let mut a3 = _mm256_setzero_pd();

        let n4 = l3 + ((n - l3) / 4) * 4;
        let mut i = l3;

        while i < n4 {
            let p = s.add(i - l3);  // base pointer for lagged samples this iteration
            a0 = _mm256_fmadd_pd(
                _mm256_set1_pd(*s.add(i)     as f64),
                _mm256_cvtepi32_pd(_mm_loadu_si128(p         as *const _)),
                a0,
            );
            a1 = _mm256_fmadd_pd(
                _mm256_set1_pd(*s.add(i + 1) as f64),
                _mm256_cvtepi32_pd(_mm_loadu_si128(p.add(1)  as *const _)),
                a1,
            );
            a2 = _mm256_fmadd_pd(
                _mm256_set1_pd(*s.add(i + 2) as f64),
                _mm256_cvtepi32_pd(_mm_loadu_si128(p.add(2)  as *const _)),
                a2,
            );
            a3 = _mm256_fmadd_pd(
                _mm256_set1_pd(*s.add(i + 3) as f64),
                _mm256_cvtepi32_pd(_mm_loadu_si128(p.add(3)  as *const _)),
                a3,
            );
            i += 4;
        }

        // Merge the 4 partial sums and handle the 0–3 tail elements serially.
        let mut acc = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
        while i < n {
            let si_f = _mm256_set1_pd(*s.add(i) as f64);
            let s4f  = _mm256_cvtepi32_pd(_mm_loadu_si128(s.add(i - l3) as *const _));
            acc = _mm256_fmadd_pd(si_f, s4f, acc);
            i += 1;
        }

        let mut tmp = [0.0f64; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
        r[l0]     += tmp[3];  // lane 3 → lag l0
        r[l0 + 1] += tmp[2];  // lane 2 → lag l0+1
        r[l0 + 2] += tmp[1];  // lane 1 → lag l0+2
        r[l0 + 3] += tmp[0];  // lane 0 → lag l0+3
    }

    // Remaining lags (0–3 lags that didn't fill a full group of 4).
    // Use 4-way scalar unrolling over samples to break the serial FMA latency
    // chain. LLVM will combine the 4 independent f64 accumulators into a
    // single VFMADD256PD instruction, eliminating the hsum overhead of the
    // group path while still being latency-independent.
    let base = full_groups * 4;
    for (lag, r) in r.iter_mut().enumerate().take(n_lags).skip(base) {
        let mut a0 = 0.0f64;
        let mut a1 = 0.0f64;
        let mut a2 = 0.0f64;
        let mut a3 = 0.0f64;

        let n4 = lag + ((n - lag) / 4) * 4;
        let mut i = lag;

        while i < n4 {
            a0 += *s.add(i)     as f64 * *s.add(i - lag)     as f64;
            a1 += *s.add(i + 1) as f64 * *s.add(i + 1 - lag) as f64;
            a2 += *s.add(i + 2) as f64 * *s.add(i + 2 - lag) as f64;
            a3 += *s.add(i + 3) as f64 * *s.add(i + 3 - lag) as f64;
            i += 4;
        }

        let mut sum = (a0 + a1) + (a2 + a3);
        while i < n {
            sum += *s.add(i) as f64 * *s.add(i - lag) as f64;
            i += 1;
        }
        *r = sum;
    }

    r
}

/// AVX2 autocorrelation (no FMA): same structure as the FMA version but uses
/// separate VMULPD + VADDPD. Used when FMA is not available.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn autocorrelation_avx2(samples: &[i32], max_order: usize) -> Vec<f64> {
    use std::arch::x86_64::*;
    let n = samples.len();
    let n_lags = max_order + 1;
    let mut r = vec![0.0f64; n_lags];
    if n == 0 { return r; }
    let s = samples.as_ptr();
    let full_groups = n_lags / 4;

    for g in 0..full_groups {
        let l0 = g * 4;
        let l3 = l0 + 3;

        for i in l0..l3.min(n) {
            let si = *s.add(i) as f64;
            r[l0]                          += si * *s.add(i - l0) as f64;
            if i > l0 { r[l0 + 1] += si * *s.add(i - l0 - 1) as f64; }
            if i >= l0 + 2 { r[l0 + 2] += si * *s.add(i - l0 - 2) as f64; }
        }

        let mut acc = _mm256_setzero_pd();
        for i in l3..n {
            let si_f = _mm256_set1_pd(*s.add(i) as f64);
            let s4i  = _mm_loadu_si128(s.add(i - l3) as *const _);
            let s4f  = _mm256_cvtepi32_pd(s4i);
            acc      = _mm256_add_pd(acc, _mm256_mul_pd(si_f, s4f));
        }

        let mut tmp = [0.0f64; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
        r[l0]     += tmp[3];
        r[l0 + 1] += tmp[2];
        r[l0 + 2] += tmp[1];
        r[l0 + 3] += tmp[0];
    }

    let base = full_groups * 4;
    for (lag, r) in r.iter_mut().enumerate().take(n_lags).skip(base) {
        let mut sum = 0.0f64;
        for i in lag..n {
            sum += *s.add(i) as f64 * *s.add(i - lag) as f64;
        }
        *r = sum;
    }

    r
}

#[cfg(target_arch = "x86_64")]
fn autocorrelation_simd(samples: &[i32], max_order: usize) -> Vec<f64> {
    // Prefer raw AVX2 intrinsics — no pre-conversion Vec<f64>, single load per 4 lags.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { autocorrelation_avx2_fma(samples, max_order) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { autocorrelation_avx2(samples, max_order) };
        }
    }
    autocorrelation_scalar(samples, max_order)
}

fn autocorrelation_scalar(samples: &[i32], max_order: usize) -> Vec<f64> {
    let n = samples.len();
    let mut r = vec![0.0f64; max_order + 1];
    for lag in 0..=max_order {
        let mut sum = 0.0f64;
        for i in lag..n {
            sum += samples[i] as f64 * samples[i - lag] as f64;
        }
        r[lag] = sum;
    }
    r
}

/// Levinson-Durbin recursion for LPC coefficient computation.
///
/// # Arguments
/// * `r` - Autocorrelation coefficients (r[0] must be > 0)
/// * `order` - Desired LPC order
///
/// # Returns
/// Tuple of (coefficients, error) where coefficients has length `order`
pub fn levinson_durbin(r: &[f64], order: usize) -> Result<(Vec<f64>, f64), FlacError> {
    if r.is_empty() || r[0] <= 0.0 {
        return Err(FlacError::LpcCoefficientOverflow);
    }

    let order = order.min(r.len() - 1);

    let mut a = vec![0.0f64; order];
    let mut a_prev = vec![0.0f64; order];
    let mut error = r[0];

    for i in 0..order {
        // Compute reflection coefficient
        let mut sum = r[i + 1];
        for j in 0..i {
            sum -= a_prev[j] * r[i - j];
        }

        if error.abs() < 1e-10 {
            // Near-zero error means signal is predictable
            break;
        }

        let k = sum / error;

        // Update coefficients
        a[i] = k;
        for j in 0..i {
            a[j] = a_prev[j] - k * a_prev[i - 1 - j];
        }

        // Update error
        error *= 1.0 - k * k;

        // Save for next iteration
        a_prev[..=i].copy_from_slice(&a[..=i]);
    }

    Ok((a, error))
}

/// Compute LPC coefficients for a block of samples.
///
/// # Arguments
/// * `samples` - Input samples
/// * `order` - Desired LPC order (1-32)
///
/// # Returns
/// Vector of LPC coefficients (length = order)
pub fn compute_lpc_coefficients(samples: &[i32], order: usize) -> Result<Vec<f64>, FlacError> {
    if order == 0 || order > MAX_LPC_ORDER {
        return Err(FlacError::InvalidLpcOrder { order: order as u8 });
    }

    if samples.len() <= order {
        return Err(FlacError::InvalidBlockSize {
            size: samples.len() as u32,
        });
    }

    let r = autocorrelation(samples, order);
    let (coeffs, _error) = levinson_durbin(&r, order)?;

    Ok(coeffs)
}

/// Quantize LPC coefficients for storage.
///
/// # Arguments
/// * `coeffs` - Floating-point LPC coefficients
/// * `precision` - Quantization precision in bits (typically 12-15)
///
/// # Returns
/// Tuple of (quantized_coeffs, shift) where:
/// - quantized_coeffs are scaled integer coefficients
/// - shift is the number of bits to shift right after multiplication
pub fn quantize_lpc_coefficients(
    coeffs: &[f64],
    precision: u8,
) -> Result<(Vec<i32>, i8), FlacError> {
    if coeffs.is_empty() {
        return Ok((vec![], 0));
    }

    // Find the maximum absolute coefficient
    let max_coeff = coeffs.iter().map(|c| c.abs()).fold(0.0f64, f64::max);

    if max_coeff < 1e-10 {
        // All coefficients are essentially zero
        return Ok((vec![0i32; coeffs.len()], 0));
    }

    // Compute optimal shift
    // We want: max_coeff * 2^shift < 2^(precision-1)
    // So: shift = precision - 1 - ceil(log2(max_coeff))
    let log2_max = max_coeff.log2();
    let shift = (precision as i32 - 1) - log2_max.ceil() as i32;

    // Clamp shift to valid range (-16 to 15 for FLAC)
    let shift = shift.clamp(-16, 15) as i8;

    // Quantize coefficients
    let scale = 2.0f64.powi(shift as i32);
    let max_val = (1i64 << (precision - 1)) - 1;
    let min_val = -(1i64 << (precision - 1));

    let mut quantized = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        let q = (c * scale).round() as i64;
        let clamped = q.clamp(min_val, max_val) as i32;
        quantized.push(clamped);
    }

    Ok((quantized, shift))
}

/// Apply LPC prediction and compute residuals.
///
/// # Arguments
/// * `samples` - Input samples
/// * `qlp_coeffs` - Quantized LPC coefficients
/// * `qlp_shift` - Right-shift amount for prediction
///
/// # Returns
/// Vector of residuals (length = samples.len() - order)
pub fn lpc_predictor_residual(
    samples: &[i32],
    qlp_coeffs: &[i32],
    qlp_shift: i8,
) -> Result<Vec<i32>, FlacError> {
    let order = qlp_coeffs.len();

    if order == 0 || order > MAX_LPC_ORDER {
        return Err(FlacError::InvalidLpcOrder { order: order as u8 });
    }

    if samples.len() < order {
        return Err(FlacError::InvalidBlockSize {
            size: samples.len() as u32,
        });
    }

    if qlp_shift < 0 {
        return Err(FlacError::InvalidLpcShift { shift: qlp_shift });
    }

    let n_out = samples.len() - order;
    let mut residuals = Vec::with_capacity(n_out);

    // AVX2 path: 4-wide lpc_residuals4::<ORDER> processes 4 residuals per iteration,
    // eliminating the hsum4 horizontal reduction that made the 1-wide version show no gain.
    // ORDER is a const generic so the inner j-loop is unrolled and inlined by LLVM.
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        use avx2_lpc::lpc_residuals4;
        let s = samples.as_ptr();
        let out: *mut i32 = residuals.as_mut_ptr();
        let shift = qlp_shift as i32;
        let c = qlp_coeffs.as_ptr();
        macro_rules! dispatch {
            ($N:literal) => { unsafe {
                lpc_residuals4::<$N>(s, c, shift, out, n_out);
                residuals.set_len(n_out);
            }};
        }
        match order {
            1  => dispatch!(1),  2  => dispatch!(2),  3  => dispatch!(3),
            4  => dispatch!(4),  5  => dispatch!(5),  6  => dispatch!(6),
            7  => dispatch!(7),  8  => dispatch!(8),  9  => dispatch!(9),
            10 => dispatch!(10), 11 => dispatch!(11), 12 => dispatch!(12),
            _ => {
                for i in order..samples.len() {
                    let mut pred: i64 = 0;
                    for (j, &c) in qlp_coeffs.iter().enumerate() {
                        pred += c as i64 * samples[i-1-j] as i64;
                    }
                    pred >>= qlp_shift;
                    residuals.push((samples[i] as i64 - pred) as i32);
                }
            }
        }
        return Ok(residuals);
    }

    macro_rules! lpc_residuals_unrolled {
        ($N:literal) => {{
            let c = &qlp_coeffs[..$N];
            for i in $N..samples.len() {
                let mut pred: i64 = 0;
                for j in 0..$N { pred += c[j] as i64 * samples[i - 1 - j] as i64; }
                pred >>= qlp_shift;
                residuals.push((samples[i] as i64 - pred) as i32);
            }
        }};
    }

    match order {
        1  => lpc_residuals_unrolled!(1),
        2  => lpc_residuals_unrolled!(2),
        3  => lpc_residuals_unrolled!(3),
        4  => lpc_residuals_unrolled!(4),
        5  => lpc_residuals_unrolled!(5),
        6  => lpc_residuals_unrolled!(6),
        7  => lpc_residuals_unrolled!(7),
        8  => lpc_residuals_unrolled!(8),
        12 => lpc_residuals_unrolled!(12),
        _  => {
            for i in order..samples.len() {
                let mut pred: i64 = 0;
                for (j, &c) in qlp_coeffs.iter().enumerate() {
                    pred += c as i64 * samples[i - 1 - j] as i64;
                }
                pred >>= qlp_shift;
                residuals.push((samples[i] as i64 - pred) as i32);
            }
        }
    }

    Ok(residuals)
}

/// Restore samples from residuals using LPC prediction.
///
/// # Arguments
/// * `warmup` - Warm-up samples (first `order` samples)
/// * `residuals` - Residual values
/// * `qlp_coeffs` - Quantized LPC coefficients
/// * `qlp_shift` - Right-shift amount for prediction
///
/// # Returns
/// Vector of restored samples (including warm-up)
pub fn lpc_predictor_restore(
    warmup: &[i32],
    residuals: &[i32],
    qlp_coeffs: &[i32],
    qlp_shift: i8,
) -> Result<Vec<i32>, FlacError> {
    let order = qlp_coeffs.len();

    if order == 0 || order > MAX_LPC_ORDER {
        return Err(FlacError::InvalidLpcOrder { order: order as u8 });
    }

    if warmup.len() < order {
        return Err(FlacError::InvalidBlockSize {
            size: warmup.len() as u32,
        });
    }

    if qlp_shift < 0 {
        return Err(FlacError::InvalidLpcShift { shift: qlp_shift });
    }

    let total = order + residuals.len();
    let mut samples = Vec::with_capacity(total);
    samples.extend_from_slice(&warmup[..order]);

    // Use the scalar multi-accumulator restore for all supported orders.
    // lpc_restore_fast is faster than the old SIMD path for the restore direction
    // because the serial carry dependency puts hsum4 (~8 cycles) on the critical
    // path when using AVX2. See the lpc_restore_fast doc-comment for details.
    let shift = qlp_shift as i32;
    let c = qlp_coeffs.as_ptr();
    samples.reserve(residuals.len());
    let buf = samples.as_mut_ptr();
    let res_ptr = residuals.as_ptr();
    let n = residuals.len();

    macro_rules! dispatch_restore {
        ($N:literal) => { unsafe {
            lpc_restore_fast::<$N>(buf, res_ptr, n, c, shift);
            samples.set_len(total);
        }};
    }

    match order {
        1  => dispatch_restore!(1),  2  => dispatch_restore!(2),
        3  => dispatch_restore!(3),  4  => dispatch_restore!(4),
        5  => dispatch_restore!(5),  6  => dispatch_restore!(6),
        7  => dispatch_restore!(7),  8  => dispatch_restore!(8),
        9  => dispatch_restore!(9),  10 => dispatch_restore!(10),
        11 => dispatch_restore!(11), 12 => dispatch_restore!(12),
        _ => {
            for &residual in residuals {
                let i = samples.len();
                let mut pred: i64 = 0;
                for (j, &c) in qlp_coeffs.iter().enumerate() {
                    pred += c as i64 * samples[i-1-j] as i64;
                }
                pred >>= qlp_shift;
                samples.push((pred + residual as i64) as i32);
            }
        }
    }

    Ok(samples)
}

/// All Levinson-Durbin outputs on the stack — no heap allocation.
///
/// `data[i][..=i]` holds the LPC coefficients for order `i+1`.
/// `errors[i]` is the Burg prediction error for order `i+1`.
/// `n` is the number of valid orders computed (≤ MAX_LPC_ORDER).
struct LdAllCoeffs {
    data:   [[f64; MAX_LPC_ORDER]; MAX_LPC_ORDER],
    errors: [f64; MAX_LPC_ORDER],
    n:      usize,
}

/// Run Levinson-Durbin up to max_order, returning all intermediate results on the stack.
///
/// More efficient than calling `levinson_durbin` separately for each order since
/// the algorithm is naturally incremental: order-k results build on order-(k-1).
/// Eliminates all heap allocation (previously ~12 Vecs per call).
fn levinson_durbin_all(r: &[f64], max_order: usize) -> Result<LdAllCoeffs, FlacError> {
    if r.is_empty() || r[0] <= 0.0 {
        return Err(FlacError::LpcCoefficientOverflow);
    }

    let order = max_order.min(r.len() - 1);
    let mut ld = LdAllCoeffs {
        data:   [[0.0f64; MAX_LPC_ORDER]; MAX_LPC_ORDER],
        errors: [0.0f64; MAX_LPC_ORDER],
        n:      order,
    };

    let mut a      = [0.0f64; MAX_LPC_ORDER];
    let mut a_prev = [0.0f64; MAX_LPC_ORDER];
    let mut error  = r[0];

    for i in 0..order {
        let mut sum = r[i + 1];
        for j in 0..i {
            sum -= a_prev[j] * r[i - j];
        }

        if error.abs() < 1e-10 {
            // Signal is fully predictable; fill remaining orders with frozen coeffs.
            for o in i..order {
                ld.data[o][..=i].copy_from_slice(&a[..=i]);
                ld.errors[o] = 0.0;
            }
            return Ok(ld);
        }

        let k = sum / error;
        a[i] = k;
        for j in 0..i {
            a[j] = a_prev[j] - k * a_prev[i - 1 - j];
        }
        error *= 1.0 - k * k;
        a_prev[..=i].copy_from_slice(&a[..=i]);

        ld.data[i][..=i].copy_from_slice(&a[..=i]);
        ld.errors[i] = error;
    }

    Ok(ld)
}

/// Find the best LPC order for given samples.
///
/// Tests orders from 1 to max_order and returns the one with minimum residual energy.
/// Uses a simple heuristic to avoid exhaustive search when not needed.
///
/// # Arguments
/// * `samples` - Input samples
/// * `max_order` - Maximum LPC order to test
/// * `qlp_precision` - Quantization precision for coefficients
/// * `exhaustive` - Whether to test all orders or use heuristics
pub fn find_best_lpc_order(
    samples: &[i32],
    max_order: usize,
    qlp_precision: u8,
    exhaustive: bool,
) -> Result<(usize, Vec<i32>, i8), FlacError> {
    let max_order = max_order.min(MAX_LPC_ORDER).min(samples.len() - 1);

    if max_order == 0 {
        return Ok((0, vec![], 0));
    }

    // Compute autocorrelation once for all orders
    let r = autocorrelation(samples, max_order);

    // Run Levinson-Durbin once up to max_order, capturing all intermediate results.
    // Returns stack-allocated LdAllCoeffs — zero heap allocation.
    let ld = levinson_durbin_all(&r, max_order)?;

    let mut best_order = 1;
    let mut best_coeffs = vec![];
    let mut best_shift = 0i8;
    let mut best_cost = f64::MAX;

    // Orders to test
    let orders: &[usize] = if exhaustive {
        // Stack array avoids Vec<usize> allocation for exhaustive mode.
        let mut arr = [0usize; MAX_LPC_ORDER];
        for (i, item) in arr.iter_mut().enumerate().take(max_order) { *item = i + 1; }
        return find_best_from_orders(qlp_precision, &r, &ld, &arr[..max_order]);
    } else {
        // Candidate orders (subset): original non-exhaustive heuristic
        match max_order {
            0 => return Ok((0, vec![], 0)),
            1 => &[1][..],
            2 => &[1, 2][..],
            3 => &[1, 2, 3][..],
            4 => &[1, 2, 4][..],
            5 => &[1, 2, 4, 5][..],
            6 => &[1, 2, 4, 6][..],
            7 => &[1, 2, 4, 6, 7][..],
            8 => &[1, 2, 4, 6, 8][..],
            9 => &[1, 2, 4, 6, 8, 9][..],
            10 => &[1, 2, 4, 6, 8, 10][..],
            11 => &[1, 2, 4, 6, 8, 10, 11][..],
            _ => &[1, 2, 4, 6, 8, 10, 12][..],  // max_order >= 12
        }
    };

    for &order in orders {
        if order > ld.n { continue; }
        let coeffs = &ld.data[order - 1][..order];
        let (qlp_coeffs, qlp_shift) = quantize_lpc_coefficients(coeffs, qlp_precision)?;
        let cost = lpc_order_cost(&r, coeffs, &qlp_coeffs, qlp_shift, ld.errors[order - 1], qlp_precision, order);
        if cost < best_cost {
            best_cost = cost;
            best_order = order;
            best_coeffs = qlp_coeffs;
            best_shift = qlp_shift;
        }
    }

    Ok((best_order, best_coeffs, best_shift))
}

/// Estimate LPC order cost using Burg prediction error + quantization noise.
/// Avoids O(N×ORDER) sample iteration — uses O(ORDER) autocorrelation values instead.
///
/// Cost = burg_error + Σ_j (δc_j)^2 × r[j+1] + coeff_bits × r[0] / N_scale
/// where δc_j = float_coeff[j] - quantized_coeff[j]/2^shift (quantization error per coeff).
#[inline]
fn lpc_order_cost(
    r: &[f64],
    float_coeffs: &[f64],
    qlp_coeffs: &[i32],
    qlp_shift: i8,
    burg_error: f64,
    qlp_precision: u8,
    order: usize,
) -> f64 {
    let scale = 2.0f64.powi(-(qlp_shift as i32));
    let quant_noise: f64 = float_coeffs.iter().zip(qlp_coeffs.iter()).enumerate()
        .map(|(j, (&fc, &qc))| {
            let delta = fc - qc as f64 * scale;
            delta * delta * r.get(j + 1).copied().unwrap_or(0.0)
        })
        .sum();
    // coeff_bits normalized to same scale as burg_error (variance units × N_approx)
    let coeff_penalty = (order * qlp_precision as usize) as f64 * r[0] * 1e-6;
    burg_error + quant_noise + coeff_penalty
}

fn find_best_from_orders(
    qlp_precision: u8,
    r: &[f64],
    ld: &LdAllCoeffs,
    orders: &[usize],
) -> Result<(usize, Vec<i32>, i8), FlacError> {
    let mut best_order = orders.first().copied().unwrap_or(1);
    let mut best_coeffs = vec![];
    let mut best_shift = 0i8;
    let mut best_cost = f64::MAX;

    for &order in orders {
        if order > ld.n { continue; }
        let coeffs = &ld.data[order - 1][..order];
        let (qlp_coeffs, qlp_shift) = quantize_lpc_coefficients(coeffs, qlp_precision)?;
        let cost = lpc_order_cost(r, coeffs, &qlp_coeffs, qlp_shift, ld.errors[order - 1], qlp_precision, order);
        if cost < best_cost {
            best_cost = cost;
            best_order = order;
            best_coeffs = qlp_coeffs;
            best_shift = qlp_shift;
        }
    }

    Ok((best_order, best_coeffs, best_shift))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_predictor_order_0() {
        let samples = vec![10, 20, 30, 40, 50];
        let residuals = fixed_predictor_residual(&samples, 0).unwrap();
        // Order 0: residual = sample (no prediction)
        assert_eq!(residuals, samples);
    }

    #[test]
    fn test_fixed_predictor_order_1() {
        let samples = vec![10, 20, 30, 40, 50];
        let residuals = fixed_predictor_residual(&samples, 1).unwrap();
        // Order 1: residual = sample - prev
        assert_eq!(residuals, vec![10, 10, 10, 10]); // All diffs are 10
    }

    #[test]
    fn test_fixed_predictor_order_2() {
        // Linear sequence: s[i] = 10 + 5*i
        // Second differences should be 0
        let samples = vec![10, 15, 20, 25, 30];
        let residuals = fixed_predictor_residual(&samples, 2).unwrap();
        assert_eq!(residuals, vec![0, 0, 0]); // Linear => zero second diff
    }

    #[test]
    fn test_fixed_predictor_roundtrip() {
        let original = vec![100, 150, 180, 200, 250, 280, 320];

        for order in 0..=4 {
            let residuals = fixed_predictor_residual(&original, order).unwrap();
            let restored = fixed_predictor_restore(&original[..order], &residuals, order).unwrap();
            assert_eq!(restored, original, "Roundtrip failed for order {}", order);
        }
    }

    #[test]
    fn test_autocorrelation() {
        let samples = vec![1, 2, 3, 2, 1];
        let r = autocorrelation(&samples, 2);

        // r[0] should be sum of squares
        assert!((r[0] - 19.0).abs() < 1e-10); // 1 + 4 + 9 + 4 + 1 = 19
    }

    #[test]
    fn test_levinson_durbin_basic() {
        // Simple test with known autocorrelation
        let r = vec![1.0, 0.5, 0.25];
        let (coeffs, _error) = levinson_durbin(&r, 2).unwrap();

        assert_eq!(coeffs.len(), 2);
        // First coefficient should be around 0.5 for first-order AR
    }

    #[test]
    fn test_quantize_coefficients() {
        let coeffs = vec![0.5, -0.25, 0.125];
        let (quantized, _shift) = quantize_lpc_coefficients(&coeffs, 12).unwrap();

        assert_eq!(quantized.len(), 3);
        // Quantized values should be reasonable
        for q in &quantized {
            assert!(q.abs() < (1 << 11)); // Within 12-bit signed range
        }
    }

    #[test]
    fn test_lpc_predictor_roundtrip() {
        // Create a simple signal
        let samples: Vec<i32> = (0..100)
            .map(|i| (100.0 * (i as f64 * 0.1).sin()) as i32)
            .collect();

        // Compute LPC coefficients
        let order = 4;
        let coeffs = compute_lpc_coefficients(&samples, order).unwrap();
        let (qlp_coeffs, qlp_shift) = quantize_lpc_coefficients(&coeffs, 12).unwrap();

        // Compute residuals
        let residuals = lpc_predictor_residual(&samples, &qlp_coeffs, qlp_shift).unwrap();

        // Restore samples
        let restored =
            lpc_predictor_restore(&samples[..order], &residuals, &qlp_coeffs, qlp_shift).unwrap();

        // Should match original
        assert_eq!(restored, samples);
    }

    #[test]
    fn test_find_best_fixed_order() {
        // Linear signal should prefer order 2
        let linear: Vec<i32> = (0..100).map(|i| i * 10).collect();
        let order = find_best_fixed_order(&linear);
        assert!(order >= 2, "Linear signal should use order >= 2");

        // Constant signal should prefer order 1
        let constant: Vec<i32> = vec![42; 100];
        let order = find_best_fixed_order(&constant);
        assert!(order >= 1, "Constant signal should use order >= 1");
    }
}
