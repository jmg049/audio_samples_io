//! Linear Predictive Coding (LPC) for FLAC.
//!
//! This module implements:
//! - Fixed predictors (orders 0-4) using predefined coefficients
//! - LPC analysis using the Levinson-Durbin algorithm
//! - Quantized LPC coefficient encoding/decoding
//! - Prediction and residual computation

use crate::flac::constants::{FIXED_COEFFICIENTS, MAX_FIXED_ORDER, MAX_LPC_ORDER};
use crate::flac::error::FlacError;

/// Apply a fixed predictor and compute residuals.
///
/// Fixed predictors use predefined integer coefficients:
/// - Order 0: residual[i] = sample[i] (constant prediction of 0)
/// - Order 1: residual[i] = sample[i] - sample[i-1]
/// - Order 2: residual[i] = sample[i] - 2*sample[i-1] + sample[i-2]
/// - Order 3: residual[i] = sample[i] - 3*sample[i-1] + 3*sample[i-2] - sample[i-3]
/// - Order 4: residual[i] = sample[i] - 4*sample[i-1] + 6*sample[i-2] - 4*sample[i-3] + sample[i-4]
///
/// # Arguments
/// * `samples` - Input samples (warm-up samples followed by samples to predict)
/// * `order` - Predictor order (0-4)
///
/// # Returns
/// Vector of residuals (length = samples.len() - order)
pub fn fixed_predictor_residual(samples: &[i32], order: usize) -> Result<Vec<i32>, FlacError> {
    if order > MAX_FIXED_ORDER {
        return Err(FlacError::InvalidFixedOrder { order: order as u8 });
    }

    if samples.len() < order {
        return Err(FlacError::InvalidBlockSize {
            size: samples.len() as u32,
        });
    }

    let coeffs = &FIXED_COEFFICIENTS[order];
    let mut residuals = Vec::with_capacity(samples.len() - order);

    for i in order..samples.len() {
        let mut prediction: i64 = 0;
        for (j, &coeff) in coeffs[..order].iter().enumerate() {
            prediction += coeff as i64 * samples[i - 1 - j] as i64;
        }
        residuals.push((samples[i] as i64 - prediction) as i32);
    }

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

    let coeffs = &FIXED_COEFFICIENTS[order];
    let mut samples = Vec::with_capacity(warmup.len() + residuals.len());
    samples.extend_from_slice(&warmup[..order]);

    for &residual in residuals {
        let i = samples.len();
        let mut prediction: i64 = 0;
        for (j, &coeff) in coeffs[..order].iter().enumerate() {
            prediction += coeff as i64 * samples[i - 1 - j] as i64;
        }
        samples.push((prediction + residual as i64) as i32);
    }

    Ok(samples)
}

/// Find the best fixed predictor order for given samples.
///
/// Tests all orders 0-4 and returns the one with minimum residual energy.
pub fn find_best_fixed_order(samples: &[i32]) -> usize {
    if samples.len() <= 4 {
        return 0;
    }

    let mut best_order = 0;
    let mut best_energy = u64::MAX;

    for order in 0..=MAX_FIXED_ORDER.min(samples.len() - 1) {
        if let Ok(residuals) = fixed_predictor_residual(samples, order) {
            let energy: u64 = residuals.iter().map(|&r| (r as i64).unsigned_abs()).sum();
            if energy < best_energy {
                best_energy = energy;
                best_order = order;
            }
        }
    }

    best_order
}

/// Compute autocorrelation coefficients.
///
/// # Arguments
/// * `samples` - Input samples
/// * `max_order` - Maximum lag to compute (0..=max_order)
///
/// # Returns
/// Vector of autocorrelation coefficients r[0..=max_order]
pub fn autocorrelation(samples: &[i32], max_order: usize) -> Vec<f64> {
    let n = samples.len();
    let mut r = vec![0.0f64; max_order + 1];

    // Convert to f64 for precision
    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

    for lag in 0..=max_order {
        let mut sum = 0.0f64;
        for i in lag..n {
            sum += x[i] * x[i - lag];
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

    let mut residuals = Vec::with_capacity(samples.len() - order);

    for i in order..samples.len() {
        let mut prediction: i64 = 0;
        for (j, &coeff) in qlp_coeffs.iter().enumerate() {
            prediction += coeff as i64 * samples[i - 1 - j] as i64;
        }
        prediction >>= qlp_shift;
        residuals.push((samples[i] as i64 - prediction) as i32);
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

    let mut samples = Vec::with_capacity(order + residuals.len());
    samples.extend_from_slice(&warmup[..order]);

    for &residual in residuals {
        let i = samples.len();
        let mut prediction: i64 = 0;
        for (j, &coeff) in qlp_coeffs.iter().enumerate() {
            prediction += coeff as i64 * samples[i - 1 - j] as i64;
        }
        prediction >>= qlp_shift;
        samples.push((prediction + residual as i64) as i32);
    }

    Ok(samples)
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

    // Compute autocorrelation for maximum order
    let r = autocorrelation(samples, max_order);

    let mut best_order = 1;
    let mut best_coeffs = vec![];
    let mut best_shift = 0i8;
    let mut best_bits = u64::MAX;

    // Orders to test
    let orders: Vec<usize> = if exhaustive {
        (1..=max_order).collect()
    } else {
        // Test a subset of orders
        let mut o = vec![1];
        if max_order >= 2 {
            o.push(2);
        }
        if max_order >= 4 {
            o.push(4);
        }
        if max_order >= 6 {
            o.push(6);
        }
        if max_order >= 8 {
            o.push(8);
        }
        if max_order >= 10 {
            o.push(10);
        }
        if max_order >= 12 {
            o.push(12);
        }
        o
    };

    for order in orders {
        // Compute LPC coefficients using Levinson-Durbin
        let (coeffs, _error) = levinson_durbin(&r, order)?;

        // Quantize coefficients
        let (qlp_coeffs, qlp_shift) = quantize_lpc_coefficients(&coeffs, qlp_precision)?;

        // Compute residuals
        let residuals = lpc_predictor_residual(samples, &qlp_coeffs, qlp_shift)?;

        // Estimate bits needed (rough estimate)
        let residual_energy: u64 = residuals.iter().map(|&r| (r as i64).unsigned_abs()).sum();

        // Add overhead for storing coefficients
        let coeff_bits = order as u64 * qlp_precision as u64;
        let estimated_bits = residual_energy + coeff_bits;

        if estimated_bits < best_bits {
            best_bits = estimated_bits;
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
        let (quantized, shift) = quantize_lpc_coefficients(&coeffs, 12).unwrap();

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
