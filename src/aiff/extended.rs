//! IEEE 754 80-bit extended-precision float, the sample-rate encoding AIFF
//! inherited from the Apple II era.
//!
//! Layout (big-endian): 1 sign bit, 15 exponent bits (bias 16383), 64 mantissa
//! bits with an *explicit* integer bit (no hidden leading 1, unlike f32/f64).

/// Encode an `f64` as an 80-bit extended float.
///
/// Exact for every value a sample rate can take (integers well below 2^53).
pub fn encode_extended(value: f64) -> [u8; 10] {
    if value == 0.0 {
        return [0u8; 10];
    }

    let bits = value.to_bits();
    let sign = ((bits >> 63) as u16) << 15;
    let exp_f64 = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;

    // Subnormals and non-finite values cannot occur for sane sample rates;
    // map them to zero rather than encoding garbage.
    if exp_f64 == 0 || exp_f64 == 0x7FF {
        return [0u8; 10];
    }

    // f64 value = 1.frac × 2^(exp_f64 - 1023). The extended mantissa stores the
    // leading 1 explicitly in bit 63, with the fraction left-aligned below it.
    let exponent = (exp_f64 - 1023 + 16383) as u16 | sign;
    let mantissa: u64 = (1u64 << 63) | (frac << 11);

    let mut out = [0u8; 10];
    out[0..2].copy_from_slice(&exponent.to_be_bytes());
    out[2..10].copy_from_slice(&mantissa.to_be_bytes());
    out
}

/// Decode an 80-bit extended float to `f64`.
pub fn decode_extended(bytes: &[u8; 10]) -> f64 {
    let exponent_field = u16::from_be_bytes([bytes[0], bytes[1]]);
    let mantissa = u64::from_be_bytes(bytes[2..10].try_into().expect("8-byte slice"));

    if mantissa == 0 && exponent_field & 0x7FFF == 0 {
        return 0.0;
    }

    let sign = if exponent_field & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (exponent_field & 0x7FFF) as i32 - 16383;

    // mantissa is a fixed-point number with the binary point after bit 63.
    sign * (mantissa as f64) * ((exponent - 63) as f64).exp2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_common_sample_rates() {
        for rate in [8_000.0, 11_025.0, 22_050.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let encoded = encode_extended(rate);
            assert_eq!(decode_extended(&encoded), rate, "rate {rate}");
        }
    }

    #[test]
    fn matches_known_encoding_of_44100() {
        // The canonical bytes every AIFF reference uses for 44.1 kHz.
        let expected: [u8; 10] = [0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(encode_extended(44_100.0), expected);
        assert_eq!(decode_extended(&expected), 44_100.0);
    }

    #[test]
    fn zero_encodes_to_zero() {
        assert_eq!(encode_extended(0.0), [0u8; 10]);
        assert_eq!(decode_extended(&[0u8; 10]), 0.0);
    }
}
