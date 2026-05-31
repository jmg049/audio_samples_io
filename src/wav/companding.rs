//! G.711 companded audio: mu-law and a-law decode to 16-bit linear PCM.
//!
//! Telephony and VoIP WAV files commonly store 8-bit companded samples (format tags
//! `0x0006` a-law and `0x0007` mu-law). Companding maps an 8-bit code through a
//! piecewise-logarithmic curve so that low-amplitude samples — where the ear is most
//! sensitive — get finer quantisation, approaching ~14-bit linear quality in 8 bits.
//!
//! Decoding expands each byte back to a signed 16-bit linear sample using the standard
//! G.711 reference algorithms.

use crate::wav::FormatCode;

/// 8-bit companding scheme used by telephony WAV files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Companding {
    /// G.711 A-law (`WAVE_FORMAT_ALAW`, 0x0006) — common in Europe.
    ALaw,
    /// G.711 mu-law (`WAVE_FORMAT_MULAW`, 0x0007) — common in North America/Japan.
    MuLaw,
}

/// Expand one mu-law byte to a 16-bit linear PCM sample (Sun reference `ulaw2linear`).
const fn mulaw_to_linear(u_val: u8) -> i16 {
    const BIAS: i32 = 0x84;
    let u = !u_val as i32;
    let mantissa = (u & 0x0F) << 3;
    let exponent = (u & 0x70) >> 4;
    let magnitude = ((mantissa + BIAS) << exponent) - BIAS;
    if (u & 0x80) != 0 {
        -magnitude as i16
    } else {
        magnitude as i16
    }
}

/// Expand one a-law byte to a 16-bit linear PCM sample (Sun reference `alaw2linear`).
const fn alaw_to_linear(a_val: u8) -> i16 {
    let a = (a_val ^ 0x55) as i32;
    let mut magnitude = (a & 0x0F) << 4;
    let segment = (a & 0x70) >> 4;
    match segment {
        0 => magnitude += 8,
        1 => magnitude += 0x108,
        _ => {
            magnitude += 0x108;
            magnitude <<= segment - 1;
        },
    }
    if (a & 0x80) != 0 {
        magnitude as i16
    } else {
        -magnitude as i16
    }
}

/// Build a full 256-entry decode table at compile time.
const fn build_lut(mu: bool) -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = if mu {
            mulaw_to_linear(i as u8)
        } else {
            alaw_to_linear(i as u8)
        };
        i += 1;
    }
    table
}

static MULAW_LUT: [i16; 256] = build_lut(true);
static ALAW_LUT: [i16; 256] = build_lut(false);

impl Companding {
    /// Map a WAV format code to its companding scheme, if any.
    pub const fn from_format(format: FormatCode) -> Option<Self> {
        match format {
            FormatCode::ALaw => Some(Companding::ALaw),
            FormatCode::MuLaw => Some(Companding::MuLaw),
            _ => None,
        }
    }

    /// Decode a single companded byte to a 16-bit linear PCM sample.
    #[inline]
    pub fn decode(self, byte: u8) -> i16 {
        match self {
            Companding::MuLaw => MULAW_LUT[byte as usize],
            Companding::ALaw => ALAW_LUT[byte as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulaw_known_values() {
        // 0xFF is mu-law for 0 (digital silence); 0x00 maps to the largest magnitude.
        assert_eq!(Companding::MuLaw.decode(0xFF), 0);
        assert_eq!(Companding::MuLaw.decode(0x7F), 0);
        assert_eq!(Companding::MuLaw.decode(0x00), -32124);
        assert_eq!(Companding::MuLaw.decode(0x80), 32124);
    }

    #[test]
    fn alaw_known_values() {
        // a-law: 0xD5 ^ 0x55 = 0x80 → silence-ish; check symmetric extremes are large.
        assert_eq!(Companding::ALaw.decode(0xD5), 8);
        assert_eq!(Companding::ALaw.decode(0x55), -8);
        assert_eq!(Companding::ALaw.decode(0x2A), -32256);
        assert_eq!(Companding::ALaw.decode(0xAA), 32256);
    }

    #[test]
    fn from_format_maps_correctly() {
        assert_eq!(Companding::from_format(FormatCode::ALaw), Some(Companding::ALaw));
        assert_eq!(Companding::from_format(FormatCode::MuLaw), Some(Companding::MuLaw));
        assert_eq!(Companding::from_format(FormatCode::Pcm), None);
    }
}
