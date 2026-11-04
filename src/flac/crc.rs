//! CRC-8 and CRC-16 computation for FLAC frame validation.
//!
//! FLAC uses two CRC algorithms:
//! - CRC-8 (polynomial 0x07) for frame headers
//! - CRC-16 (polynomial 0x8005, bit-reversed as 0xA001) for complete frames
//!
//! Both are computed MSB-first with no reflection.

/// CRC-8 lookup table (polynomial 0x07)
const CRC8_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u8;
        let mut j = 0;
        while j < 8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// CRC-16 lookup table (polynomial 0x8005, MSB-first)
const CRC16_TABLE: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut j = 0;
        while j < 8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Advance CRC-16 state by one zero byte.
const fn crc16_advance1(x: u16) -> u16 {
    ((x & 0xFF) << 8) ^ CRC16_TABLE[(x >> 8) as usize]
}

/// Slicing-by-4 tables for CRC-16/IBM (0x8005, MSB-first).
///
/// Allows 4 bytes to be processed with 6 independent table lookups (no serial
/// dependency chain). Based on the linearity of CRC over GF(2):
///
///   CRC([b0,b1,b2,b3] from state c) =
///       M⁴·c  ⊕  M³·R(b0)  ⊕  M²·R(b1)  ⊕  M·R(b2)  ⊕  R(b3)
///
/// where M is the "multiply by x⁸ mod poly" matrix and R(b)=CRC16_TABLE[b].
///
/// State contribution: M⁴·c = TABLE_STATE_HI[c>>8] ⊕ TABLE_STATE_LO[c&0xFF]
/// Data contributions: TABLE_B0..B2 plus CRC16_TABLE itself for b3.
/// M⁴ applied to state (k << 8): advance high byte of CRC by 4 zero bytes.
const CRC16_TABLE_STATE_HI: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let x = crc16_advance1(crc16_advance1(crc16_advance1(crc16_advance1((i as u16) << 8))));
        t[i] = x;
        i += 1;
    }
    t
};

/// M⁴ applied to state k: advance low byte of CRC by 4 zero bytes.
const CRC16_TABLE_STATE_LO: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let x = crc16_advance1(crc16_advance1(crc16_advance1(crc16_advance1(i as u16))));
        t[i] = x;
        i += 1;
    }
    t
};

/// M³·R(b): contribution of data byte b at position 0 (3 bytes before end of chunk).
const CRC16_TABLE_B0: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let x = crc16_advance1(crc16_advance1(crc16_advance1(CRC16_TABLE[i])));
        t[i] = x;
        i += 1;
    }
    t
};

/// M²·R(b): contribution of data byte b at position 1.
const CRC16_TABLE_B1: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let x = crc16_advance1(crc16_advance1(CRC16_TABLE[i]));
        t[i] = x;
        i += 1;
    }
    t
};

/// M·R(b): contribution of data byte b at position 2.
const CRC16_TABLE_B2: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let x = crc16_advance1(CRC16_TABLE[i]);
        t[i] = x;
        i += 1;
    }
    t
};

/// CRC-8 calculator for FLAC frame headers.
///
/// # Example
///
/// ```
/// use audio_samples_io::flac::crc::Crc8;
///
/// let mut crc = Crc8::new();
/// crc.update(&[0xFF, 0xF8, 0x69, 0x02]);
/// let checksum = crc.finalize();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Crc8 {
    crc: u8,
}

impl Crc8 {
    /// Create a new CRC-8 calculator initialized to 0.
    #[inline]
    pub const fn new() -> Self {
        Crc8 { crc: 0 }
    }

    /// Update the CRC with a single byte.
    #[inline]
    pub const fn update_byte(&mut self, byte: u8) {
        self.crc = CRC8_TABLE[(self.crc ^ byte) as usize];
    }

    /// Update the CRC with a slice of bytes.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.update_byte(byte);
        }
    }

    /// Finalize and return the CRC value.
    #[inline]
    pub const fn finalize(self) -> u8 {
        self.crc
    }

    /// Reset the CRC to initial state.
    #[inline]
    pub const fn reset(&mut self) {
        self.crc = 0;
    }

    /// Compute CRC-8 for a byte slice in one call.
    #[inline]
    pub fn compute(data: &[u8]) -> u8 {
        let mut crc = Self::new();
        crc.update(data);
        crc.finalize()
    }
}

impl Default for Crc8 {
    fn default() -> Self {
        Self::new()
    }
}
/// ```
/// use audio_samples_io::flac::crc::Crc16;
///
/// let frame_bytes: &[u8] = b"...";
/// let mut crc = Crc16::new();
/// crc.update(frame_bytes);
/// let checksum = crc.finalize();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Crc16 {
    crc: u16,
}

impl Crc16 {
    /// Create a new CRC-16 calculator initialized to 0.
    #[inline]
    pub const fn new() -> Self {
        Crc16 { crc: 0 }
    }

    /// Update the CRC with a single byte.
    #[inline]
    pub const fn update_byte(&mut self, byte: u8) {
        self.crc = (self.crc << 8) ^ CRC16_TABLE[((self.crc >> 8) as u8 ^ byte) as usize];
    }

    /// Update the CRC with a slice of bytes.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.update_byte(byte);
        }
    }

    /// Finalize and return the CRC value.
    #[inline]
    pub const fn finalize(self) -> u16 {
        self.crc
    }

    /// Reset the CRC to initial state.
    #[inline]
    pub const fn reset(&mut self) {
        self.crc = 0;
    }

    /// Compute CRC-16 for a byte slice: by carry-less multiplication where the CPU has it, otherwise
    /// with slicing-by-4 (4 bytes per iteration, 6 independent table lookups).
    #[inline]
    pub fn compute(data: &[u8]) -> u16 {
        #[cfg(target_arch = "x86_64")]
        if data.len() >= 64 && is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("ssse3") {
            // SAFETY: the features were just detected.
            return unsafe { crc16_fold(data) };
        }
        let mut crc: u16 = 0;
        let mut chunks = data.chunks_exact(4);
        for chunk in &mut chunks {
            crc = CRC16_TABLE_STATE_HI[(crc >> 8) as usize]
                ^ CRC16_TABLE_STATE_LO[(crc & 0xFF) as usize]
                ^ CRC16_TABLE_B0[chunk[0] as usize]
                ^ CRC16_TABLE_B1[chunk[1] as usize]
                ^ CRC16_TABLE_B2[chunk[2] as usize]
                ^ CRC16_TABLE[chunk[3] as usize];
        }
        for &byte in chunks.remainder() {
            crc = (crc << 8) ^ CRC16_TABLE[((crc >> 8) as u8 ^ byte) as usize];
        }
        crc
    }
}

impl Default for Crc16 {
    fn default() -> Self {
        Self::new()
    }
}

/// x^n mod the CRC-16 polynomial.
const fn x_pow_mod(n: u32) -> u64 {
    let mut r: u32 = 1;
    let mut i = 0;
    while i < n {
        r <<= 1;
        if r & 0x1_0000 != 0 {
            r ^= 0x1_8005;
        }
        i += 1;
    }
    r as u64
}

/// CRC-16 of `data` by folding: 16-byte blocks are multiplied forward with carry-less
/// multiplication, which keeps their value modulo the polynomial, and the table finishes the
/// last block and the tail.
///
/// # Safety
///
/// The CPU must support PCLMULQDQ and SSSE3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,ssse3,sse2")]
unsafe fn crc16_fold(data: &[u8]) -> u16 {
    use std::arch::x86_64::*;

    const FOLD_1: (u64, u64) = (x_pow_mod(128 + 64), x_pow_mod(128));
    const FOLD_4: (u64, u64) = (x_pow_mod(512 + 64), x_pow_mod(512));

    let swap = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    // SAFETY (all loads): callers pass offsets at least 16 bytes before the end of `data`.
    let load = |at: usize| unsafe { _mm_shuffle_epi8(_mm_loadu_si128(data.as_ptr().add(at).cast()), swap) };
    let fold = |a: __m128i, k: __m128i, next: __m128i| {
        // High half with k's low lane (x^(n + 64)), low half with its high lane (x^n).
        let hi = _mm_clmulepi64_si128::<0x01>(a, k);
        let lo = _mm_clmulepi64_si128::<0x10>(a, k);
        _mm_xor_si128(_mm_xor_si128(hi, lo), next)
    };

    let mut at = 16usize;
    let mut a = load(0);
    if data.len() >= 128 {
        let k4 = _mm_set_epi64x(FOLD_4.1 as i64, FOLD_4.0 as i64);
        let (mut b, mut c, mut d) = (load(16), load(32), load(48));
        at = 64;
        while at + 64 <= data.len() {
            a = fold(a, k4, load(at));
            b = fold(b, k4, load(at + 16));
            c = fold(c, k4, load(at + 32));
            d = fold(d, k4, load(at + 48));
            at += 64;
        }
        let k1 = _mm_set_epi64x(FOLD_1.1 as i64, FOLD_1.0 as i64);
        a = fold(a, k1, b);
        a = fold(a, k1, c);
        a = fold(a, k1, d);
    }
    let k1 = _mm_set_epi64x(FOLD_1.1 as i64, FOLD_1.0 as i64);
    while at + 16 <= data.len() {
        a = fold(a, k1, load(at));
        at += 16;
    }
    let mut block = [0u8; 16];
    // SAFETY: `block` holds 16 bytes.
    unsafe { _mm_storeu_si128(block.as_mut_ptr().cast(), _mm_shuffle_epi8(a, swap)) };
    let mut crc = Crc16::new();
    crc.update(&block);
    crc.update(&data[at..]);
    crc.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc16_fold_matches_table() {
        let data: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        for len in [
            0usize, 1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 200, 255, 256, 1000, 4096,
        ] {
            let mut table = Crc16::new();
            table.update(&data[..len]);
            assert_eq!(Crc16::compute(&data[..len]), table.finalize(), "length {len}");
        }
    }

    #[test]
    fn test_crc8_empty() {
        assert_eq!(Crc8::compute(&[]), 0);
    }

    #[test]
    fn test_crc8_single_byte() {
        // Known test vector
        assert_eq!(Crc8::compute(&[0x00]), 0x00);
        assert_eq!(Crc8::compute(&[0x01]), 0x07);
        assert_eq!(Crc8::compute(&[0x02]), 0x0E);
    }

    #[test]
    fn test_crc8_incremental() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let single = Crc8::compute(&data);

        let mut incremental = Crc8::new();
        incremental.update(&data[..2]);
        incremental.update(&data[2..]);

        assert_eq!(single, incremental.finalize());
    }

    #[test]
    fn test_crc16_empty() {
        assert_eq!(Crc16::compute(&[]), 0);
    }

    #[test]
    fn test_crc16_single_byte() {
        // Known test vectors for CRC-16 with polynomial 0x8005
        assert_eq!(Crc16::compute(&[0x00]), 0x0000);
    }

    #[test]
    fn test_crc16_incremental() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let single = Crc16::compute(&data);

        let mut incremental = Crc16::new();
        incremental.update(&data[..4]);
        incremental.update(&data[4..]);

        assert_eq!(single, incremental.finalize());
    }

    #[test]
    fn test_crc8_reset() {
        let mut crc = Crc8::new();
        crc.update(&[0x01, 0x02]);
        let first = crc.finalize();

        crc.reset();
        crc.update(&[0x01, 0x02]);
        let second = crc.finalize();

        assert_eq!(first, second);
    }

    #[test]
    fn test_crc16_reset() {
        let mut crc = Crc16::new();
        crc.update(&[0x01, 0x02, 0x03]);
        let first = crc.finalize();

        crc.reset();
        crc.update(&[0x01, 0x02, 0x03]);
        let second = crc.finalize();

        assert_eq!(first, second);
    }

    #[test]
    fn test_crc8_table_generation() {
        // Verify table is correctly generated
        // First entry should be 0 (CRC of 0x00 XOR'd with CRC 0 = 0)
        assert_eq!(CRC8_TABLE[0], 0);
        // Entry 1 should be polynomial (0x07) since MSB of 0x01 << 7 is 0
        // Actually for i=1: initial = 0x01, after 8 shifts with poly 0x07
        // Let's verify by manual calculation
        let mut crc: u8 = 0x01;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
        assert_eq!(CRC8_TABLE[1], crc);
    }

    #[test]
    fn test_crc16_table_generation() {
        // Verify table is correctly generated
        assert_eq!(CRC16_TABLE[0], 0);

        // Manual verification for entry 1
        let mut crc: u16 = 0x0100; // 1 << 8
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
        assert_eq!(CRC16_TABLE[1], crc);
    }
}
