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

    /// Compute CRC-16 for a byte slice using slicing-by-4 (4 bytes per iteration).
    ///
    /// Processes 4 bytes at a time with 6 independent table lookups — no serial
    /// dependency chain — giving ~2-3× throughput vs the byte-by-byte loop.
    #[inline]
    pub fn compute(data: &[u8]) -> u16 {
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

#[cfg(test)]
mod tests {
    use super::*;

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
