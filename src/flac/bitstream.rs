//! Bitstream reader and writer for FLAC encoding/decoding.
//!
//! FLAC uses big-endian (MSB-first) bit ordering for all bitstream operations.

use crate::flac::error::FlacError;

/// Bitstream reader for decoding FLAC data.
///
/// Reads bits from a byte slice in big-endian (MSB-first) order.
#[derive(Debug)]
pub struct BitReader<'a> {
    /// Source bytes
    data: &'a [u8],
    /// Current byte position
    byte_pos: usize,
    /// Current bit position within the byte (0-7, 0 = MSB)
    bit_pos: u8,
    /// Total bits consumed (for tracking)
    bits_consumed: usize,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader from a byte slice.
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
            bits_consumed: 0,
        }
    }

    /// Get the current byte position.
    #[inline]
    pub const fn byte_position(&self) -> usize {
        self.byte_pos
    }

    /// Get the current bit position within the current byte.
    #[inline]
    pub const fn bit_position(&self) -> u8 {
        self.bit_pos
    }

    /// Get the total number of bits consumed.
    #[inline]
    pub const fn bits_consumed(&self) -> usize {
        self.bits_consumed
    }

    /// Check if the reader is byte-aligned.
    #[inline]
    pub const fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Get remaining bytes (only valid when byte-aligned).
    #[inline]
    pub fn remaining_bytes(&self) -> &'a [u8] {
        if self.bit_pos == 0 {
            &self.data[self.byte_pos..]
        } else {
            &self.data[self.byte_pos + 1..]
        }
    }

    /// Get the total number of bits remaining.
    #[inline]
    pub fn bits_remaining(&self) -> usize {
        let full_bytes = self.data.len().saturating_sub(self.byte_pos);
        if full_bytes == 0 {
            0
        } else {
            full_bytes * 8 - self.bit_pos as usize
        }
    }

    /// Skip to the next byte boundary.
    #[inline]
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bits_consumed += (8 - self.bit_pos) as usize;
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }

    /// Read a single bit.
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool, FlacError> {
        if self.byte_pos >= self.data.len() {
            return Err(FlacError::UnexpectedEof);
        }

        let byte = self.data[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;

        self.bit_pos += 1;
        self.bits_consumed += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit != 0)
    }

    /// Read up to 32 bits as an unsigned value.
    #[inline]
    pub fn read_bits(&mut self, count: u8) -> Result<u32, FlacError> {
        debug_assert!(count <= 32, "Cannot read more than 32 bits at once");

        if count == 0 {
            return Ok(0);
        }

        let mut result: u32 = 0;
        let mut remaining = count;

        while remaining > 0 {
            if self.byte_pos >= self.data.len() {
                return Err(FlacError::UnexpectedEof);
            }

            let byte = self.data[self.byte_pos];
            let bits_in_byte = 8 - self.bit_pos;
            let bits_to_read = remaining.min(bits_in_byte);

            // Extract bits from current byte
            let shift = bits_in_byte - bits_to_read;
            // Handle full byte case specially to avoid overflow
            let mask = if bits_to_read == 8 {
                0xFF
            } else {
                (1u8 << bits_to_read) - 1
            };
            let bits = (byte >> shift) & mask;

            result = (result << bits_to_read) | bits as u32;

            self.bit_pos += bits_to_read;
            self.bits_consumed += bits_to_read as usize;
            remaining -= bits_to_read;

            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Ok(result)
    }

    /// Read up to 64 bits as an unsigned value.
    #[inline]
    pub fn read_bits_u64(&mut self, count: u8) -> Result<u64, FlacError> {
        debug_assert!(count <= 64, "Cannot read more than 64 bits at once");

        if count <= 32 {
            return Ok(self.read_bits(count)? as u64);
        }

        // Read in two parts
        let high_bits = count - 32;
        let high = self.read_bits(high_bits)? as u64;
        let low = self.read_bits(32)? as u64;

        Ok((high << 32) | low)
    }

    /// Read a signed value using two's complement.
    #[inline]
    pub fn read_bits_signed(&mut self, count: u8) -> Result<i32, FlacError> {
        let unsigned = self.read_bits(count)?;

        // Sign-extend if the sign bit is set
        if count > 0 && count < 32 && (unsigned & (1 << (count - 1))) != 0 {
            // Sign bit is set, extend it
            let mask = !((1u32 << count) - 1);
            Ok((unsigned | mask) as i32)
        } else {
            Ok(unsigned as i32)
        }
    }

    /// Read a unary coded value (count of 1s followed by 0, or vice versa).
    ///
    /// For FLAC Rice coding, we count 0s until we hit a 1.
    #[inline]
    pub fn read_unary(&mut self) -> Result<u32, FlacError> {
        let mut count = 0u32;

        loop {
            if self.byte_pos >= self.data.len() {
                return Err(FlacError::UnexpectedEof);
            }

            let byte = self.data[self.byte_pos];

            // Check remaining bits in current byte
            while self.bit_pos < 8 {
                let bit = (byte >> (7 - self.bit_pos)) & 1;
                self.bit_pos += 1;
                self.bits_consumed += 1;

                if bit != 0 {
                    // Found the terminating 1
                    if self.bit_pos == 8 {
                        self.bit_pos = 0;
                        self.byte_pos += 1;
                    }
                    return Ok(count);
                }
                count += 1;
            }

            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Read a UTF-8 coded value (used for frame/sample numbers in FLAC).
    ///
    /// Returns the decoded value and the number of bytes consumed.
    pub fn read_utf8_coded(&mut self) -> Result<u64, FlacError> {
        if self.byte_pos >= self.data.len() {
            return Err(FlacError::UnexpectedEof);
        }

        let first = self.read_bits(8)? as u8;

        // Count leading 1s to determine length
        let leading_ones = first.leading_ones() as usize;

        match leading_ones {
            0 => {
                // Single byte: 0xxxxxxx
                Ok(first as u64)
            }
            1 => {
                // Invalid: continuation byte as first byte
                Err(FlacError::InvalidUtf8CodedNumber)
            }
            2..=6 => {
                // Multi-byte: 110xxxxx, 1110xxxx, 11110xxx, 111110xx, 1111110x
                let mask = 0x7F >> leading_ones;
                let mut value = (first & mask) as u64;

                for _ in 1..leading_ones {
                    if self.byte_pos >= self.data.len() {
                        return Err(FlacError::UnexpectedEof);
                    }

                    let cont = self.read_bits(8)? as u8;

                    // Continuation bytes must be 10xxxxxx
                    if cont & 0xC0 != 0x80 {
                        return Err(FlacError::InvalidUtf8CodedNumber);
                    }

                    value = (value << 6) | (cont & 0x3F) as u64;
                }

                Ok(value)
            }
            7 => {
                // 7 leading ones: 11111110 followed by 6 continuation bytes
                let mut value = 0u64;

                for _ in 0..6 {
                    if self.byte_pos >= self.data.len() {
                        return Err(FlacError::UnexpectedEof);
                    }

                    let cont = self.read_bits(8)? as u8;

                    if cont & 0xC0 != 0x80 {
                        return Err(FlacError::InvalidUtf8CodedNumber);
                    }

                    value = (value << 6) | (cont & 0x3F) as u64;
                }

                Ok(value)
            }
            _ => {
                // 8 leading ones (0xFF) is invalid
                Err(FlacError::InvalidUtf8CodedNumber)
            }
        }
    }

    /// Read raw bytes (must be byte-aligned).
    pub fn read_bytes(&mut self, count: usize) -> Result<&'a [u8], FlacError> {
        if self.bit_pos != 0 {
            return Err(FlacError::BitstreamError(
                "read_bytes requires byte alignment".to_string(),
            ));
        }

        if self.byte_pos + count > self.data.len() {
            return Err(FlacError::UnexpectedEof);
        }

        let bytes = &self.data[self.byte_pos..self.byte_pos + count];
        self.byte_pos += count;
        self.bits_consumed += count * 8;

        Ok(bytes)
    }

    /// Peek at the next N bits without consuming them.
    pub fn peek_bits(&self, count: u8) -> Result<u32, FlacError> {
        let mut clone = BitReader {
            data: self.data,
            byte_pos: self.byte_pos,
            bit_pos: self.bit_pos,
            bits_consumed: self.bits_consumed,
        };
        clone.read_bits(count)
    }
}

/// Bitstream writer for encoding FLAC data.
///
/// Writes bits to a byte vector in big-endian (MSB-first) order.
#[derive(Debug)]
pub struct BitWriter {
    /// Output bytes
    data: Vec<u8>,
    /// Current byte being filled
    current_byte: u8,
    /// Current bit position within the byte (0-7, 0 = MSB)
    bit_pos: u8,
    /// Total bits written
    bits_written: usize,
}

impl BitWriter {
    /// Create a new bit writer.
    #[inline]
    pub fn new() -> Self {
        BitWriter {
            data: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
            bits_written: 0,
        }
    }

    /// Create a new bit writer with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(bytes: usize) -> Self {
        BitWriter {
            data: Vec::with_capacity(bytes),
            current_byte: 0,
            bit_pos: 0,
            bits_written: 0,
        }
    }

    /// Get the total number of bits written.
    #[inline]
    pub const fn bits_written(&self) -> usize {
        self.bits_written
    }

    /// Check if the writer is byte-aligned.
    #[inline]
    pub const fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Get the current byte position (complete bytes written).
    #[inline]
    pub fn byte_position(&self) -> usize {
        self.data.len()
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
        self.bits_written += 1;

        if self.bit_pos == 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Write up to 32 bits from an unsigned value.
    #[inline]
    pub fn write_bits(&mut self, value: u32, count: u8) {
        debug_assert!(count <= 32, "Cannot write more than 32 bits at once");

        if count == 0 {
            return;
        }

        // Ensure value fits in count bits
        debug_assert!(
            count == 32 || value < (1 << count),
            "Value {} does not fit in {} bits",
            value,
            count
        );

        let mut remaining = count;
        let mut val = value;

        while remaining > 0 {
            let bits_available = 8 - self.bit_pos;
            let bits_to_write = remaining.min(bits_available);

            // Extract the top bits_to_write bits from val
            let shift = remaining - bits_to_write;
            let mask = (1u32 << bits_to_write) - 1;
            let bits = ((val >> shift) & mask) as u8;

            // Position them in the current byte
            self.current_byte |= bits << (bits_available - bits_to_write);

            self.bit_pos += bits_to_write;
            self.bits_written += bits_to_write as usize;
            remaining -= bits_to_write;

            // Clear the bits we've written from val
            if shift > 0 {
                val &= (1 << shift) - 1;
            }

            if self.bit_pos == 8 {
                self.data.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    /// Write up to 64 bits from an unsigned value.
    #[inline]
    pub fn write_bits_u64(&mut self, value: u64, count: u8) {
        debug_assert!(count <= 64, "Cannot write more than 64 bits at once");

        if count <= 32 {
            self.write_bits(value as u32, count);
        } else {
            // Write in two parts
            let high_bits = count - 32;
            self.write_bits((value >> 32) as u32, high_bits);
            self.write_bits(value as u32, 32);
        }
    }

    /// Write a signed value.
    #[inline]
    pub fn write_bits_signed(&mut self, value: i32, count: u8) {
        // Two's complement representation
        let unsigned = value as u32;
        let mask = if count < 32 {
            (1u32 << count) - 1
        } else {
            u32::MAX
        };
        self.write_bits(unsigned & mask, count);
    }

    /// Write a unary coded value (count zeros followed by a one).
    #[inline]
    pub fn write_unary(&mut self, value: u32) {
        // Write 'value' zeros
        for _ in 0..value {
            self.write_bit(false);
        }
        // Write terminating one
        self.write_bit(true);
    }

    /// Write a UTF-8 coded value.
    pub fn write_utf8_coded(&mut self, value: u64) {
        if value < 0x80 {
            // Single byte
            self.write_bits(value as u32, 8);
        } else if value < 0x800 {
            // Two bytes
            self.write_bits((0xC0 | (value >> 6)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x10000 {
            // Three bytes
            self.write_bits((0xE0 | (value >> 12)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x200000 {
            // Four bytes
            self.write_bits((0xF0 | (value >> 18)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x4000000 {
            // Five bytes
            self.write_bits((0xF8 | (value >> 24)) as u32, 8);
            self.write_bits((0x80 | ((value >> 18) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x80000000 {
            // Six bytes
            self.write_bits((0xFC | (value >> 30)) as u32, 8);
            self.write_bits((0x80 | ((value >> 24) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 18) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else {
            // Seven bytes (maximum for FLAC)
            self.write_bits(0xFE, 8);
            self.write_bits((0x80 | ((value >> 30) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 24) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 18) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        }
    }

    /// Write raw bytes (must be byte-aligned).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert!(
            self.is_byte_aligned(),
            "write_bytes requires byte alignment"
        );

        self.data.extend_from_slice(bytes);
        self.bits_written += bytes.len() * 8;
    }

    /// Pad to byte boundary with zeros.
    #[inline]
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            let padding = 8 - self.bit_pos;
            self.bits_written += padding as usize;
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Consume the writer and return the byte vector.
    ///
    /// Automatically pads to byte boundary if needed.
    #[inline]
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.data
    }

    /// Get a reference to the written data so far (excluding partial byte).
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get all data including partial byte (for CRC computation).
    pub fn data_with_partial(&self) -> Vec<u8> {
        let mut result = self.data.clone();
        if self.bit_pos > 0 {
            result.push(self.current_byte);
        }
        result
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits_basic() {
        let data = [0b10110100, 0b01101001];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(8).unwrap(), 0b01101001);
    }

    #[test]
    fn test_read_bits_across_bytes() {
        let data = [0b10110100, 0b01101001];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(6).unwrap(), 0b101101);
        assert_eq!(reader.read_bits(6).unwrap(), 0b000110);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1001);
    }

    #[test]
    fn test_read_single_bits() {
        let data = [0b10110100];
        let mut reader = BitReader::new(&data);

        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
    }

    #[test]
    fn test_read_unary() {
        // 0001 = 3 zeros, then 1
        let data = [0b00010000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_unary().unwrap(), 3);

        // 1 = 0 zeros
        let data = [0b10000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_unary().unwrap(), 0);

        // 00000001 00000000 = 15 zeros
        let data = [0b00000001, 0b00000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_unary().unwrap(), 7);
    }

    #[test]
    fn test_read_utf8_coded() {
        // Single byte: 0x45
        let data = [0x45];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_utf8_coded().unwrap(), 0x45);

        // Two bytes: 0xC2 0x80 = 0x80
        let data = [0xC2, 0x80];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_utf8_coded().unwrap(), 0x80);

        // Three bytes: 0xE0 0xA0 0x80 = 0x800
        let data = [0xE0, 0xA0, 0x80];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_utf8_coded().unwrap(), 0x800);
    }

    #[test]
    fn test_write_bits_basic() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1011, 4);
        writer.write_bits(0b0100, 4);
        let data = writer.finish();
        assert_eq!(data, vec![0b10110100]);
    }

    #[test]
    fn test_write_bits_across_bytes() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101101, 6);
        writer.write_bits(0b000110, 6);
        writer.write_bits(0b1001, 4);
        let data = writer.finish();
        assert_eq!(data, vec![0b10110100, 0b01101001]);
    }

    #[test]
    fn test_write_unary() {
        let mut writer = BitWriter::new();
        writer.write_unary(3); // 0001
        writer.write_unary(0); // 1
        writer.write_unary(2); // 001
        let data = writer.finish();
        assert_eq!(data, vec![0b00011001]);
    }

    #[test]
    fn test_roundtrip() {
        // Write some data
        let mut writer = BitWriter::new();
        writer.write_bits(0xABCD, 16);
        writer.write_bits(0x12, 8);
        writer.write_unary(5);
        writer.write_bits(0b101, 3);
        let data = writer.finish();

        // Read it back
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(16).unwrap(), 0xABCD);
        assert_eq!(reader.read_bits(8).unwrap(), 0x12);
        assert_eq!(reader.read_unary().unwrap(), 5);
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
    }

    #[test]
    fn test_utf8_roundtrip() {
        for value in [0u64, 1, 127, 128, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x1FFFFF] {
            let mut writer = BitWriter::new();
            writer.write_utf8_coded(value);
            let data = writer.finish();

            let mut reader = BitReader::new(&data);
            assert_eq!(
                reader.read_utf8_coded().unwrap(),
                value,
                "Failed for value {}",
                value
            );
        }
    }

    #[test]
    fn test_signed_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits_signed(-1, 8);
        writer.write_bits_signed(-128, 8);
        writer.write_bits_signed(127, 8);
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits_signed(8).unwrap(), -1);
        assert_eq!(reader.read_bits_signed(8).unwrap(), -128);
        assert_eq!(reader.read_bits_signed(8).unwrap(), 127);
    }
}
