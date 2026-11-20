//! Bitstream reader and writer for FLAC encoding/decoding.
//!
//! FLAC uses big-endian (MSB-first) bit ordering for all bitstream operations.

use crate::flac::error::FlacError;

/// Decode groups of `N` Rice codes, each group from one 64-bit load, until a group does not fit,
/// the data or `count` runs out.
///
/// # Safety
///
/// `out` must be valid for writes of `count` values.
#[inline(always)]
unsafe fn rice_group<const N: usize>(
    data: &[u8],
    bit: &mut usize,
    done: &mut usize,
    out: *mut i32,
    count: usize,
    k: u32,
) {
    if data.len() < 8 {
        return;
    }
    let last_load = data.len() - 8;
    let mut at = *bit;
    let mut written = *done;
    while written + N <= count && (at >> 3) <= last_load {
        // SAFETY: the loop condition keeps the 8-byte load inside the data.
        let mut word = u64::from_be(unsafe { data.as_ptr().add(at >> 3).cast::<u64>().read_unaligned() }) << (at & 7);
        let mut used = 0u32;
        let mut values = [0u32; N];
        for value in &mut values {
            let zeros = word.leading_zeros();
            // Past the stop bit, then the top k bits. A word of zeros gives nonsense here, which
            // the check on `used` discards.
            word = word.wrapping_shl(zeros) << 1;
            let remainder = (word >> 1) >> (63 - k);
            word = word.checked_shl(k).unwrap_or(0);
            *value = zeros.wrapping_shl(k) | remainder as u32;
            used = used.wrapping_add(zeros.min(64) + 1 + k);
        }
        if used > 57 {
            break;
        }
        for (i, &u) in values.iter().enumerate() {
            // SAFETY: written + N <= count.
            unsafe { out.add(written + i).write(((u >> 1) as i32) ^ -((u & 1) as i32)) };
        }
        written += N;
        at += used as usize;
    }
    *bit = at;
    *done = written;
}

/// Bitstream reader for decoding FLAC data.
///
/// Uses a 64-bit left-aligned register for efficient bulk bit extraction.
/// Each read_bits/read_unary call is a single shift+mask rather than a per-byte loop.
/// Reads bits from a byte slice in big-endian (MSB-first) order.
#[derive(Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the next byte to load into `buf`
    byte_pos: usize,
    /// Left-aligned 64-bit buffer: the next bit to read is at position 63 (MSB)
    buf: u64,
    /// Number of valid bits currently in `buf` (0-64)
    buf_bits: u32,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader from a byte slice.
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            buf: 0,
            buf_bits: 0,
        }
    }

    /// Logical stream position in bits (total bits consumed by caller).
    #[inline]
    const fn stream_bit_pos(&self) -> usize {
        self.byte_pos * 8 - self.buf_bits as usize
    }

    /// Refill the buffer from `data`, loading bytes until buf_bits > 56 or data exhausted.
    ///
    /// Fast path: when ≥ 8 bytes remain, loads all required bytes as a single u64 read
    /// rather than byte by byte.
    #[inline(always)]
    fn refill(&mut self) {
        if self.byte_pos + 8 <= self.data.len() {
            // How many bytes we need to bring buf_bits above 56.
            let to_load = ((64 - self.buf_bits) / 8) as usize; // 1–8
            // Load 8 bytes as a single u64 (single cache-line read).
            let word = u64::from_be_bytes(
                self.data[self.byte_pos..self.byte_pos + 8]
                    .try_into()
                    .expect("slice length checked above"),
            );
            // Keep only the top `to_load * 8` bits of word; mask the rest to zero
            // so phantom stream bytes don't corrupt the buffer.
            let keep_bits = to_load * 8;
            let word = word & (u64::MAX << (64 - keep_bits));
            // Place the kept bits just below the existing valid bits.
            self.buf |= word >> self.buf_bits;
            self.byte_pos += to_load;
            self.buf_bits += keep_bits as u32;
        } else {
            // Slow path: near end of stream — byte-by-byte.
            while self.buf_bits <= 56 {
                if self.byte_pos >= self.data.len() {
                    break;
                }
                self.buf |= (self.data[self.byte_pos] as u64) << (56 - self.buf_bits);
                self.byte_pos += 1;
                self.buf_bits += 8;
            }
        }
    }

    /// Get the number of bytes fully consumed from the stream.
    #[inline]
    pub const fn byte_position(&self) -> usize {
        self.stream_bit_pos() / 8
    }

    /// Get the current bit position within the current byte.
    #[inline]
    pub const fn bit_position(&self) -> u8 {
        (self.stream_bit_pos() % 8) as u8
    }

    /// Get the total number of bits consumed.
    #[inline]
    pub const fn bits_consumed(&self) -> usize {
        self.stream_bit_pos()
    }

    /// Check if the reader is byte-aligned.
    #[inline]
    pub const fn is_byte_aligned(&self) -> bool {
        self.stream_bit_pos().is_multiple_of(8)
    }

    /// Get remaining bytes (only valid when byte-aligned).
    #[inline]
    pub fn remaining_bytes(&self) -> &'a [u8] {
        let pos = self.byte_position().min(self.data.len());
        &self.data[pos..]
    }

    /// Get the total number of bits remaining.
    #[inline]
    pub const fn bits_remaining(&self) -> usize {
        self.data.len() * 8 - self.stream_bit_pos()
    }

    /// Decode `count` Rice codes with parameter `k` into `out` as signed values.
    ///
    /// Each code is taken from a 64-bit word loaded at the code's own byte, so there is no refill
    /// branch, whose outcome varies with the code lengths and would often be mispredicted.
    ///
    /// # Safety
    ///
    /// `out` must be valid for writes of `count` values.
    #[inline]
    pub unsafe fn read_rice_run(&mut self, out: *mut i32, count: usize, k: u8) -> Result<(), FlacError> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("lzcnt") && is_x86_feature_detected!("bmi2") {
            // SAFETY: the features were just detected; the caller's guarantee passes on.
            return unsafe { self.read_rice_run_bmi(out, count, k) };
        }
        // SAFETY: the caller's guarantee passes on.
        unsafe { self.read_rice_run_body(out, count, k) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "lzcnt,bmi1,bmi2")]
    unsafe fn read_rice_run_bmi(&mut self, out: *mut i32, count: usize, k: u8) -> Result<(), FlacError> {
        // SAFETY: the caller's guarantee passes on.
        unsafe { self.read_rice_run_body(out, count, k) }
    }

    #[inline(always)]
    unsafe fn read_rice_run_body(&mut self, out: *mut i32, count: usize, k: u8) -> Result<(), FlacError> {
        let data = self.data;
        let mut bit = self.stream_bit_pos();
        let mut done = 0usize;
        // Codes taken from one load. Four codes of a typical length fit in its 57 usable bits, and
        // when they do not, the check below sends them to the single-code loop.
        match k {
            0..=8 => unsafe { rice_group::<4>(data, &mut bit, &mut done, out, count, u32::from(k)) },
            9..=22 => unsafe { rice_group::<2>(data, &mut bit, &mut done, out, count, u32::from(k)) },
            _ => {},
        }
        let k32 = u32::from(k);
        let last_load = data.len().saturating_sub(8);
        while done < count {
            let byte = bit >> 3;
            if byte <= last_load && data.len() >= 8 {
                // SAFETY: byte + 8 <= data.len().
                let word = u64::from_be(unsafe { data.as_ptr().add(byte).cast::<u64>().read_unaligned() });
                let word = word << (bit & 7);
                let zeros = word.leading_zeros();
                let used = zeros + 1 + k32;
                // After the shift at least 57 bits of the word are real.
                if used <= 57 {
                    // Past the stop bit, then the top k bits. Two shifts, as k may be 0.
                    let remainder = ((word << zeros << 1) >> 1) >> (63 - k32);
                    let unsigned = (zeros << k32) | remainder as u32;
                    // SAFETY: done < count.
                    unsafe { out.add(done).write(((unsigned >> 1) as i32) ^ -((unsigned & 1) as i32)) };
                    bit += used as usize;
                    done += 1;
                    if done + 4 <= count && used <= 24 {
                        // The long code is behind: go back to groups.
                        match k {
                            0..=8 => unsafe { rice_group::<4>(data, &mut bit, &mut done, out, count, k32) },
                            9..=22 => unsafe { rice_group::<2>(data, &mut bit, &mut done, out, count, k32) },
                            _ => {},
                        }
                    }
                    continue;
                }
            }
            // A long code, or the end of the data: the general reader handles both.
            self.seek_to_bit(bit);
            let unsigned = if k == 0 {
                self.read_unary()?
            } else {
                self.read_rice_unsigned(k)?
            };
            // SAFETY: done < count.
            unsafe { out.add(done).write(((unsigned >> 1) as i32) ^ -((unsigned & 1) as i32)) };
            bit = self.stream_bit_pos();
            done += 1;
        }
        self.seek_to_bit(bit);
        Ok(())
    }

    /// Continue reading from the bit `bit` of the data.
    #[inline]
    fn seek_to_bit(&mut self, bit: usize) {
        self.byte_pos = bit >> 3;
        self.buf = 0;
        self.buf_bits = 0;
        let skip = (bit & 7) as u32;
        if skip != 0 {
            self.refill();
            self.buf <<= skip;
            self.buf_bits = self.buf_bits.saturating_sub(skip);
        }
    }

    /// Skip to the next byte boundary.
    #[inline]
    pub fn align_to_byte(&mut self) {
        let bit_offset = (self.stream_bit_pos() % 8) as u32;
        if bit_offset != 0 {
            let skip = 8 - bit_offset;
            if self.buf_bits < skip {
                self.refill();
            }
            if skip <= self.buf_bits {
                self.buf <<= skip;
                self.buf_bits -= skip;
            }
            // If still not enough (truncated stream), next read will return UnexpectedEof
        }
    }

    /// Read a single bit.
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool, FlacError> {
        if self.buf_bits == 0 {
            self.refill();
            if self.buf_bits == 0 {
                return Err(FlacError::UnexpectedEof);
            }
        }
        let bit = self.buf >> 63;
        self.buf <<= 1;
        self.buf_bits -= 1;
        Ok(bit != 0)
    }

    /// Read up to 32 bits as an unsigned value.
    #[inline]
    pub fn read_bits(&mut self, count: u8) -> Result<u32, FlacError> {
        debug_assert!(count <= 32, "Cannot read more than 32 bits at once");

        if count == 0 {
            return Ok(0);
        }

        if self.buf_bits < count as u32 {
            self.refill();
            if self.buf_bits < count as u32 {
                return Err(FlacError::UnexpectedEof);
            }
        }

        let val = (self.buf >> (64 - count)) as u32;
        self.buf <<= count;
        self.buf_bits -= count as u32;
        Ok(val)
    }

    /// Read up to 64 bits as an unsigned value.
    #[inline]
    pub fn read_bits_u64(&mut self, count: u8) -> Result<u64, FlacError> {
        debug_assert!(count <= 64, "Cannot read more than 64 bits at once");

        if count <= 32 {
            return Ok(self.read_bits(count)? as u64);
        }

        let high_bits = count - 32;
        let high = self.read_bits(high_bits)? as u64;
        let low = self.read_bits(32)? as u64;

        Ok((high << 32) | low)
    }

    /// Read a signed value using two's complement.
    #[inline]
    pub fn read_bits_signed(&mut self, count: u8) -> Result<i32, FlacError> {
        let unsigned = self.read_bits(count)?;

        if count > 0 && count < 32 && (unsigned & (1 << (count - 1))) != 0 {
            let mask = !((1u32 << count) - 1);
            Ok((unsigned | mask) as i32)
        } else {
            Ok(unsigned as i32)
        }
    }

    /// Drop `count` buffered bits, `1 <= count <= buf_bits`.
    ///
    /// `count` can be 64: the buffer holds 64 bits after a refill from empty, and a
    /// Rice quotient of 63 zeros or more can end exactly at its last bit. A shift by
    /// 64 is out of range for `u64` (it panics in debug builds and leaves the buffer
    /// unchanged in release builds), so that case clears the buffer.
    #[inline(always)]
    fn consume_buffered(&mut self, count: u32) {
        self.buf = self.buf.checked_shl(count).unwrap_or(0);
        self.buf_bits -= count;
    }

    /// Read a unary coded value (count 0s before the terminating 1).
    ///
    /// For FLAC Rice coding, we count 0s until we hit a 1.
    #[inline]
    pub fn read_unary(&mut self) -> Result<u32, FlacError> {
        let mut count = 0u32;

        loop {
            if self.buf_bits == 0 {
                self.refill();
                if self.buf_bits == 0 {
                    return Err(FlacError::UnexpectedEof);
                }
            }

            // leading_zeros on the full u64, capped to valid bits
            let lz = self.buf.leading_zeros().min(self.buf_bits);

            if lz < self.buf_bits {
                // Terminating 1 is within the buffered bits
                count += lz;
                self.consume_buffered(lz + 1);
                return Ok(count);
            } else {
                // All buf_bits are zeros — consume them and continue
                count += self.buf_bits;
                self.buf = 0;
                self.buf_bits = 0;
            }
        }
    }

    /// Fused Rice-code read: unary quotient || k remainder bits, combined into a
    /// single buffer manipulation in the common case (small quotient, buffer full).
    ///
    /// Equivalent to `(read_unary()? << k) | read_bits(k)?` but avoids an
    /// intermediate buffer update pair in the fast path.  Only call when `k > 0`;
    /// the `param == 0` case must still use `read_unary` directly.
    #[inline(always)]
    pub fn read_rice_unsigned(&mut self, k: u8) -> Result<u32, FlacError> {
        debug_assert!(k > 0);

        // Refill when fewer than 9 valid bits remain (one byte threshold).
        // For k>0 this ensures at least k+1 bits are available, keeping the fast
        // path reachable for typical quotients while refilling only once every
        // ~(57/bits_per_symbol) samples.
        if self.buf_bits <= 8 {
            self.refill();
            if self.buf_bits == 0 {
                return Err(FlacError::UnexpectedEof);
            }
        }

        let lz = self.buf.leading_zeros();

        if lz < self.buf_bits {
            let needed = lz + 1 + k as u32;
            if needed <= self.buf_bits {
                // Fast path: unary quotient + k-bit suffix both fit in the buffer.
                // Extract the k bits that immediately follow the terminating 1.
                let r = ((self.buf << (lz + 1)) >> (64 - k)) as u32;
                self.consume_buffered(needed);
                return Ok((lz << k) | r);
            }
            // Terminator is in buffer but k-bits straddle the refill boundary.
            let q = lz;
            self.consume_buffered(lz + 1);
            self.refill();
            if self.buf_bits < k as u32 {
                return Err(FlacError::UnexpectedEof);
            }
            let r = (self.buf >> (64 - k)) as u32;
            self.buf <<= k;
            self.buf_bits -= k as u32;
            return Ok((q << k) | r);
        }

        // Slow path: all valid bits are zeros (large quotient — rare for typical audio).
        // read_unary handles multi-refill accumulation correctly.
        let q = self.read_unary()?;
        if self.buf_bits < k as u32 {
            self.refill();
            if self.buf_bits < k as u32 {
                return Err(FlacError::UnexpectedEof);
            }
        }
        let r = (self.buf >> (64 - k)) as u32;
        self.buf <<= k;
        self.buf_bits -= k as u32;
        Ok((q << k) | r)
    }

    /// Read a UTF-8 coded value (used for frame/sample numbers in FLAC).
    pub fn read_utf8_coded(&mut self) -> Result<u64, FlacError> {
        let first = self.read_bits(8)? as u8;

        let leading_ones = first.leading_ones() as usize;

        match leading_ones {
            0 => Ok(first as u64),
            2..=6 => {
                let mask = 0x7F >> leading_ones;
                let mut value = (first & mask) as u64;

                for _ in 1..leading_ones {
                    let cont = self.read_bits(8)? as u8;
                    if cont & 0xC0 != 0x80 {
                        return Err(FlacError::InvalidUtf8CodedNumber);
                    }
                    value = (value << 6) | (cont & 0x3F) as u64;
                }

                Ok(value)
            },
            7 => {
                let mut value = 0u64;
                for _ in 0..6 {
                    let cont = self.read_bits(8)? as u8;
                    if cont & 0xC0 != 0x80 {
                        return Err(FlacError::InvalidUtf8CodedNumber);
                    }
                    value = (value << 6) | (cont & 0x3F) as u64;
                }
                Ok(value)
            },
            _ => Err(FlacError::InvalidUtf8CodedNumber),
        }
    }

    /// Read raw bytes (must be byte-aligned).
    pub fn read_bytes(&mut self, count: usize) -> Result<&'a [u8], FlacError> {
        if !self.is_byte_aligned() {
            return Err(FlacError::BitstreamError(
                "read_bytes requires byte alignment".to_string(),
            ));
        }

        // Compute the actual stream byte position (pre-loaded bytes minus buffered bytes)
        let start = self.byte_position();

        if start + count > self.data.len() {
            return Err(FlacError::UnexpectedEof);
        }

        // Discard buffered bytes that overlap with the range we're returning
        let bytes_in_buf = (self.buf_bits / 8) as usize;
        if count <= bytes_in_buf {
            self.buf <<= count * 8;
            self.buf_bits -= (count * 8) as u32;
        } else {
            // Count exceeds what's in the buffer; flush and advance byte_pos
            self.buf = 0;
            self.buf_bits = 0;
            self.byte_pos = start + count;
        }

        Ok(&self.data[start..start + count])
    }

    /// Peek at the next N bits without consuming them.
    pub fn peek_bits(&self, count: u8) -> Result<u32, FlacError> {
        let mut clone = BitReader {
            data: self.data,
            byte_pos: self.byte_pos,
            buf: self.buf,
            buf_bits: self.buf_bits,
        };
        clone.read_bits(count)
    }
}

/// Bitstream writer for encoding FLAC data.
///
/// Uses a 64-bit left-aligned accumulator: new bits are placed MSB-first.
/// Flushing is lazy — 4 bytes are emitted only when `accum_bits >= 32`. This
/// means `write_bits` on the hot rice-encoding path triggers a flush only once
/// per ~3–4 samples.
///
/// Invariant: `accum_bits ∈ [0, 31]` between public calls.
///
/// `bits_written()` is computed on-demand as `data.len() * 8 + accum_bits`
/// rather than tracked incrementally, saving one store per hot-path call.
#[derive(Debug)]
pub struct BitWriter {
    /// Completed output bytes.
    data: Vec<u8>,
    /// Left-aligned pending bits: bit 63 is the next bit to be output.
    accum: u64,
    /// Number of valid (pending) bits in `accum`; always 0–31 between calls.
    accum_bits: u32,
}

impl BitWriter {
    #[inline]
    pub const fn new() -> Self {
        BitWriter {
            data: Vec::new(),
            accum: 0,
            accum_bits: 0,
        }
    }

    #[inline]
    pub fn with_capacity(bytes: usize) -> Self {
        BitWriter {
            data: Vec::with_capacity(bytes + 8),
            accum: 0,
            accum_bits: 0,
        }
    }

    /// Total bits written, including any still pending in the accumulator.
    ///
    /// Computed as `data.len() * 8 + accum_bits` on demand, so `write_bits` keeps
    /// no separate counter.
    #[inline]
    pub const fn bits_written(&self) -> usize {
        self.data.len() * 8 + self.accum_bits as usize
    }

    #[inline]
    pub const fn is_byte_aligned(&self) -> bool {
        self.accum_bits == 0
    }

    /// Returns the number of complete bytes flushed to `data`.
    #[inline]
    pub const fn byte_position(&self) -> usize {
        self.data.len()
    }

    /// Forget everything written, keeping the allocation.
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.accum = 0;
        self.accum_bits = 0;
    }

    /// Make room for `bits` more bits plus the slack that [`Self::write_rice_run`] overwrites.
    #[inline]
    pub fn reserve_bits(&mut self, bits: usize) {
        self.data.reserve(bits / 8 + 32);
    }

    /// Write `residuals` as Rice codes with parameter `k`.
    ///
    /// Each code is OR-ed into a 64-bit word that is stored whole; the output pointer then moves
    /// by the complete bytes in it. There is no branch on whether a flush is due, which codes of
    /// varying length would mispredict.
    ///
    /// # Safety
    ///
    /// The spare capacity must hold every code of `residuals` plus 16 bytes.
    #[inline(always)]
    pub unsafe fn write_rice_run(&mut self, residuals: &[i32], k: u32) {
        self.flush_complete_bytes();
        let mut accum = self.accum;
        let mut pending = self.accum_bits;
        let start = self.data.as_mut_ptr();
        let len = self.data.len();
        // SAFETY: `len` is within the allocation.
        let mut out = unsafe { start.add(len) };
        let mask = (1u64 << k) - 1;
        for &r in residuals {
            let u = u64::from(((r << 1) ^ (r >> 31)) as u32);
            let quotient = (u >> k) as u32;
            let mut total = quotient + 1 + k;
            if total > 56 {
                // A long run of zeros: emit it in pieces no flush can overflow.
                let mut zeros = quotient;
                while zeros > 0 {
                    let run = zeros.min(56);
                    pending += run;
                    // SAFETY: the caller reserved room for every code plus the 8-byte store.
                    unsafe { (out as *mut u64).write_unaligned(accum.to_be()) };
                    let bytes = pending >> 3;
                    // SAFETY: as above.
                    out = unsafe { out.add(bytes as usize) };
                    accum = if bytes >= 8 { 0 } else { accum << (bytes * 8) };
                    pending &= 7;
                    zeros -= run;
                }
                total = 1 + k;
            }
            accum |= ((1u64 << k) | (u & mask)) << (64 - pending - total);
            pending += total;
            // SAFETY: the caller reserved room for every code plus the 8-byte store.
            unsafe { (out as *mut u64).write_unaligned(accum.to_be()) };
            let bytes = pending >> 3;
            // SAFETY: as above.
            out = unsafe { out.add(bytes as usize) };
            accum <<= bytes * 8;
            pending &= 7;
        }
        self.accum = accum;
        self.accum_bits = pending;
        // SAFETY: every byte up to `out` was stored above.
        unsafe { self.data.set_len(out.offset_from(start) as usize) };
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.accum |= (bit as u64) << (63 - self.accum_bits);
        self.accum_bits += 1;
        if self.accum_bits == 32 {
            self.flush_32();
        }
    }

    /// Flush 4 bytes from the high end of the accumulator.
    ///
    /// Called when `accum_bits >= 32`. Writes the top 4 bytes in one store and
    /// shifts the accumulator left by 32. The common case avoids the capacity
    /// check since `with_capacity` pre-reserves enough space.
    #[inline(always)]
    fn flush_32(&mut self) {
        if self.data.len() + 4 > self.data.capacity() {
            self.data.reserve(self.data.capacity().max(64) + 256);
        }
        let tmp = ((self.accum >> 32) as u32).to_be_bytes();
        // SAFETY: capacity is guaranteed above; writing exactly 4 bytes.
        unsafe {
            let dst = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, 4);
            self.data.set_len(self.data.len() + 4);
        }
        self.accum <<= 32;
        self.accum_bits -= 32;
    }

    /// Flush all complete bytes from the accumulator (used by `align_to_byte` / `finish`).
    #[inline(always)]
    fn flush_complete_bytes(&mut self) {
        let full_bytes = (self.accum_bits / 8) as usize;
        if full_bytes > 0 {
            if self.data.len() + full_bytes > self.data.capacity() {
                self.data.reserve(self.data.capacity().max(64) + 64);
            }
            let tmp = self.accum.to_be_bytes();
            // SAFETY: capacity guaranteed; full_bytes ≤ 3 (accum_bits ≤ 31 → ≤ 3 full bytes).
            unsafe {
                let dst = self.data.as_mut_ptr().add(self.data.len());
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, full_bytes);
                self.data.set_len(self.data.len() + full_bytes);
            }
            self.accum <<= full_bytes * 8;
            self.accum_bits %= 8;
        }
    }

    /// Write up to 32 bits, MSB first.
    ///
    /// Core hot path. Single shift-OR into the 64-bit accumulator; flushes 4 bytes
    /// lazily when `accum_bits >= 32` — roughly once per 3–4 rice-coded samples.
    ///
    /// Invariant: `accum_bits ≤ 31` before this call and after. With count ≤ 32,
    /// the shift `64 - accum_bits - count ≥ 1` — no underflow.
    #[inline]
    pub fn write_bits(&mut self, value: u32, count: u8) {
        if count == 0 {
            return;
        }
        let count = count as u32;
        debug_assert!(count <= 32);
        // accum_bits ≤ 31, count ≤ 32 → shift = 64 - accum_bits - count ≥ 1.
        let shift = 64 - self.accum_bits - count;
        let masked = (value as u64) & ((1u64 << count) - 1);
        self.accum |= masked << shift;
        self.accum_bits += count;
        // Lazy flush: emit 4 bytes only when ≥ 32 bits are pending.
        if self.accum_bits >= 32 {
            self.flush_32();
        }
    }

    /// Write up to 64 bits, MSB first.
    #[inline]
    pub fn write_bits_u64(&mut self, value: u64, count: u8) {
        if count <= 32 {
            self.write_bits(value as u32, count);
        } else {
            self.write_bits((value >> 32) as u32, count - 32);
            self.write_bits(value as u32, 32);
        }
    }

    /// Write a signed value in two's complement.
    #[inline]
    pub fn write_bits_signed(&mut self, value: i32, count: u8) {
        let mask = if count < 32 { (1u32 << count) - 1 } else { u32::MAX };
        self.write_bits(value as u32 & mask, count);
    }

    /// Write a unary-coded value: `value` zero bits followed by a one bit.
    ///
    /// For values ≤ 31 this is a single `write_bits(1, value+1)` call.
    /// Larger values are written in 32-bit zero chunks + a final chunk.
    #[inline]
    pub fn write_unary(&mut self, value: u32) {
        // For value ≤ 31: write the integer 1 with (value+1) bits.
        // In binary that's 0…01 with `value` leading zeros. Single call.
        if value < 32 {
            self.write_bits(1, (value + 1) as u8);
        } else {
            // Write chunks of 32 zero bits, then the trailing 1
            let mut remaining = value;
            while remaining >= 32 {
                self.write_bits(0, 32);
                remaining -= 32;
            }
            self.write_bits(1, (remaining + 1) as u8);
        }
    }

    /// Write a UTF-8 coded value (for FLAC frame/sample numbers).
    pub fn write_utf8_coded(&mut self, value: u64) {
        if value < 0x80 {
            self.write_bits(value as u32, 8);
        } else if value < 0x800 {
            self.write_bits((0xC0 | (value >> 6)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x10000 {
            self.write_bits((0xE0 | (value >> 12)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x200000 {
            self.write_bits((0xF0 | (value >> 18)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x4000000 {
            self.write_bits((0xF8 | (value >> 24)) as u32, 8);
            self.write_bits((0x80 | ((value >> 18) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else if value < 0x80000000 {
            self.write_bits((0xFC | (value >> 30)) as u32, 8);
            self.write_bits((0x80 | ((value >> 24) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 18) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        } else {
            self.write_bits(0xFE, 8);
            self.write_bits((0x80 | ((value >> 30) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 24) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 18) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 12) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | ((value >> 6) & 0x3F)) as u32, 8);
            self.write_bits((0x80 | (value & 0x3F)) as u32, 8);
        }
    }

    /// Write raw bytes (requires byte alignment).
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert!(self.is_byte_aligned(), "write_bytes requires byte alignment");
        self.data.extend_from_slice(bytes);
    }

    /// Write `bits` significant bits of packed MSB-first data from `data`.
    ///
    /// When the writer is currently byte-aligned (no pending bits in accumulator),
    /// the full-byte portion is copied directly via `copy_nonoverlapping`, which is
    /// much faster than per-byte `write_bits` calls for large payloads such as
    /// encoded subframe data.
    ///
    /// When not byte-aligned, falls back to per-byte `write_bits` calls (which
    /// still benefit from the lazy 32-bit flush invariant).
    #[inline]
    pub fn write_packed_bits(&mut self, data: &[u8], bits: usize) {
        let full_bytes = bits / 8;
        let remainder = bits % 8;

        // Flush any complete bytes that haven't been emitted yet due to lazy flushing.
        // This is cheap (0–3 bytes) and unlocks the byte-aligned fast path below.
        if self.accum_bits >= 8 {
            self.flush_complete_bytes();
        }

        if self.accum_bits == 0 && full_bytes > 0 {
            // Byte-aligned fast path: direct bulk copy into output buffer.
            if self.data.len() + full_bytes > self.data.capacity() {
                self.data.reserve(full_bytes + 256);
            }
            unsafe {
                let dst = self.data.as_mut_ptr().add(self.data.len());
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, full_bytes);
                self.data.set_len(self.data.len() + full_bytes);
            }
        } else {
            for &byte in &data[..full_bytes] {
                self.write_bits(byte as u32, 8);
            }
        }

        if remainder > 0 {
            self.write_bits((data[full_bytes] >> (8 - remainder)) as u32, remainder as u8);
        }
    }

    /// Pad to byte boundary with zero bits, then flush all complete bytes.
    #[inline]
    pub fn align_to_byte(&mut self) {
        let rem = self.accum_bits % 8;
        if rem != 0 {
            self.accum_bits += 8 - rem;
        }
        // Flush any complete bytes (0–3) that may be pending after lazy-flush accumulation.
        self.flush_complete_bytes();
    }

    /// Return a reference to the completely flushed bytes written so far.
    ///
    /// Requires that the writer is byte-aligned (`accum_bits == 0`), which is
    /// always the case after byte-aligned header/footer writes or `align_to_byte()`.
    #[inline]
    pub fn data(&self) -> &[u8] {
        debug_assert_eq!(self.accum_bits, 0, "data() called with pending bits in accumulator");
        &self.data
    }

    /// Consume the writer and return the output bytes, padding to a byte boundary.
    ///
    /// Flushes any remaining complete bytes from the lazy accumulator before returning.
    #[inline]
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte(); // pads + flushes complete bytes
        self.data
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

    /// A Rice code whose unary part ends exactly at the last bit of a full buffer.
    #[test]
    fn test_unary_ending_at_the_last_buffered_bit() {
        // 63 zeros and the terminating 1 fill the first 8 bytes.
        let mut data = vec![0u8; 7];
        data.push(0b0000_0001);
        data.extend_from_slice(&[0b1010_0000, 0, 0, 0, 0, 0, 0, 0, 0]);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_unary().unwrap(), 63);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1010);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_rice_unsigned(4).unwrap(), (63 << 4) | 0b1010);

        // Quotient and remainder together end at the 64th bit.
        let mut data = vec![0u8; 7];
        data.push(0b0000_1101);
        data.extend_from_slice(&[0b1110_0000, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_rice_unsigned(3).unwrap(), (60 << 3) | 0b101);
        assert_eq!(reader.read_bits(3).unwrap(), 0b111);
    }

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
            assert_eq!(reader.read_utf8_coded().unwrap(), value, "Failed for value {}", value);
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
