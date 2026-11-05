//! MD5, for the signature of the decoded audio in STREAMINFO.

const INIT: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];

macro_rules! step {
    ($f:expr, $a:ident, $b:ident, $c:ident, $d:ident, $m:expr, $k:expr, $s:expr) => {
        $a = $b.wrapping_add($a.wrapping_add($f).wrapping_add($m).wrapping_add($k).rotate_left($s));
    };
}

#[inline(always)]
fn compress(state: &mut [u32; 4], block: &[u8]) {
    debug_assert_eq!(block.len(), 64);
    let words = block.as_ptr().cast::<u32>();
    // SAFETY: `block` holds 64 bytes, and each use reads one of its 16 words.
    let word = |i: usize| u32::from_le(unsafe { words.add(i).read_unaligned() });
    let [mut a, mut b, mut c, mut d] = *state;
    macro_rules! f {
        ($x:ident, $y:ident, $z:ident) => {
            $z ^ ($x & ($y ^ $z))
        };
    }
    macro_rules! g {
        ($x:ident, $y:ident, $z:ident) => {
            ($z & $x).wrapping_add(!$z & $y)
        };
    }
    macro_rules! h {
        ($x:ident, $y:ident, $z:ident) => {
            $x ^ $y ^ $z
        };
    }
    macro_rules! i {
        ($x:ident, $y:ident, $z:ident) => {
            $y ^ ($x | !$z)
        };
    }

    step!(f!(b, c, d), a, b, c, d, word(0), 0xd76a_a478, 7);
    step!(f!(a, b, c), d, a, b, c, word(1), 0xe8c7_b756, 12);
    step!(f!(d, a, b), c, d, a, b, word(2), 0x2420_70db, 17);
    step!(f!(c, d, a), b, c, d, a, word(3), 0xc1bd_ceee, 22);
    step!(f!(b, c, d), a, b, c, d, word(4), 0xf57c_0faf, 7);
    step!(f!(a, b, c), d, a, b, c, word(5), 0x4787_c62a, 12);
    step!(f!(d, a, b), c, d, a, b, word(6), 0xa830_4613, 17);
    step!(f!(c, d, a), b, c, d, a, word(7), 0xfd46_9501, 22);
    step!(f!(b, c, d), a, b, c, d, word(8), 0x6980_98d8, 7);
    step!(f!(a, b, c), d, a, b, c, word(9), 0x8b44_f7af, 12);
    step!(f!(d, a, b), c, d, a, b, word(10), 0xffff_5bb1, 17);
    step!(f!(c, d, a), b, c, d, a, word(11), 0x895c_d7be, 22);
    step!(f!(b, c, d), a, b, c, d, word(12), 0x6b90_1122, 7);
    step!(f!(a, b, c), d, a, b, c, word(13), 0xfd98_7193, 12);
    step!(f!(d, a, b), c, d, a, b, word(14), 0xa679_438e, 17);
    step!(f!(c, d, a), b, c, d, a, word(15), 0x49b4_0821, 22);

    step!(g!(b, c, d), a, b, c, d, word(1), 0xf61e_2562, 5);
    step!(g!(a, b, c), d, a, b, c, word(6), 0xc040_b340, 9);
    step!(g!(d, a, b), c, d, a, b, word(11), 0x265e_5a51, 14);
    step!(g!(c, d, a), b, c, d, a, word(0), 0xe9b6_c7aa, 20);
    step!(g!(b, c, d), a, b, c, d, word(5), 0xd62f_105d, 5);
    step!(g!(a, b, c), d, a, b, c, word(10), 0x0244_1453, 9);
    step!(g!(d, a, b), c, d, a, b, word(15), 0xd8a1_e681, 14);
    step!(g!(c, d, a), b, c, d, a, word(4), 0xe7d3_fbc8, 20);
    step!(g!(b, c, d), a, b, c, d, word(9), 0x21e1_cde6, 5);
    step!(g!(a, b, c), d, a, b, c, word(14), 0xc337_07d6, 9);
    step!(g!(d, a, b), c, d, a, b, word(3), 0xf4d5_0d87, 14);
    step!(g!(c, d, a), b, c, d, a, word(8), 0x455a_14ed, 20);
    step!(g!(b, c, d), a, b, c, d, word(13), 0xa9e3_e905, 5);
    step!(g!(a, b, c), d, a, b, c, word(2), 0xfcef_a3f8, 9);
    step!(g!(d, a, b), c, d, a, b, word(7), 0x676f_02d9, 14);
    step!(g!(c, d, a), b, c, d, a, word(12), 0x8d2a_4c8a, 20);

    step!(h!(b, c, d), a, b, c, d, word(5), 0xfffa_3942, 4);
    step!(h!(a, b, c), d, a, b, c, word(8), 0x8771_f681, 11);
    step!(h!(d, a, b), c, d, a, b, word(11), 0x6d9d_6122, 16);
    step!(h!(c, d, a), b, c, d, a, word(14), 0xfde5_380c, 23);
    step!(h!(b, c, d), a, b, c, d, word(1), 0xa4be_ea44, 4);
    step!(h!(a, b, c), d, a, b, c, word(4), 0x4bde_cfa9, 11);
    step!(h!(d, a, b), c, d, a, b, word(7), 0xf6bb_4b60, 16);
    step!(h!(c, d, a), b, c, d, a, word(10), 0xbebf_bc70, 23);
    step!(h!(b, c, d), a, b, c, d, word(13), 0x289b_7ec6, 4);
    step!(h!(a, b, c), d, a, b, c, word(0), 0xeaa1_27fa, 11);
    step!(h!(d, a, b), c, d, a, b, word(3), 0xd4ef_3085, 16);
    step!(h!(c, d, a), b, c, d, a, word(6), 0x0488_1d05, 23);
    step!(h!(b, c, d), a, b, c, d, word(9), 0xd9d4_d039, 4);
    step!(h!(a, b, c), d, a, b, c, word(12), 0xe6db_99e5, 11);
    step!(h!(d, a, b), c, d, a, b, word(15), 0x1fa2_7cf8, 16);
    step!(h!(c, d, a), b, c, d, a, word(2), 0xc4ac_5665, 23);

    step!(i!(b, c, d), a, b, c, d, word(0), 0xf429_2244, 6);
    step!(i!(a, b, c), d, a, b, c, word(7), 0x432a_ff97, 10);
    step!(i!(d, a, b), c, d, a, b, word(14), 0xab94_23a7, 15);
    step!(i!(c, d, a), b, c, d, a, word(5), 0xfc93_a039, 21);
    step!(i!(b, c, d), a, b, c, d, word(12), 0x655b_59c3, 6);
    step!(i!(a, b, c), d, a, b, c, word(3), 0x8f0c_cc92, 10);
    step!(i!(d, a, b), c, d, a, b, word(10), 0xffef_f47d, 15);
    step!(i!(c, d, a), b, c, d, a, word(1), 0x8584_5dd1, 21);
    step!(i!(b, c, d), a, b, c, d, word(8), 0x6fa8_7e4f, 6);
    step!(i!(a, b, c), d, a, b, c, word(15), 0xfe2c_e6e0, 10);
    step!(i!(d, a, b), c, d, a, b, word(6), 0xa301_4314, 15);
    step!(i!(c, d, a), b, c, d, a, word(13), 0x4e08_11a1, 21);
    step!(i!(b, c, d), a, b, c, d, word(4), 0xf753_7e82, 6);
    step!(i!(a, b, c), d, a, b, c, word(11), 0xbd3a_f235, 10);
    step!(i!(d, a, b), c, d, a, b, word(2), 0x2ad7_d2bb, 15);
    step!(i!(c, d, a), b, c, d, a, word(9), 0xeb86_d391, 21);

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// An MD5 calculation in progress.
#[derive(Debug, Clone)]
pub(crate) struct Md5 {
    state: [u32; 4],
    pending: [u8; 64],
    pending_len: usize,
    total: u64,
}

impl Md5 {
    pub(crate) const fn new() -> Self {
        Self {
            state: INIT,
            pending: [0; 64],
            pending_len: 0,
            total: 0,
        }
    }

    pub(crate) fn update(&mut self, data: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("bmi2") {
            // SAFETY: the feature was just detected.
            unsafe { self.update_bmi2(data) };
            return;
        }
        self.update_body(data);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "bmi2")]
    unsafe fn update_bmi2(&mut self, data: &[u8]) {
        self.update_body(data);
    }

    #[inline(always)]
    fn update_body(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.pending_len > 0 {
            let take = (64 - self.pending_len).min(data.len());
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&data[..take]);
            self.pending_len += take;
            data = &data[take..];
            if self.pending_len < 64 {
                return;
            }
            let block = self.pending;
            compress(&mut self.state, &block);
            self.pending_len = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            compress(&mut self.state, block);
        }
        self.pending[..rest.len()].copy_from_slice(rest);
        self.pending_len = rest.len();
    }

    pub(crate) fn finish(mut self) -> [u8; 16] {
        let bits = self.total.wrapping_mul(8);
        let mut tail = [0u8; 72];
        tail[0] = 0x80;
        let pad = (119 - self.pending_len) % 64 + 1;
        tail[pad..pad + 8].copy_from_slice(&bits.to_le_bytes());
        self.update(&tail[..pad + 8]);
        debug_assert_eq!(self.pending_len, 0);
        let mut out = [0u8; 16];
        for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *bytes = word.to_le_bytes();
        }
        out
    }
}

impl std::io::Write for Md5 {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Md5;

    fn hex(data: &[u8]) -> String {
        let mut md5 = Md5::new();
        md5.update(data);
        md5.finish().iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn test_known_digests() {
        assert_eq!(hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(b"message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
        assert_eq!(
            hex(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    #[test]
    fn test_split_updates_match_whole() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let whole = hex(&data);
        for cut in [1usize, 55, 56, 63, 64, 65, 500] {
            let mut md5 = Md5::new();
            md5.update(&data[..cut]);
            md5.update(&data[cut..]);
            let split: String = md5.finish().iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(split, whole, "cut at {cut}");
        }
    }
}
