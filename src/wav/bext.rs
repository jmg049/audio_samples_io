use super::error::WavError;

/// Parsed BEXT chunk — Broadcast Wave Format (BWF) extension metadata.
///
/// Defined in EBU Tech 3285. Required by broadcast and post-production
/// workflows. Contains origin information, a timecode reference, an optional
/// SMPTE UMID, and (in version ≥ 2) EBU R128 loudness measurements.
pub struct BextChunk<'a> {
    bytes: &'a [u8],
}

impl<'a> BextChunk<'a> {
    /// Minimum valid bext chunk size (without `CodingHistory`).
    ///
    /// 256 (Description) + 32 (Originator) + 32 (OriginatorReference)
    /// + 10 (OriginationDate) + 8 (OriginationTime)
    /// + 4 + 4 (TimeReference) + 2 (Version) + 64 (UMID)
    /// + 2+2+2+2+2 (loudness fields) + 180 (Reserved) = 602
    pub const MIN_SIZE: usize = 602;

    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, WavError> {
        if bytes.len() < Self::MIN_SIZE {
            return Err(WavError::chunk_parsing(
                "bext",
                "0",
                format!(
                    "bext chunk must be at least {} bytes, got {}",
                    Self::MIN_SIZE,
                    bytes.len()
                ),
            ));
        }
        Ok(BextChunk { bytes })
    }

    /// Free-form description of the audio content (max 256 chars).
    pub fn description(&self) -> String {
        fixed_str(self.bytes, 0, 256)
    }

    /// Name of the file originator / author (max 32 chars).
    pub fn originator(&self) -> String {
        fixed_str(self.bytes, 256, 32)
    }

    /// Unique originator reference string (max 32 chars).
    pub fn originator_reference(&self) -> String {
        fixed_str(self.bytes, 288, 32)
    }

    /// Date of creation formatted as `"YYYY-MM-DD"`.
    pub fn origination_date(&self) -> String {
        fixed_str(self.bytes, 320, 10)
    }

    /// Time of creation formatted as `"HH:MM:SS"`.
    pub fn origination_time(&self) -> String {
        fixed_str(self.bytes, 330, 8)
    }

    /// 64-bit sample count since midnight at the file's origination sample rate.
    ///
    /// This is the primary timecode reference for synchronisation.
    pub fn time_reference(&self) -> u64 {
        let lo = u32::from_le_bytes([self.bytes[338], self.bytes[339], self.bytes[340], self.bytes[341]]) as u64;
        let hi = u32::from_le_bytes([self.bytes[342], self.bytes[343], self.bytes[344], self.bytes[345]]) as u64;
        lo | (hi << 32)
    }

    /// BWF spec version (0 = original, 1 = added UMID, 2 = added loudness values).
    pub fn version(&self) -> u16 {
        u16::from_le_bytes([self.bytes[346], self.bytes[347]])
    }

    /// SMPTE 330M Unique Material Identifier (64 bytes, raw).
    ///
    /// All-zeros if not set.
    pub fn umid(&self) -> &[u8] {
        &self.bytes[348..412]
    }

    /// Integrated loudness in LUFS×100 (e.g. -2300 = -23.00 LUFS).
    ///
    /// Returns `None` for version < 2 or if the field is 0x7FFF (unset).
    pub fn loudness_value(&self) -> Option<i16> {
        self.loudness_field(412)
    }

    /// Loudness range in LU×100.
    ///
    /// Returns `None` for version < 2 or if the field is 0x7FFF (unset).
    pub fn loudness_range(&self) -> Option<u16> {
        if self.version() < 2 {
            return None;
        }
        let v = u16::from_le_bytes([self.bytes[414], self.bytes[415]]);
        if v == 0x7FFF { None } else { Some(v) }
    }

    /// Maximum true-peak level in dBTP×100.
    ///
    /// Returns `None` for version < 2 or if the field is 0x7FFF (unset).
    pub fn max_true_peak_level(&self) -> Option<i16> {
        self.loudness_field(416)
    }

    /// Maximum momentary loudness in LUFS×100.
    ///
    /// Returns `None` for version < 2 or if the field is 0x7FFF (unset).
    pub fn max_momentary_loudness(&self) -> Option<i16> {
        self.loudness_field(418)
    }

    /// Maximum short-term loudness in LUFS×100.
    ///
    /// Returns `None` for version < 2 or if the field is 0x7FFF (unset).
    pub fn max_short_term_loudness(&self) -> Option<i16> {
        self.loudness_field(420)
    }

    /// Free-form history of coding processes applied to the file.
    ///
    /// Located after the fixed 602-byte header; empty string if absent.
    pub fn coding_history(&self) -> String {
        if self.bytes.len() <= Self::MIN_SIZE {
            return String::new();
        }
        fixed_str(self.bytes, Self::MIN_SIZE, self.bytes.len() - Self::MIN_SIZE)
    }

    /// Read a signed i16 loudness field — `None` if version < 2 or value is 0x7FFF (unset sentinel).
    fn loudness_field(&self, offset: usize) -> Option<i16> {
        if self.version() < 2 {
            return None;
        }
        let v = i16::from_le_bytes([self.bytes[offset], self.bytes[offset + 1]]);
        if v == 0x7FFF_u16 as i16 { None } else { Some(v) }
    }
}

/// Decode a fixed-length, null-terminated field, lossily converting non-UTF-8 bytes.
fn fixed_str(bytes: &[u8], start: usize, len: usize) -> String {
    let field = &bytes[start..start + len];
    let end = field.iter().position(|&b| b == 0).unwrap_or(len);
    String::from_utf8_lossy(&field[..end]).trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_bext(
        description: &str,
        originator: &str,
        originator_ref: &str,
        date: &str,
        time: &str,
        time_ref: u64,
        version: u16,
        coding_history: &str,
    ) -> Vec<u8> {
        let mut buf = vec![0u8; BextChunk::MIN_SIZE];

        let copy_str = |buf: &mut Vec<u8>, s: &str, offset: usize, max: usize| {
            let b = s.as_bytes();
            let n = b.len().min(max);
            buf[offset..offset + n].copy_from_slice(&b[..n]);
        };

        copy_str(&mut buf, description, 0, 256);
        copy_str(&mut buf, originator, 256, 32);
        copy_str(&mut buf, originator_ref, 288, 32);
        copy_str(&mut buf, date, 320, 10);
        copy_str(&mut buf, time, 330, 8);

        let lo = (time_ref & 0xFFFF_FFFF) as u32;
        let hi = (time_ref >> 32) as u32;
        buf[338..342].copy_from_slice(&lo.to_le_bytes());
        buf[342..346].copy_from_slice(&hi.to_le_bytes());

        buf[346..348].copy_from_slice(&version.to_le_bytes());

        // Loudness sentinels (0x7FFF = unset)
        for offset in [412, 414, 416, 418, 420] {
            buf[offset] = 0xFF;
            buf[offset + 1] = 0x7F;
        }

        if !coding_history.is_empty() {
            buf.extend_from_slice(coding_history.as_bytes());
            buf.push(0);
        }

        buf
    }

    #[test]
    fn test_bext_basic_fields() {
        let bytes = make_bext(
            "Test recording",
            "Studio A",
            "REF-001",
            "2024-01-15",
            "10:30:00",
            44100 * 3600,
            1,
            "",
        );
        let chunk = BextChunk::from_bytes(&bytes).expect("valid bext chunk");
        assert_eq!(chunk.description(), "Test recording");
        assert_eq!(chunk.originator(), "Studio A");
        assert_eq!(chunk.originator_reference(), "REF-001");
        assert_eq!(chunk.origination_date(), "2024-01-15");
        assert_eq!(chunk.origination_time(), "10:30:00");
        assert_eq!(chunk.time_reference(), 44100 * 3600);
        assert_eq!(chunk.version(), 1);
    }

    #[test]
    fn test_bext_loudness_unset_for_version_1() {
        let bytes = make_bext("", "", "", "", "", 0, 1, "");
        let chunk = BextChunk::from_bytes(&bytes).expect("valid bext");
        assert!(chunk.loudness_value().is_none());
        assert!(chunk.loudness_range().is_none());
        assert!(chunk.max_true_peak_level().is_none());
        assert!(chunk.max_momentary_loudness().is_none());
        assert!(chunk.max_short_term_loudness().is_none());
    }

    #[test]
    fn test_bext_loudness_version_2() {
        let mut bytes = make_bext("", "", "", "", "", 0, 2, "");
        // -23.00 LUFS → -2300 as i16 LE
        let lufs: i16 = -2300;
        bytes[412..414].copy_from_slice(&lufs.to_le_bytes());
        // Range: 8 LU → 800 as u16
        bytes[414..416].copy_from_slice(&800u16.to_le_bytes());

        let chunk = BextChunk::from_bytes(&bytes).expect("valid bext");
        assert_eq!(chunk.loudness_value(), Some(-2300));
        assert_eq!(chunk.loudness_range(), Some(800));
        // Remaining fields still sentinel (unset)
        assert!(chunk.max_true_peak_level().is_none());
    }

    #[test]
    fn test_bext_coding_history() {
        let bytes = make_bext("", "", "", "", "", 0, 0, "A=PCM,F=44100,W=24,M=stereo");
        let chunk = BextChunk::from_bytes(&bytes).expect("valid bext");
        assert_eq!(chunk.coding_history(), "A=PCM,F=44100,W=24,M=stereo");
    }

    #[test]
    fn test_bext_no_coding_history() {
        let bytes = make_bext("", "", "", "", "", 0, 0, "");
        let chunk = BextChunk::from_bytes(&bytes).expect("valid bext");
        assert_eq!(chunk.coding_history(), "");
    }

    #[test]
    fn test_bext_too_small() {
        assert!(BextChunk::from_bytes(&[0u8; 601]).is_err());
    }

    #[test]
    fn test_bext_time_reference_64bit() {
        let large_ref: u64 = u64::from(u32::MAX) + 100;
        let bytes = make_bext("", "", "", "", "", large_ref, 0, "");
        let chunk = BextChunk::from_bytes(&bytes).expect("valid bext");
        assert_eq!(chunk.time_reference(), large_ref);
    }

    #[test]
    fn test_bext_umid_all_zeros() {
        let bytes = make_bext("", "", "", "", "", 0, 0, "");
        let chunk = BextChunk::from_bytes(&bytes).expect("valid bext");
        assert_eq!(chunk.umid(), &[0u8; 64]);
    }
}
