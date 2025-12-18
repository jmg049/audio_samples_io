pub mod chunks;
pub mod data;
pub mod error;
pub mod fmt;
pub mod streaming;
pub mod streaming_writer;
pub mod wav_file;
use core::fmt::{Display, Formatter, Result as FmtResult};
pub use streaming::{StreamedFrameIter, StreamedWavFile, StreamedWindowIter};
pub use streaming_writer::StreamedWavWriter;
pub use wav_file::WavFile;

use crate::{
    error::{AudioIOError, AudioIOResult},
    wav::error::WavError,
};

/// WAV format codes (wFormatTag)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FormatCode {
    /// PCM (uncompressed)
    Pcm,
    /// IEEE Float
    IeeeFloat,
    /// A-law
    ALaw,
    /// Mu-law
    MuLaw,
    /// WAVE_FORMAT_EXTENSIBLE
    Extensible,
    /// Unknown or unsupported format
    Unknown(u16),
}

impl FormatCode {
    /// Canonical numeric WAV format tag
    pub const fn as_u16(self) -> u16 {
        match self {
            FormatCode::Pcm => 0x0001,
            FormatCode::IeeeFloat => 0x0003,
            FormatCode::ALaw => 0x0006,
            FormatCode::MuLaw => 0x0007,
            FormatCode::Extensible => 0xFFFE,
            FormatCode::Unknown(code) => code,
        }
    }

    pub const fn const_from(code: u16) -> Self {
        match code {
            0x0001 => FormatCode::Pcm,
            0x0003 => FormatCode::IeeeFloat,
            0x0006 => FormatCode::ALaw,
            0x0007 => FormatCode::MuLaw,
            0xFFFE => FormatCode::Extensible,
            other => FormatCode::Unknown(other),
        }
    }

    /// Short symbolic name
    pub const fn as_str(self) -> &'static str {
        match self {
            FormatCode::Pcm => "PCM",
            FormatCode::IeeeFloat => "IEEE_FLOAT",
            FormatCode::ALaw => "A_LAW",
            FormatCode::MuLaw => "MU_LAW",
            FormatCode::Extensible => "EXTENSIBLE",
            FormatCode::Unknown(_) => "UNKNOWN",
        }
    }

    /// Human-readable description
    pub const fn description(self) -> &'static str {
        match self {
            FormatCode::Pcm => "Uncompressed PCM",
            FormatCode::IeeeFloat => "IEEE 32-bit or 64-bit floating point",
            FormatCode::ALaw => "A-law companded PCM",
            FormatCode::MuLaw => "Mu-law companded PCM",
            FormatCode::Extensible => "WAVE_FORMAT_EXTENSIBLE container",
            FormatCode::Unknown(_) => "Unknown or unsupported WAV format",
        }
    }

    /// True if this format uses companding
    pub const fn is_companded(self) -> bool {
        matches!(self, FormatCode::ALaw | FormatCode::MuLaw)
    }

    /// True if this format represents floating-point samples
    pub const fn is_float(self) -> bool {
        matches!(self, FormatCode::IeeeFloat)
    }

    /// True if this is raw integer PCM
    pub const fn is_pcm(self) -> bool {
        matches!(self, FormatCode::Pcm)
    }
}

impl From<u16> for FormatCode {
    fn from(code: u16) -> Self {
        match code {
            0x0001 => FormatCode::Pcm,
            0x0003 => FormatCode::IeeeFloat,
            0x0006 => FormatCode::ALaw,
            0x0007 => FormatCode::MuLaw,
            0xFFFE => FormatCode::Extensible,
            other => FormatCode::Unknown(other),
        }
    }
}

impl From<FormatCode> for u16 {
    fn from(val: FormatCode) -> Self {
        val.as_u16()
    }
}

impl TryFrom<&str> for FormatCode {
    type Error = ();

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "pcm" | "PCM" => Ok(FormatCode::Pcm),
            "float" | "ieee" | "IEEE" => Ok(FormatCode::IeeeFloat),
            "alaw" | "A_LAW" => Ok(FormatCode::ALaw),
            "mulaw" | "MU_LAW" => Ok(FormatCode::MuLaw),
            "ext" | "extensible" => Ok(FormatCode::Extensible),
            _ => Err(()),
        }
    }
}

impl Display for FormatCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if f.alternate() {
            match self {
                FormatCode::Unknown(code) => {
                    write!(f, "{} (0x{:04X})", self.description(), code)
                }
                other => write!(f, "{}", other.description()),
            }
        } else {
            match self {
                FormatCode::Unknown(code) => write!(f, "UNKNOWN(0x{:04X})", code),
                other => write!(f, "{}", other.as_str()),
            }
        }
    }
}

/// Extended format information for WAVE_FORMAT_EXTENSIBLE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtendedFormatInfo {
    /// Valid bits per sample (may be less than container size)
    pub valid_bits_per_sample: u16,
    /// Channel mask indicating speaker positions
    pub channel_mask: u32,
    /// Sub-format GUID (first 16 bits indicate actual format)
    pub sub_format: [u8; 16],
    pub format_code: FormatCode,
}

impl ExtendedFormatInfo {
    /// Canonical GUID tail for WAV sub-formats
    pub const WAV_SUBFORMAT_GUID_TAIL: [u8; 14] = [
        0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71, 0x00, 0x00,
    ];

    pub const fn try_new(
        valid_bits_per_sample: u16,
        channel_mask: u32,
        sub_format: [u8; 16],
    ) -> AudioIOResult<Self> {
        // Catch invalid sub-format GUIDs at const construction time
        match FormatCode::const_from(u16::from_le_bytes([sub_format[0], sub_format[1]])) {
            FormatCode::Unknown(_) => Err(AudioIOError::WavError(WavError::invalid_subformat())),
            format_code => Ok(ExtendedFormatInfo {
                valid_bits_per_sample,
                channel_mask,
                sub_format,
                format_code,
            }),
        }
    }

    /// True if the sub-format GUID matches the WAV extensible schema
    pub fn is_standard_wav_subformat(&self) -> bool {
        self.sub_format[2..] == Self::WAV_SUBFORMAT_GUID_TAIL
    }

    pub const fn bytes_per_container_sample(&self) -> u16 {
        self.valid_bits_per_sample.div_ceil(8)
    }
}
