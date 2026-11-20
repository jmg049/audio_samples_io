//! FLAC files: a FLAC decoder and encoder in Rust.
//!
//! [`FlacFile`] reads a whole file, [`StreamedFlacFile`] reads one incrementally, and
//! [`StreamedFlacWriter`] writes one incrementally. [`write_flac`] and
//! [`write_flac_without_md5`] encode a whole signal; [`CompressionLevel`] trades
//! encoding time for size, with levels 0 to 8.
//!
//! The decoder reads 1 to 32 bits per sample, up to 8 channels, every channel
//! assignment (independent, left-side, right-side, mid-side), the constant, verbatim,
//! fixed and LPC subframes (LPC orders up to 32) and both Rice partition types, and
//! checks the CRC-8 and CRC-16 of every frame. It parses the STREAMINFO, SEEKTABLE,
//! VORBIS_COMMENT and PICTURE blocks and skips the others. The encoder writes 16-bit
//! samples for `i16` audio and 24-bit samples for every other type, with fixed or LPC
//! prediction (LPC orders up to 12 at level 8), and a STREAMINFO block that carries
//! the MD5 signature of the audio unless it is written without one.
//!
//! The submodules `metadata`, `frame`, `subframe`, `lpc`, `rice` and `crc` follow the
//! layers of the format.
//!
//! # Example
//!
//! ```no_run
//! use audio_samples_io::flac::FlacFile;
//! use audio_samples_io::traits::{AudioFile, AudioFileRead};
//!
//! let flac = FlacFile::open("audio.flac")?;
//! let samples = flac.read::<f32>()?;
//! # Ok::<(), audio_samples_io::error::AudioIOError>(())
//! ```

pub mod crc;
pub mod data;
pub mod error;
pub mod frame;
pub mod lpc;
pub mod metadata;
pub mod rice;
pub mod subframe;

mod bitstream;
mod constants;
mod encoder;
pub(crate) mod flac_file;
mod md5;
pub(crate) mod parallel;
mod streaming;
mod streaming_writer;

// Re-exports
use core::fmt::{Display, Formatter, Result as FmtResult};

pub use error::FlacError;
pub use flac_file::{FlacFile, FlacFileInfo, write_flac, write_flac_without_md5};
pub use metadata::{MetadataBlock, MetadataBlockType, SeekPoint, SeekTable, StreamInfo, VorbisComment};
pub use streaming::StreamedFlacFile;
pub use streaming_writer::StreamedFlacWriter;

/// FLAC compression levels (0-8, matching reference encoder)
///
/// Higher levels use more CPU for better compression:
/// - Level 0: Fastest, fixed predictors only, small block size
/// - Level 5: Default, good balance of speed and compression
/// - Level 8: Best compression, exhaustive parameter search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CompressionLevel(u8);

impl CompressionLevel {
    /// Fastest compression (level 0)
    pub const FASTEST: Self = Self(0);
    /// Fast compression (level 2)
    pub const FAST: Self = Self(2);
    /// Default compression (level 5)
    pub const DEFAULT: Self = Self(5);
    /// Best compression (level 8)
    pub const BEST: Self = Self(8);

    /// Create a compression level (clamped to 0-8)
    pub const fn new(level: u8) -> Self {
        Self(if level > 8 { 8 } else { level })
    }

    /// Get the numeric level (0-8)
    pub const fn level(self) -> u8 {
        self.0
    }

    /// Get recommended block size for this level
    pub const fn block_size(self) -> u32 {
        match self.0 {
            0..=2 => 1152,
            _ => 4096,
        }
    }

    /// Get maximum LPC order for this level
    pub const fn max_lpc_order(self) -> u8 {
        match self.0 {
            0..=2 => 0, // Fixed predictors only
            3 => 6,
            7 | 8 => 12,
            _ => 8,
        }
    }

    /// Whether to compare the Rice parameters on either side of the one a partition's mean implies
    pub const fn exhaustive_rice_search(self) -> bool {
        self.0 >= 6
    }

    /// Whether to use exhaustive LPC order search
    pub const fn exhaustive_lpc_search(self) -> bool {
        self.0 >= 7
    }

    /// Get Rice partition order range (min, max)
    pub const fn rice_partition_order_range(self) -> (u8, u8) {
        match self.0 {
            0..=2 => (0, 3),
            3 | 4 => (0, 4),
            6..=8 => (0, 6),
            _ => (0, 5),
        }
    }

    /// Whether to try mid-side stereo encoding
    pub const fn try_mid_side(self) -> bool {
        self.0 >= 1 && self.0 != 3
    }

    /// Largest number of parts a block is divided into for partial analysis windows, which
    /// suit signals that change within a block. 1 analyses the whole block only.
    pub const fn window_parts(self) -> u8 {
        match self.0 {
            0..=5 => 1,
            6 | 7 => 2,
            _ => 3,
        }
    }

    /// LPC coefficient precision for this level's block size and 16-bit samples. Wider samples
    /// use up to 15 bits.
    pub const fn qlp_precision(self) -> u8 {
        encoder::auto_qlp_precision(16, self.block_size())
    }
}

impl Display for CompressionLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Level {} ({})",
            self.0,
            match self.0 {
                0 => "fastest",
                1..=2 => "fast",
                3..=4 => "normal",
                5 => "default",
                6..=7 => "high",
                8 => "best",
                _ => "unknown",
            }
        )
    }
}

impl From<u8> for CompressionLevel {
    fn from(level: u8) -> Self {
        Self::new(level)
    }
}

/// Channel assignment modes for stereo encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelAssignment {
    /// Independent channels (no decorrelation)
    Independent(u8),
    /// Left-side stereo (left, left-right)
    LeftSide,
    /// Right-side stereo (left-right, right)  
    RightSide,
    /// Mid-side stereo ((left+right)/2, left-right)
    MidSide,
}

impl ChannelAssignment {
    /// Get the channel assignment code for the frame header
    pub const fn code(self) -> u8 {
        match self {
            ChannelAssignment::Independent(n) => n - 1,
            ChannelAssignment::LeftSide => 0b1000,
            ChannelAssignment::RightSide => 0b1001,
            ChannelAssignment::MidSide => 0b1010,
        }
    }

    /// Parse channel assignment from frame header code
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0..=7 => Some(ChannelAssignment::Independent(code + 1)),
            0b1000 => Some(ChannelAssignment::LeftSide),
            0b1001 => Some(ChannelAssignment::RightSide),
            0b1010 => Some(ChannelAssignment::MidSide),
            _ => None,
        }
    }

    /// Get the number of channels
    pub const fn channels(self) -> u8 {
        match self {
            ChannelAssignment::Independent(n) => n,
            ChannelAssignment::LeftSide | ChannelAssignment::RightSide | ChannelAssignment::MidSide => 2,
        }
    }

    /// Whether this is a stereo decorrelation mode
    pub const fn is_stereo_decorrelated(self) -> bool {
        matches!(
            self,
            ChannelAssignment::LeftSide | ChannelAssignment::RightSide | ChannelAssignment::MidSide
        )
    }
}

impl Display for ChannelAssignment {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ChannelAssignment::Independent(n) => write!(f, "{} independent channel(s)", n),
            ChannelAssignment::LeftSide => write!(f, "left-side stereo"),
            ChannelAssignment::RightSide => write!(f, "right-side stereo"),
            ChannelAssignment::MidSide => write!(f, "mid-side stereo"),
        }
    }
}
