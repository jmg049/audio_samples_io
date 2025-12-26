//! FLAC (Free Lossless Audio Codec) implementation.
//!
//! This module provides a complete pure-Rust FLAC decoder and encoder,
//! implementing the full FLAC specification including:
//!
//! - All metadata block types (STREAMINFO, SEEKTABLE, VORBIS_COMMENT, etc.)
//! - LPC prediction (orders 1-32) and fixed predictors (orders 0-4)
//! - Rice entropy coding with partition optimization
//! - CRC-8/CRC-16 validation
//! - All channel configurations (mono, stereo, surround up to 8 channels)
//! - Stereo decorrelation modes (independent, left-side, right-side, mid-side)
//!
//! # Architecture
//!
//! The module is organized into layers matching the FLAC specification:
//!
//! - `metadata`: Metadata block parsing and serialization
//! - `frame`: Frame-level encoding/decoding with CRC validation
//! - `subframe`: Subframe types (CONSTANT, VERBATIM, FIXED, LPC)
//! - `lpc`: Linear predictive coding with Levinson-Durbin
//! - `rice`: Rice/Rice2 entropy coding for residuals
//! - `crc`: CRC-8 and CRC-16 computation
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
mod flac_file;

// Re-exports
pub use error::FlacError;
pub use flac_file::{FlacFile, FlacFileInfo, write_flac};
pub use metadata::{
    MetadataBlock, MetadataBlockType, SeekPoint, SeekTable, StreamInfo, VorbisComment,
};

use core::fmt::{Display, Formatter, Result as FmtResult};

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
            0 => 1152,
            1 => 1152,
            2 => 2304,
            3 => 2304,
            4 => 4096,
            5 => 4096,
            6 => 4096,
            7 => 4096,
            8 => 4096,
            _ => 4096,
        }
    }

    /// Get maximum LPC order for this level
    pub const fn max_lpc_order(self) -> u8 {
        match self.0 {
            0 => 0, // Fixed predictors only
            1 => 0, // Fixed predictors only
            2 => 0, // Fixed predictors only
            3 => 6,
            4 => 8,
            5 => 8,
            6 => 8,
            7 => 12,
            8 => 12,
            _ => 8,
        }
    }

    /// Whether to use exhaustive Rice parameter search
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
            0 => (0, 3),
            1 => (0, 3),
            2 => (0, 4),
            3 => (0, 4),
            4 => (0, 5),
            5 => (0, 5),
            6 => (0, 6),
            7 => (0, 6),
            8 => (0, 6),
            _ => (0, 5),
        }
    }

    /// Whether to try mid-side stereo encoding
    pub const fn try_mid_side(self) -> bool {
        self.0 >= 1
    }

    /// LPC coefficient quantization precision
    pub const fn qlp_precision(self) -> u8 {
        match self.0 {
            0..=4 => 12,
            5..=7 => 14,
            8 => 15,
            _ => 12,
        }
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
            ChannelAssignment::LeftSide
            | ChannelAssignment::RightSide
            | ChannelAssignment::MidSide => 2,
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
