//! FLAC-specific error types.

use thiserror::Error;

/// Errors specific to FLAC encoding and decoding.
#[derive(Debug, Clone, Error)]
pub enum FlacError {
    // ========================================================================
    // Stream-level errors
    // ========================================================================
    #[error("Invalid FLAC marker: expected 'fLaC', found {found:?}")]
    InvalidMarker { found: [u8; 4] },

    #[error("Missing STREAMINFO block (must be first metadata block)")]
    MissingStreamInfo,

    #[error("Invalid metadata block type: {0}")]
    InvalidMetadataBlockType(u8),

    #[error("Invalid metadata block size: {size} bytes (max 16MB)")]
    InvalidMetadataBlockSize { size: u32 },

    #[error("STREAMINFO block has invalid size: expected 34 bytes, found {0}")]
    InvalidStreamInfoSize(usize),

    // ========================================================================
    // Frame-level errors
    // ========================================================================
    #[error("Invalid frame sync code: expected 0x3FFE, found 0x{found:04X}")]
    InvalidFrameSync { found: u16 },

    #[error("Reserved block size code in frame header")]
    ReservedBlockSizeCode,

    #[error("Reserved sample rate code in frame header")]
    ReservedSampleRateCode,

    #[error("Reserved bits per sample code in frame header")]
    ReservedBitsPerSampleCode,

    #[error("Invalid channel assignment code: {0}")]
    InvalidChannelAssignment(u8),

    #[error("Frame header CRC-8 mismatch: expected 0x{expected:02X}, computed 0x{computed:02X}")]
    FrameHeaderCrcMismatch { expected: u8, computed: u8 },

    #[error("Frame CRC-16 mismatch: expected 0x{expected:04X}, computed 0x{computed:04X}")]
    FrameCrcMismatch { expected: u16, computed: u16 },

    #[error("Invalid UTF-8 coded number in frame header")]
    InvalidUtf8CodedNumber,

    #[error("Frame/sample number overflow")]
    FrameNumberOverflow,

    // ========================================================================
    // Subframe-level errors
    // ========================================================================
    #[error("Invalid subframe type code: {0}")]
    InvalidSubframeType(u8),

    #[error("Reserved subframe type")]
    ReservedSubframeType,

    #[error("Invalid LPC order: {order} (must be 1-32)")]
    InvalidLpcOrder { order: u8 },

    #[error("Invalid fixed predictor order: {order} (must be 0-4)")]
    InvalidFixedOrder { order: u8 },

    #[error("Invalid QLP coefficient precision: {precision}")]
    InvalidQlpPrecision { precision: u8 },

    #[error("LPC shift is negative or too large: {shift}")]
    InvalidLpcShift { shift: i8 },

    #[error("Wasted bits per sample exceeds sample size")]
    ExcessiveWastedBits,

    // ========================================================================
    // Rice coding errors
    // ========================================================================
    #[error("Invalid Rice partition order: {order} (max {max})")]
    InvalidRicePartitionOrder { order: u8, max: u8 },

    #[error("Invalid Rice parameter: {param} (max 14 for RICE, 30 for RICE2)")]
    InvalidRiceParameter { param: u8 },

    #[error("Rice escape code indicates {bits} bits per sample, exceeds limit")]
    RiceEscapeBitsTooLarge { bits: u8 },

    #[error("Rice partition has more samples than block")]
    RicePartitionOverflow,

    // ========================================================================
    // Encoding errors
    // ========================================================================
    #[error("Block size {size} is invalid (must be 16-65535)")]
    InvalidBlockSize { size: u32 },

    #[error("Sample rate {rate} is invalid (must be 1-655350 Hz)")]
    InvalidSampleRate { rate: u32 },

    #[error("Bits per sample {bits} is invalid (must be 4-32)")]
    InvalidBitsPerSample { bits: u8 },

    #[error("Channel count {channels} is invalid (must be 1-8)")]
    InvalidChannelCount { channels: u8 },

    #[error("Sample value {value} exceeds {bits}-bit range")]
    SampleOverflow { value: i64, bits: u8 },

    #[error("LPC coefficient overflow during encoding")]
    LpcCoefficientOverflow,

    #[error("Residual value overflow")]
    ResidualOverflow,

    // ========================================================================
    // MD5 errors
    // ========================================================================
    #[error("MD5 signature mismatch: file may be corrupted")]
    Md5Mismatch,

    #[error("MD5 signature is all zeros (not computed)")]
    Md5NotComputed,

    // ========================================================================
    // I/O and general errors
    // ========================================================================
    #[error("Unexpected end of stream")]
    UnexpectedEof,

    #[error("Bitstream read error: {0}")]
    BitstreamError(String),

    #[error("Seek table is not sorted")]
    SeekTableNotSorted,

    #[error("Invalid seek point: sample {sample} exceeds total {total}")]
    InvalidSeekPoint { sample: u64, total: u64 },

    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

impl FlacError {
    /// Create an invalid marker error
    pub const fn invalid_marker(found: [u8; 4]) -> Self {
        FlacError::InvalidMarker { found }
    }

    /// Create a frame sync error
    pub const fn invalid_frame_sync(found: u16) -> Self {
        FlacError::InvalidFrameSync { found }
    }

    /// Create a CRC-8 mismatch error
    pub const fn header_crc_mismatch(expected: u8, computed: u8) -> Self {
        FlacError::FrameHeaderCrcMismatch { expected, computed }
    }

    /// Create a CRC-16 mismatch error
    pub const fn frame_crc_mismatch(expected: u16, computed: u16) -> Self {
        FlacError::FrameCrcMismatch { expected, computed }
    }

    /// Create an unsupported feature error
    pub fn unsupported<S: Into<String>>(feature: S) -> Self {
        FlacError::UnsupportedFeature(feature.into())
    }
}
