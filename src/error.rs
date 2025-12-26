#[cfg(feature = "flac")]
use crate::flac::error::FlacError;
#[cfg(feature = "wav")]
use crate::wav::error::WavError;

use core::fmt::{Display, Formatter, Result as FmtResult};
use std::io;
use thiserror::Error;

/// Result type for audio_samples_io operations
#[allow(clippy::result_large_err)]
pub type AudioIOResult<T> = Result<T, AudioIOError>;

/// Comprehensive error type for audio_samples_io operations
#[derive(Debug, Error)]
pub enum AudioIOError {
    /// File I/O errors (file not found, permission denied, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("AudioSamples error: {0}")]
    AudioSamples(#[from] audio_samples::AudioSampleError),

    #[error("Corrupted data at {position}: {description} - {details}")]
    /// Data corruption or format compliance errors
    CorruptedData {
        description: String,
        details: String,
        position: ErrorPosition,
    },

    #[cfg(feature = "wav")]
    #[error("Wav error: {0}")]
    WavError(#[from] WavError),

    #[cfg(feature = "flac")]
    #[error("FLAC error: {0}")]
    FlacError(#[from] FlacError),

    #[error("Seek error: {0}")]
    SeekError(String),

    #[error("End of stream: {0}")]
    EndOfStream(String),

    #[error("Missing feature: {0}")]
    MissingFeature(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// Position information for errors that occur during parsing
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ErrorPosition {
    /// Byte offset in the file where the error occurred
    pub offset: usize,
    /// Human-readable description of the position
    pub description: String,
}

impl ErrorPosition {
    /// Create a new error position at the given byte offset
    pub fn new(offset: usize) -> Self {
        Self {
            offset,
            description: format!("byte offset {}", offset),
        }
    }

    /// Set a custom description for the error position
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl Display for ErrorPosition {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.description)
    }
}

impl AudioIOError {
    /// Create a CorruptedData error with position information
    pub fn corrupted_data(
        description: impl Into<String>,
        details: impl Into<String>,
        position: ErrorPosition,
    ) -> Self {
        AudioIOError::CorruptedData {
            description: description.into(),
            details: details.into(),
            position,
        }
    }

    /// Create a CorruptedData error without position information (uses default position)
    pub fn corrupted_data_simple(
        description: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        AudioIOError::CorruptedData {
            description: description.into(),
            details: details.into(),
            position: ErrorPosition::default(),
        }
    }

    /// Create a SeekError with a custom message
    pub fn seek_error(message: impl Into<String>) -> Self {
        AudioIOError::SeekError(message.into())
    }

    /// Create an EndOfStream error with a custom message
    pub fn end_of_stream(message: impl Into<String>) -> Self {
        AudioIOError::EndOfStream(message.into())
    }

    /// Create a MissingFeature error with a custom message
    pub fn missing_feature(message: impl Into<String>) -> Self {
        AudioIOError::MissingFeature(message.into())
    }

    /// Create an UnsupportedFormat error with a custom message
    pub fn unsupported_format(message: impl Into<String>) -> Self {
        AudioIOError::UnsupportedFormat(message.into())
    }
}
